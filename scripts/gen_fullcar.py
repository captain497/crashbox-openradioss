#!/usr/bin/env python3
"""Simplified full car (shell body) into a fixed rigid wall, with the FE crash boxes, an occupant, belt and airbag.

Global axes: x forward (towards the wall), y left, z up. The bumper front face is at x = 0, the wall at x = +2 mm.
Deformable front end: bumper beam (steel), two crash boxes (full shell model from gen_crashbox.py, any core),
front rails (steel), bonnet, nose panel and fenders (steel sheet). Firewall, floor, doors, roof, pillars, boot and wheels
form one rigid cabin body that carries the rest of the car mass. Occupant (rigid 40 kg block), two belt springs and a
pre-inflated closed cushion are inside the cabin, as in gen_airbag.py. Everything starts at the impact speed.

This is NOT a validated vehicle model: geometry, gauges and masses are ASSUMED values for a small car.
Units g, mm, ms (N, MPa, mJ).
"""
import argparse, json, math, os, subprocess, sys, tempfile
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--name", default="FULLCAR"); p.add_argument("--out", default=".")
p.add_argument("--box_args", default="--core none", help="arguments passed to gen_crashbox.py for the crash box design")
p.add_argument("--box_esize", type=float, default=4.0)
p.add_argument("--car_kg", type=float, default=1114.0)      # 130 kJ at 55 km/h
p.add_argument("--v", type=float, default=15.28)
p.add_argument("--rail_t", type=float, default=2.0); p.add_argument("--bumper_t", type=float, default=2.0)
p.add_argument("--p0_kPa", type=float, default=30.0); p.add_argument("--occ_kg", type=float, default=40.0)
p.add_argument("--tend", type=float, default=100.0); p.add_argument("--dt_cst", type=float, default=2.5e-4)
a = p.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
f20 = lambda *v: "".join(f"{x:20.10g}" for x in v)
i10 = lambda *v: "".join(f"{int(x):10d}" for x in v)
B10, B20 = " " * 10, " " * 20
idl = lambda ids: "\n".join(i10(*ids[k:k + 10]) for k in range(0, len(ids), 10))

# ---------------- node and shell store ----------------
coords = []; key2id = {}; shells = {}   # part -> list of quads (node ids, 1-based)
def node(x, y, z, merge=True):
    k = (round(x, 2), round(y, 2), round(z, 2))
    if merge and k in key2id: return key2id[k]
    coords.append((x, y, z)); key2id.setdefault(k, len(coords)); return len(coords)
def grid(P, nu, nv, part, merge=True, close_u=False):
    """Quad mesh of the surface P(u, v), u, v in [0, 1]; returns node id array [nu+1, nv+1]."""
    ids = np.zeros((nu + 1, nv + 1), int)
    for i in range(nu + 1):
        for j in range(nv + 1):
            if close_u and i == nu: ids[i, j] = ids[0, j]; continue
            ids[i, j] = node(*P(i / nu, j / nv), merge=merge)
    q = shells.setdefault(part, [])
    for i in range(nu):
        for j in range(nv):
            q.append((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]))
    return ids
def tube_x(x0, x1, yc, zc, wy, hz, h, part):
    """Rectangular tube along x (rail): perimeter in the y-z plane."""
    per = []
    for (ya, za), (yb, zb) in zip([(-1, -1), (1, -1), (1, 1), (-1, 1)], [(1, -1), (1, 1), (-1, 1), (-1, -1)]):
        L = abs(yb - ya) * wy / 2 + abs(zb - za) * hz / 2; n = max(2, round(L / h))
        per += [(yc + (ya + (yb - ya) * k / n) * wy / 2, zc + (za + (zb - za) * k / n) * hz / 2) for k in range(n)]
    nx = max(2, round(abs(x1 - x0) / h)); q = shells.setdefault(part, []); ring = []
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx; ring.append([node(x, y, z) for y, z in per])
    for i in range(nx):
        for k in range(len(per)):
            k2 = (k + 1) % len(per); q.append((ring[i][k], ring[i][k2], ring[i + 1][k2], ring[i + 1][k]))
    return ring

# ---------------- crash boxes from the crash box generator ----------------
tmp = tempfile.mkdtemp()
out = subprocess.run([sys.executable, os.path.join(here, "gen_crashbox.py"), "--name", "BOX", "--out", tmp, "--anim", "0",
                      "--esize", str(a.box_esize)] + a.box_args.split(), capture_output=True, text=True, check=True)
binfo = json.load(open(os.path.join(tmp, "BOX_info.json")))
bn, bsh, sec = {}, {}, None
for line in open(os.path.join(tmp, "BOX_0000.rad")):
    if line.startswith("/"): sec = line.strip(); continue
    if line.startswith("#") or not line.strip(): continue
    if sec == "/NODE": bn[int(line[:10])] = (float(line[10:30]), float(line[30:50]), float(line[50:70]))
    elif sec in ("/SHELL/1", "/SHELL/2", "/SHELL/3"):
        v = [int(line[k:k + 10]) for k in range(0, 50, 10)]; bsh.setdefault(sec[-1], []).append(v[1:])
mat_line = open(os.path.join(tmp, "BOX_0000.rad")).read().split("/MAT/PLAS_JOHNS/1\n")[1].split("\n/")[0]
LB, WB = binfo["length"], binfo["width"]
X_BOX_FRONT = -61.0; YC = 450.0; ZC = 500.0           # boxes sit behind the bumper, on the rail line
box_front, box_rear = {}, {}
for side, ys in (("L", YC), ("R", -YC)):
    loc = {}
    for nid_, (x, y, z) in bn.items():
        if nid_ not in {n for q in sum(bsh.values(), []) for n in q}: continue
        loc[nid_] = node(X_BOX_FRONT - (LB - z), ys + x, ZC + y, merge=False)
    for part_k, part in (("1", 1), ("2", 2), ("3", 2)):
        for q in bsh.get(part_k, []): shells.setdefault(part, []).append(tuple(loc[n] for n in q))
    box_front[side] = [loc[n] for n, (x, y, z) in bn.items() if n in loc and abs(z - LB) < 1e-6]
    box_rear[side] = [loc[n] for n, (x, y, z) in bn.items() if n in loc and abs(z) < 1e-6]
# ---------------- bumper beam (closed box section along y) ----------------
bum = []
for (xa, za), (xb, zb) in (((0, 450), (-60, 450)), ((-60, 450), (-60, 550)), ((-60, 550), (0, 550)), ((0, 550), (0, 450))):
    n = max(2, round(math.hypot(xb - xa, zb - za) / 10.0))
    bum.append(grid(lambda u, v, xa=xa, za=za, xb=xb, zb=zb: (xa + (xb - xa) * u, -650 + 1300 * v, za + (zb - za) * u), n, 130, 5))
bumper_back = sorted({int(i) for i in bum[1].ravel()})
# ---------------- rails ----------------
rails = {s: tube_x(X_BOX_FRONT - LB - 1.0, -860.0, ys, ZC, 80.0, 100.0, 10.0, 6) for s, ys in (("L", YC), ("R", -YC))}
# ---------------- nose, bonnet, fenders (thin steel) ----------------
grid(lambda u, v: (-30.0, -620 + 1240 * v, 550 + 230 * u), 11, 62, 7)                       # nose panel
bon = grid(lambda u, v: (-30.0 - 830 * u, -620 + 1240 * v, 780 + 50 * u), 42, 62, 7)        # bonnet
for ys in (625.0, -625.0):
    grid(lambda u, v, ys=ys: (-30.0 - 830 * u, ys, 560 + (215 + 50 * u) * v), 42, 11, 7)   # fenders (free front edge)
# ---------------- rigid cabin (visual body + mass carrier) ----------------
CAB = 9
grid(lambda u, v: (-860.0, -700 + 1400 * u, 200 + 700 * v), 28, 14, CAB)                   # firewall / dash panel
grid(lambda u, v: (-860 - 2700 * u, -700 + 1400 * v, 200.0), 54, 28, CAB)                   # floor
for ys in (700.0, -700.0):
    grid(lambda u, v, ys=ys: (-860 - 2700 * u, ys, 200 + 700 * v), 54, 14, 10 if ys > 0 else CAB)   # door/side panels (left side = part 10)
    grid(lambda u, v, ys=ys: (-1000 - 700 * u - 80 * v, ys * (1 - 0.08 * u), 900 + 500 * u), 14, 2, CAB)       # A-pillar
    grid(lambda u, v, ys=ys: (-2350 - 0 * u + 60 * v, ys * 0.92, 900 + 500 * u), 10, 1, CAB)          # B-pillar
    grid(lambda u, v, ys=ys: (-3000 - 500 * u + 80 * v, ys * (0.92 + 0.08 * u), 1400 - 450 * u), 10, 2, CAB)   # C-pillar
grid(lambda u, v: (-1700 - 1300 * u, -644 + 1288 * v, 1400.0), 26, 26, CAB)                # roof
grid(lambda u, v: (-1000 - 700 * u, -640 + 1280 * v, 900 + 500 * u), 17, 26, 11)          # windscreen (glass, part 11)
grid(lambda u, v: (-3500 - 300 * u, -700 + 1400 * v, 950.0), 6, 28, CAB)                   # boot lid
grid(lambda u, v: (-3800.0, -700 + 1400 * u, 200 + 750 * v), 28, 15, CAB)                  # rear panel
grid(lambda u, v: (-860 - 140 * u, -640 + 1280 * v, 830 + 70 * u), 3, 26, CAB)             # cowl
for xw in (-700.0, -3200.0):
    for ys in (560.0, -560.0):
        sg = 1 if ys > 0 else -1
        grid(lambda u, v, xw=xw, ys=ys, sg=sg: (xw + 300 * math.cos(2 * math.pi * u), ys + sg * 180 * v,
                                                 300 + 300 * math.sin(2 * math.pi * u)), 24, 3, 12, close_u=True)   # tyre tread
        for yy in (ys, ys + sg * 180):
            grid(lambda u, v, xw=xw, yy=yy: (xw + (120 + 180 * v) * math.cos(2 * math.pi * u), yy,
                                             300 + (120 + 180 * v) * math.sin(2 * math.pi * u)), 24, 2, 12, close_u=True)   # tyre walls
# ---------------- occupant, belt, cushion ----------------
OCC_X, OCC_Y = -1450.0, 350.0
def block(x0, x1, y0, y1, z0, z1, h, part):
    nx, ny, nz = (max(1, round((x1 - x0) / h)), max(1, round((y1 - y0) / h)), max(1, round((z1 - z0) / h)))
    for (fx, P) in ((nx, lambda u, v: (x0 + (x1 - x0) * u, y0 + (y1 - y0) * v, z0)), (nx, lambda u, v: (x0 + (x1 - x0) * u, y1 - (y1 - y0) * v, z1)),
                    (nx, lambda u, v: (x0 + (x1 - x0) * u, y0, z1 - (z1 - z0) * v)), (nx, lambda u, v: (x0 + (x1 - x0) * u, y1, z0 + (z1 - z0) * v)),
                    (ny, lambda u, v: (x0, y0 + (y1 - y0) * u, z0 + (z1 - z0) * v)), (ny, lambda u, v: (x1, y1 - (y1 - y0) * u, z0 + (z1 - z0) * v))):
        grid(P, fx, nz if P(0, 1)[2] != P(0, 0)[2] else ny, part)
block(OCC_X - 220, OCC_X, OCC_Y - 180, OCC_Y + 180, 450, 1050, 40, 13)
R = 240.0; bcx, bcz = OCC_X + 100 + R, 780.0
bag_ids = []
n16 = 16
for face in range(6):
    def P(u, v, face=face):
        s, t = 2 * u - 1, 2 * v - 1
        c = [(s, t, -1), (t, s, 1), (s, -1, t), (t, 1, s), (-1, t, s), (1, s, t)][face]
        r = math.sqrt(sum(cc * cc for cc in c)); return (bcx + R * c[0] / r, OCC_Y + R * c[1] / r, bcz + R * c[2] / r)
    bag_ids.append(grid(P, n16, n16, 14))
bag_nodes = sorted({int(i) for g in bag_ids for i in g.ravel()})
bag_fixed = [n for n in bag_nodes if coords[n - 1][0] > bcx + 0.85 * R]
# rigid-body main nodes and belt anchors
def free_node(x, y, z): coords.append((x, y, z)); return len(coords)
occM = free_node(OCC_X - 110, OCC_Y, 750.0)
bL = free_node(OCC_X - 220, OCC_Y - 150, 750.0); bR = free_node(OCC_X - 220, OCC_Y + 150, 750.0)
aL = free_node(OCC_X - 600, OCC_Y - 150, 750.0); aR = free_node(OCC_X - 600, OCC_Y + 150, 750.0)
cabM = free_node(-1900.0, 0.0, 500.0)
mounts = {s: free_node(X_BOX_FRONT, ys, ZC) for s, ys in (("L", YC), ("R", -YC))}
rmounts = {s: free_node(X_BOX_FRONT - LB - 0.5, ys, ZC) for s, ys in (("L", YC), ("R", -YC))}

# rigid sets
nodes_of = lambda parts: sorted({n for pp in parts for q in shells.get(pp, []) for n in q})
cab_nodes = set(nodes_of([CAB, 10, 11, 12])) | set(bag_fixed) | {aL, aR}
for s in rails: cab_nodes |= set(rails[s][-1])                          # rail rear ends welded to the firewall
cab_nodes |= {int(i) for i in bon[-1, :]}                                # bonnet hinge line
f7 = set(nodes_of([7]))
for n_, (x, y, z) in enumerate(coords, 1):
    if abs(x + 860) < 1e-6 and n_ in f7: cab_nodes.add(n_)   # fender rear edges
occ_nodes = set(nodes_of([13])) | {bL, bR}
front_mount = {s: set(box_front[s]) | {n for n in bumper_back if abs(coords[n - 1][1] - (YC if s == "L" else -YC)) <= WB / 2 + 5
                                        and abs(coords[n - 1][2] - ZC) <= 50 + 1e-6} for s in ("L", "R")}
rear_mount = {s: set(box_rear[s]) | set(rails[s][0]) for s in ("L", "R")}

# ---------------- masses ----------------
rho_al = float(mat_line.split("\n")[1]); rho_st = 7.85e-3
def area(q):
    P_ = [np.array(coords[n - 1]) for n in q]; return 0.5 * np.linalg.norm(np.cross(P_[2] - P_[0], P_[3] - P_[1]))
thick = {1: binfo["t"], 2: binfo["tc"], 5: a.bumper_t, 6: a.rail_t, 7: 0.8, CAB: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 0.35}
rho_p = {1: rho_al, 2: rho_al, 5: rho_st, 6: rho_st, 7: rho_st, CAB: 1e-4, 10: 1e-4, 11: 1e-4, 12: 1e-4, 13: 1e-3, 14: 8e-4}
pmass = {pp: sum(area(q) for q in shells.get(pp, [])) * thick[pp] * rho_p[pp] for pp in shells}
m_target = a.car_kg * 1000.0
cab_mass = m_target - sum(pmass.values()) - a.occ_kg * 1000.0 - 4 * 500.0
info = dict(vars(a), box=binfo, part_mass_g=pmass, cabin_rb_mass_g=cab_mass, n_nodes=len(coords),
            n_shells={k: len(v) for k, v in shells.items()}, occ_main=occM, cab_main=cabM)

# ---------------- deck ----------------
d = ["#RADIOSS STARTER\n/BEGIN\n" + f"{a.name:<80s}\n" + i10(2024) + "\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}",
     "/TITLE\nSimplified car with FE crash boxes, occupant and airbag into a rigid wall",
     "/NODE\n" + "\n".join(i10(k) + f20(*c) for k, c in enumerate(coords, 1))]
d.append("/MAT/PLAS_JOHNS/1\n" + mat_line)
d.append("/MAT/PLAS_JOHNS/2\nMILD_STEEL\n" + f20(rho_st) + "\n" + f20(200000.0, 0.28) + i10(0) + "\n" + f20(265.0, 339.4, 0.541, 0.0, 0.0) + "\n" +
         f20(0.0, 1.0e-3) + i10(1, 1) + f20(10.0, 0.0) + "\n" + f20(0, 0, 0, 0))
d.append("/MAT/ELAST/3\nRIGID_BODY_SKIN\n" + f20(1e-4) + "\n" + f20(1000.0, 0.3))
d.append("/MAT/ELAST/4\nAIRBAG_FABRIC\n" + f20(8.0e-4) + "\n" + f20(380.0, 0.35))
d.append("/MAT/ELAST/5\nOCCUPANT_SKIN\n" + f20(1.0e-3) + "\n" + f20(1000.0, 0.3))
def prop(pid, title, t, nip=5, ish=24):
    return (f"/PROP/SHELL/{pid}\n{title}\n" + i10(ish, 0, 0, 0, 0) + B10 + f20(0) + "\n" + f20(0, 0, 0, 0, 0) + "\n" + i10(nip) + B10 + f20(t, 0) + B10 + i10(1, 1, 0))
pdef = {1: ("BOX_TUBE", 1, 5), 2: ("BOX_CORE", 1, 5), 5: ("BUMPER", 2, 5), 6: ("RAILS", 2, 5), 7: ("BONNET_FENDERS", 2, 3),
        CAB: ("CABIN_RIGID", 3, 1), 10: ("LEFT_DOORS_RIGID", 3, 1), 11: ("WINDSCREEN_RIGID", 3, 1), 12: ("WHEELS_RIGID", 3, 1),
        13: ("OCCUPANT", 5, 1), 14: ("AIRBAG", 4, 1)}
eid = 0; shell_ids = {}
for pp in sorted(shells):
    if pp == 2 and not shells[2]: continue
    title, mid, nip = pdef[pp]
    d.append(prop(pp, title, thick[pp], nip)); d.append(f"/PART/{pp}\n{title}\n" + i10(pp, mid, 0))
    lines = []
    for q in shells[pp]:
        eid += 1; lines.append(i10(eid, *q)); shell_ids.setdefault(pp, []).append(eid)
    d.append(f"/SHELL/{pp}\n" + "\n".join(lines))
# rigid bodies
def rbody(k, title, main, mass, nodes_, J):
    d.append(f"/GRNOD/NODE/{100 + k}\n{title}_NODES\n" + idl(sorted(nodes_)))
    d.append(f"/RBODY/{k}\n{title}\n" + i10(main, 0, 0, 0) + f20(mass) + i10(100 + k, 0, 0, 0) + "\n" + f20(J, J, J) + "\n" + f20(0, 0, 0) + "\n" + i10(0, 0, 0))
rbody(1, "CABIN", cabM, cab_mass, cab_nodes, 1.5e12)
rbody(2, "OCCUPANT", occM, a.occ_kg * 1000.0, occ_nodes, 4.0e8)
for k, s in enumerate(("L", "R")):
    rbody(3 + k, f"BOX_FRONT_PLATE_{s}", mounts[s], 500.0, front_mount[s], 1e6)
    rbody(5 + k, f"BOX_REAR_PLATE_{s}", rmounts[s], 500.0, rear_mount[s], 1e6)
d.append("/GRNOD/NODE/20\nOCC_MAIN\n" + i10(occM)); d.append("/BCS/20\nOCC_PLANE\n   010 101" + i10(0, 20))
# belt
F = 3000.0; kb = 100.0
d.append("/FUNCT/21\nBELT_FORCE\n" + "\n".join(f20(x, y) for x, y in ((-5000, 0), (0, 0), (F / kb, F), (5000, F))))
d.append("/PROP/TYPE4/21\nBELT\n" + f20(1.0) + B20 + i10(0, 0, 0) + "\n" + f20(kb, 0.2 * 2 * math.sqrt(kb * 1.0), 1, 0, 1) + "\n" +
         i10(21, 1, 0, 0, 0) + B20 + f20(0, 0) + "\n" + f20(0, 0, 0, 0))
d.append("/PART/21\nBELT\n" + i10(21, 0, 0)); d.append("/SPRING/21\n" + i10(900001, aL, bL) + "\n" + i10(900002, aR, bR))
# cushion
pext = 0.101325; p0 = pext + a.p0_kPa / 1000.0
d.append("/SURF/PART/16\nBAG_SURF\n" + i10(14)); d.append("/SURF/PART/17\nOCC_SURF\n" + i10(13))
d.append("/GRNOD/PART/18\nBAG_NODES\n" + i10(14)); d.append("/GRNOD/PART/19\nOCC_NODES\n" + i10(13))
d.append("/MONVOL/GAS/1\nCUSHION\n" + i10(16, 0) + "\n" + f20(0, 0, 0, 0, 0) + "\n" + f20(1.4, 0.01, 0, 295.0, p0 * 1e6 / (287.0 * 295.0) * 1e-6) + "\n" +
         f20(pext, p0, 0, 0, 0) + "\n" + i10(0))
# contacts
def type7(k, title, sec, mst, gap, fric, inacti=0):
    d.append(f"/INTER/TYPE7/{k}\n{title}\n" + i10(sec, mst, 4, 0, 0) + B10 + i10(0, 0, 0, 0) + "\n" + f20(0, 0, 0) + B20 + i10(0) + "\n" +
             f20(0, 0, 0, 0) + i10(0, 0) + "\n" + f20(1, fric, gap, 0, 0) + "\n" + "       000" + B20 + i10(inacti) + f20(0, 0, 0) + "\n" +
             i10(0, 0) + f20(0) + i10(0, 0, 0) + f20(0) + i10(0))
front_parts = [pp for pp in (1, 2, 5, 6, 7) if shells.get(pp)]
d.append("/GRNOD/PART/30\nFRONT_NODES\n" + i10(*front_parts)); d.append("/SURF/PART/31\nFRONT_SURF\n" + i10(*front_parts))
type7(1, "FRONT_SELF", 30, 31, 0.5 * min(thick[pp] for pp in front_parts), 0.15, inacti=1)
type7(2, "OCC_ON_BAG", 19, 16, 2.0, 0.3); type7(3, "BAG_ON_OCC", 18, 17, 2.0, 0.3)
d.append("/GRNOD/PART/32\nALL_PART_NODES\n" + i10(*sorted(shells)))
d.append("/RWALL/PLANE/1\nWALL\n" + i10(0, 0, 32, 0) + "\n" + f20(0, 0.2, 0) + "\n" + f20(2.0, 0.0, 0.0) + "\n" + f20(1.0, 0.0, 0.0))
# initial speed for every node
d.append("/GRNOD/NODE/33\nFREE_NODES\n" + idl([occM, bL, bR, aL, aR, cabM, *mounts.values(), *rmounts.values()]))
d.append("/INIVEL/TRA/1\nSPEED_PARTS\n" + f20(a.v, 0, 0) + i10(32, 0)); d.append("/INIVEL/TRA/2\nSPEED_NODES\n" + f20(a.v, 0, 0) + i10(33, 0))
# output
d.append("/TH/RWALL/1\nWALL\nDEF\n" + i10(1))
d.append("/TH/NODE/2\nCAB_OCC\nDEF\n" + i10(cabM, 0) + "\n" + i10(occM, 0))
d.append("/TH/SPRING/3\nBELT\nDEF\n" + i10(900001, 900002))
d.append("/TH/MONV/4\nCUSHION\nDEF\n" + i10(1))
d.append("/TH/PART/5\nPART_ENERGY\nDEF\n" + "\n".join(i10(k) for k in front_parts))
def_parts = [1, 2, 5, 6, 7]   # deformable shells: crash box wall/core, bumper, rails, front panels
shl = [e for pp in def_parts for e in shell_ids.get(pp, [])]
d.append("/BTH/SHEL/10\nANIM_SHELLS\nPLAS      STRESS\n" + "\n".join(i10(e) for e in shl))
d.append("/ATH/NODE/9\nANIM_NODES\nDX        DY        DZ\n" + "\n".join(i10(k, 0) for k in range(1, len(coords) + 1)))
d.append("/END")
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, f"{a.name}_0000.rad"), "w").write("\n".join(d) + "\n")
open(os.path.join(a.out, f"{a.name}_0001.rad"), "w").write(
    f"/RUN/{a.name}/1\n{a.tend}\n/VERS/2024\n/TFILE/4\n0.02\n/ATFILE/4\n{a.tend / 50:.3f}\n/BTFILE/4\n{a.tend / 50:.3f}\n/PRINT/-5000\n/DT/NODA/CST/0\n0.9 {a.dt_cst}\n/RFILE/0\n10000000\n")
json.dump(info, open(os.path.join(a.out, f"{a.name}_info.json"), "w"), indent=1, default=float)
print(json.dumps(dict(nodes=len(coords), shells={k: len(v) for k, v in shells.items()}, cab_mass_kg=cab_mass / 1000,
                      part_mass_kg={k: round(v / 1000, 3) for k, v in pmass.items()})))

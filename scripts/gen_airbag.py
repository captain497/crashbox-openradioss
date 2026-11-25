#!/usr/bin/env python3
"""Simplified occupant + seat belt + vented airbag model, loaded by a car crash pulse (sled-test style).

Car frame: the car deceleration pulse a(t) from a car crash run is applied as a body acceleration
(/GRAV, +x = forward) to the occupant and the cushion; the steering-wheel side of the cushion and the
belt anchors are fixed to the car. The airbag is a pre-inflated vented cushion (/MONVOL/GAS):
initial gauge pressure P0 is the parameter studied. This is NOT an inflator / folded-bag deployment model.
Default is a closed (non-vented) cushion: in this OpenRadioss build the /MONVOL/GAS vent hole released the whole
over-pressure within one output step whatever the vent area (tested 0.001 to 1400 mm2), so venting is not modelled.

All values marked ASSUMED are simplifications chosen for this study, not measured data.
Units g, mm, ms (N, MPa).
"""
import argparse, json, math, os
import numpy as np
p = argparse.ArgumentParser()
p.add_argument("--name", default="BAG"); p.add_argument("--out", default=".")
p.add_argument("--pulse", required=True, help="csv with time_ms, car_decel_g (positive = slowing down)")
p.add_argument("--p0_kPa", type=float, default=30.0)      # initial gauge pressure of the cushion
p.add_argument("--no_bag", type=int, default=0)           # 1 = belt only
p.add_argument("--occ_kg", type=float, default=40.0)      # ASSUMED effective upper-body mass (head + thorax)
p.add_argument("--belt_k", type=float, default=100.0)     # ASSUMED belt stiffness per strap, N/mm
p.add_argument("--belt_lim_kN", type=float, default=3.0)  # ASSUMED load limiter per strap (2 straps -> 6 kN)
p.add_argument("--gap", type=float, default=100.0)        # ASSUMED occupant-to-cushion gap, mm
p.add_argument("--vent_mm2", type=float, default=1400.0)  # ASSUMED vent area (two 30 mm holes)
# vent membrane opens once the gauge pressure climbs back above the initial fill pressure (occupant loading)
p.add_argument("--vent_mode", default="none", help="none = closed cushion; open = vent open from t=0; membrane = opens at p0+3 kPa")
p.add_argument("--tend", type=float, default=120.0)
p.add_argument("--anim", type=int, default=1)
a = p.parse_args()
f20 = lambda *v: "".join(f"{x:20.10g}" for x in v)
i10 = lambda *v: "".join(f"{int(x):10d}" for x in v)
B10, B20 = " " * 10, " " * 20
idl = lambda ids: "\n".join(i10(*ids[k:k + 10]) for k in range(0, len(ids), 10))

nodes, shells = [], {}
def add_node(x, y, z):
    nodes.append((len(nodes) + 1, x, y, z)); return len(nodes)
def box(x0, x1, y0, y1, z0, z1, h, part):
    """Closed box surface of quads with outward normals; returns node ids grid dict."""
    nx, ny, nz = (max(1, round((x1 - x0) / h)), max(1, round((y1 - y0) / h)), max(1, round((z1 - z0) / h)))
    X = np.linspace(x0, x1, nx + 1); Y = np.linspace(y0, y1, ny + 1); Z = np.linspace(z0, z1, nz + 1)
    ids = {}
    def n(i, j, k):
        if (i, j, k) not in ids: ids[(i, j, k)] = add_node(X[i], Y[j], Z[k])
        return ids[(i, j, k)]
    q = shells.setdefault(part, [])
    for i in range(nx):            # z faces
        for j in range(ny):
            q.append((n(i, j, 0), n(i, j + 1, 0), n(i + 1, j + 1, 0), n(i + 1, j, 0)))
            q.append((n(i, j, nz), n(i + 1, j, nz), n(i + 1, j + 1, nz), n(i, j + 1, nz)))
    for i in range(nx):            # y faces
        for k in range(nz):
            q.append((n(i, 0, k), n(i + 1, 0, k), n(i + 1, 0, k + 1), n(i, 0, k + 1)))
            q.append((n(i, ny, k), n(i, ny, k + 1), n(i + 1, ny, k + 1), n(i + 1, ny, k)))
    for j in range(ny):            # x faces
        for k in range(nz):
            q.append((n(0, j, k), n(0, j, k + 1), n(0, j + 1, k + 1), n(0, j + 1, k)))
            q.append((n(nx, j, k), n(nx, j + 1, k), n(nx, j + 1, k + 1), n(nx, j, k + 1)))
    return ids, nx, ny, nz

# occupant upper body (ASSUMED rigid block 220 x 360 x 600 mm, front face at x = 0)
occ_ids, *_ = box(-220.0, 0.0, -180.0, 180.0, 50.0, 650.0, 40.0, 1)
occ_nodes = sorted(set(occ_ids.values()))
# cushion (ASSUMED 240 x 500 x 500 mm = 60 L), front face at x = gap
bag_nodes, bag_fixed = [], []
if not a.no_bag:
    # spherical cushion (ASSUMED diameter 480 mm, about 58 L): cube-sphere quad mesh, front at x = gap
    R = 240.0; cx, cz = a.gap + R, 350.0
    bag_ids, bnx, bny, bnz = box(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 2.0 / 16, 2)
    bag_nodes = sorted(set(bag_ids.values()))
    for nid_ in bag_nodes:
        _, x, y, z = nodes[nid_ - 1]; r = math.sqrt(x * x + y * y + z * z)
        nodes[nid_ - 1] = (nid_, cx + R * x / r, R * y / r, cz + R * z / r)
    bag_fixed = sorted(v for v in bag_nodes if nodes[v - 1][1] > cx + 0.85 * R)   # steering-wheel side
occM = add_node(-110.0, 0.0, 300.0)           # occupant rigid body main node (centre)
ancL = add_node(-600.0, -150.0, 300.0); ancR = add_node(-600.0, 150.0, 300.0)   # belt anchors (fixed to car)
bL = add_node(-220.0, -150.0, 300.0); bR = add_node(-220.0, 150.0, 300.0)       # belt attachment on occupant (in RB)

pulse = np.loadtxt(a.pulse, delimiter=",", skiprows=1)
tp, gp = pulse[:, 0], pulse[:, 1]
g_mm_ms2 = 9.81e-3
d = ["#RADIOSS STARTER\n/BEGIN\n" + f"{a.name:<80s}\n" + i10(2024) + "\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}",
     "/TITLE\nSimplified occupant, belt and vented airbag under a car crash pulse",
     "/NODE\n" + "\n".join(i10(i) + f20(x, y, z) for i, x, y, z in nodes)]
# occupant: rigid shell block
d.append("/MAT/ELAST/3\nOCCUPANT_SKIN\n" + f20(1.0e-3) + "\n" + f20(1000.0, 0.3))
d.append("/PROP/SHELL/3\nOCCUPANT_SKIN\n" + i10(24, 0, 0, 0, 0) + B10 + f20(0) + "\n" + f20(0, 0, 0, 0, 0) + "\n" + i10(1) + B10 + f20(1.0, 0) + B10 + i10(1, 1, 0))
d.append("/PART/1\nOCCUPANT\n" + i10(3, 3, 0))
eid = 0; lines = []
for q in shells[1]:
    eid += 1; lines.append(i10(eid, *q))
d.append("/SHELL/1\n" + "\n".join(lines))
d.append("/GRNOD/NODE/11\nOCC_SLAVES\n" + idl(occ_nodes + [bL, bR]))
d.append("/RBODY/1\nOCCUPANT\n" + i10(occM, 0, 0, 0) + f20(a.occ_kg * 1000.0) + i10(11, 0, 0, 0) + "\n" +
         f20(4.0e8, 4.0e8, 4.0e8) + "\n" + f20(0, 0, 0) + "\n" + i10(0, 0, 0))
d.append("/GRNOD/NODE/12\nOCC_MAIN\n" + i10(occM))
d.append("/BCS/12\nOCC_1D\n   011 111" + i10(0, 12))   # forward/back (x) and vertical (z) translation, pitch rotation (y) free
# belt: two elasto-plastic springs with load limiter
F = a.belt_lim_kN * 1000.0; dy = F / a.belt_k
d.append("/FUNCT/21\nBELT_FORCE\n" + "\n".join(f20(x, y) for x, y in ((-5000, 0), (0, 0), (dy, F), (5000, F))))
d.append("/PROP/TYPE4/21\nBELT\n" + f20(1.0) + B20 + i10(0, 0, 0) + "\n" + f20(a.belt_k, 0.2 * 2 * math.sqrt(a.belt_k * 1.0), 1, 0, 1) + "\n" +
         i10(21, 1, 0, 0, 0) + B20 + f20(0, 0) + "\n" + f20(0, 0, 0, 0))
d.append("/PART/21\nBELT\n" + i10(21, 0, 0))
d.append("/SPRING/21\n" + i10(800001, ancL, bL) + "\n" + i10(800002, ancR, bR))
d.append("/GRNOD/NODE/13\nANCHORS\n" + i10(ancL, ancR))
d.append("/BCS/13\nANCHORS\n   111 111" + i10(0, 13))
# crash pulse as body acceleration in the car frame
d.append("/FUNCT/31\nCAR_DECEL_PULSE_G\n" + "\n".join(f20(t, g) for t, g in zip(tp, gp)) + "\n" + f20(tp[-1] + 1000.0, 0.0))
grav_grp = [occM] + bag_nodes
d.append("/GRNOD/NODE/14\nPULSE_LOADED\n" + idl(grav_grp))
d.append("/GRAV/1\nCAR_PULSE\n" + i10(31) + "         X" + i10(0, 0, 14) + B10 + f20(1.0, g_mm_ms2))
pext = 0.101325; p0 = pext + a.p0_kPa / 1000.0
if not a.no_bag:
    d.append("/MAT/ELAST/4\nAIRBAG_FABRIC\n" + f20(8.0e-4) + "\n" + f20(380.0, 0.35))   # ASSUMED nylon-like fabric
    d.append("/PROP/SHELL/4\nAIRBAG_FABRIC\n" + i10(24, 0, 0, 0, 0) + B10 + f20(0) + "\n" + f20(0, 0, 0, 0, 0) + "\n" + i10(1) + B10 + f20(0.35, 0) + B10 + i10(1, 1, 0))
    d.append("/PART/2\nAIRBAG\n" + i10(4, 4, 0))
    lines = []
    for q in shells[2]:
        eid += 1; lines.append(i10(eid, *q))
    d.append("/SHELL/2\n" + "\n".join(lines))
    d.append("/GRNOD/NODE/15\nBAG_MOUNT\n" + idl(bag_fixed))
    d.append("/BCS/15\nBAG_MOUNT\n   111 111" + i10(0, 15))
    d.append("/SURF/PART/16\nBAG_SURF\n" + i10(2))
    d.append("/SURF/PART/17\nOCC_SURF\n" + i10(1))
    d.append("/GRNOD/PART/18\nBAG_NODES\n" + i10(2))
    d.append("/GRNOD/PART/19\nOCC_SHELL_NODES\n" + i10(1))
    for k, (sec, mst, nm) in enumerate(((19, 16, "OCC_ON_BAG"), (18, 17, "BAG_ON_OCC")), start=1):
        d.append(f"/INTER/TYPE7/{k}\n{nm}\n" + i10(sec, mst, 4, 0, 0) + B10 + i10(0, 0, 0, 0) + "\n" + f20(0, 0, 0) + B20 + i10(0) + "\n" +
                 f20(0, 0, 0, 0) + i10(0, 0) + "\n" + f20(1, 0.3, 2.0, 0, 0) + "\n" + "       000" + B20 + i10(0) + f20(0, 0, 0) + "\n" +
                 i10(0, 0) + f20(0) + i10(0, 0, 0) + f20(0) + i10(0))
    d.append("/MONVOL/GAS/1\nCUSHION\n" + i10(16, 0) + "\n" + f20(0, 0, 0, 0, 0) + "\n" + f20(1.4, 0.01, 0, 295.0, p0 * 1e6 / (287.0 * 295.0) * 1e-6) + "\n" +
             f20(pext, p0, 0, 0, 0) + "\n" + (i10(0) + "\n" if a.vent_mode == "none" else i10(1) + "\n" + i10(0) + f20(a.vent_mm2) + B20 + i10(0) + "\n" +
             (f20(1.0e6, a.p0_kPa / 1000.0 + 0.003, 0) if a.vent_mode == "membrane" else f20(0, 0, 0)) + "\n" + i10(0, 0, 0) + B10 + f20(0, 0, 0)))
    d.append("/TH/MONV/1\nCUSHION\nDEF\n" + i10(1))
    d.append("/TH/INTER/2\nCONTACT\nDEF\n" + i10(1, 2))
d.append("/TH/NODE/3\nOCC_MAIN\nDEF\n" + i10(occM, 0))
d.append("/TH/SPRING/4\nBELT\nDEF\n" + i10(800001, 800002))
if a.anim:
    d.append("/ATH/NODE/9\nANIM_NODES\nDX        DY        DZ\n" + "\n".join(i10(i, 0) for i in sorted(occ_nodes + bag_nodes)))
d.append("/END")
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, f"{a.name}_0000.rad"), "w").write("\n".join(d) + "\n")
open(os.path.join(a.out, f"{a.name}_0001.rad"), "w").write(f"/RUN/{a.name}/1\n{a.tend}\n/VERS/2024\n/TFILE/4\n0.05\n/ATFILE/4\n{a.tend/60:.3f}\n/PRINT/-2000\n/RFILE/0\n10000000\n")
json.dump(dict(vars(a), n_nodes=len(nodes), occ_main=occM), open(os.path.join(a.out, f"{a.name}_info.json"), "w"), indent=1)
print(f"nodes {len(nodes)} shells {eid}")

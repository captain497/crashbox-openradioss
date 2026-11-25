#!/usr/bin/env python3
"""OpenRadioss deck for a square AA6061-T6 crash box, empty or filled with an
out-of-plane re-entrant (auxetic) honeycomb core, crushed by a heavy moving wall.

The core is a 2D re-entrant hexagon network in the cross-section (x-y), extruded
along the crush axis (z), i.e. a multi-cell extrusion. Core ribs share nodes with
the tube wall where they meet it (integral profile).

Re-entrant cell (classic definition, Ren et al. 2018): vertical ribs of length h,
inclined ribs of length l at re-entrant angle theta to the x-axis.
  cell width  = 2 l cos(theta)       -> l from the number of cells across, n
  row pitch   = h - l sin(theta)     -> rows from the cell aspect ratio beta = h/l
Valid cells need h > 2 l sin(theta) (beta > 2 sin(theta)).

Units: g, mm, ms (force N, stress MPa, energy mJ).
"""
import argparse, json, math, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--name", default="CB")
p.add_argument("--width", type=float, default=80.0)    # outer tube mid-surface width (mm)
p.add_argument("--length", type=float, default=200.0)  # crash box length (mm)
p.add_argument("--t", type=float, default=2.0)         # tube wall thickness (mm)
p.add_argument("--core", choices=["none", "reentrant"], default="none")
p.add_argument("--theta", type=float, default=20.0)    # re-entrant angle (deg)
p.add_argument("--ncell", type=int, default=3)         # cells across the width
p.add_argument("--beta", type=float, default=2.0)      # h/l cell aspect ratio
p.add_argument("--tc", type=float, default=1.0)        # core rib thickness (mm)
p.add_argument("--esize", type=float, default=2.5)     # element size (mm)
p.add_argument("--mass_kg", type=float, default=557.0) # impactor mass (kg)
p.add_argument("--v", type=float, default=15.28)       # impact speed (m/s)
p.add_argument("--stroke_frac", type=float, default=0.75)  # run until this crush fraction (time estimate)
p.add_argument("--tend", type=float, default=0.0)      # end time (ms); 0 = from stroke
p.add_argument("--dt_cst", type=float, default=0.0)    # /DT/NODA/CST target (ms); 0 = off
p.add_argument("--anim", type=int, default=1)
p.add_argument("--ishell", type=int, default=24)    # 24 = QEPH
p.add_argument("--out", default=".")
a = p.parse_args()

W, L, T = a.width, a.length, a.t
H = W / 2.0
h_el = a.esize
nz = max(2, round(L / h_el)); hz = L / nz
TOL = 1e-6

# ---------------- core 2D network ----------------
segs = []
core_info = {}
if a.core == "reentrant":
    th = math.radians(a.theta)
    l = W / (2 * a.ncell * math.cos(th))
    h = a.beta * l
    if h <= 2 * l * math.sin(th) + 1e-9:
        raise SystemExit("invalid cell: beta must exceed 2 sin(theta)")
    pitch = h - l * math.sin(th)
    nrow = max(1, round(W / pitch))
    pitch = W / nrow                    # stretch rows to fit exactly
    h = pitch + l * math.sin(th)
    px = 2 * l * math.cos(th)
    core_info = dict(l=l, h=h, row_pitch=pitch, nrow=nrow, cell_width=px)
    raw = []
    for j in range(-1, nrow + 2):
        y0 = -H + j * pitch
        off = (j % 2) * px / 2
        for i in range(-2, a.ncell + 3):
            x0 = -H + i * px + off
            P1 = (x0, y0 - h / 2); P2 = (x0, y0 + h / 2)
            P3 = (x0 + px / 2, y0 + h / 2 - l * math.sin(th))
            P4 = (x0 + px, y0 + h / 2); P5 = (x0 + px, y0 - h / 2)
            P6 = (x0 + px / 2, y0 - h / 2 + l * math.sin(th))
            raw += [(P1, P2), (P2, P3), (P3, P4), (P5, P6), (P6, P1)]

    def clip(p0, p1):  # Liang-Barsky clip to the square
        x0, y0 = p0; dx = p1[0] - x0; dy = p1[1] - y0
        t0, t1 = 0.0, 1.0
        for pp, q in ((-dx, x0 + H), (dx, H - x0), (-dy, y0 + H), (dy, H - y0)):
            if abs(pp) < 1e-12:
                if q < 0: return None
            else:
                r = q / pp
                if pp < 0: t0 = max(t0, r)
                else: t1 = min(t1, r)
        if t1 - t0 < 1e-9: return None
        return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)

    seen = set()
    for p0, p1 in raw:
        c = clip(p0, p1)
        if c is None: continue
        q0, q1 = c
        # drop ribs lying on the tube wall itself
        if (abs(q0[0] - q1[0]) < 1e-6 and abs(abs(q0[0]) - H) < 1e-6) or \
           (abs(q0[1] - q1[1]) < 1e-6 and abs(abs(q0[1]) - H) < 1e-6):
            continue
        key = tuple(sorted([(round(q0[0], 5), round(q0[1], 5)), (round(q1[0], 5), round(q1[1], 5))]))
        if key in seen or key[0] == key[1]: continue
        seen.add(key)
        segs.append(key)

# ---------------- perimeter positions (tube) ----------------
def on_wall(pt):
    x, y = pt
    return abs(abs(x) - H) < 1e-6 or abs(abs(y) - H) < 1e-6

def perim_s(pt):  # arc-length position on the square, CCW from (-H,-H)
    x, y = pt
    if abs(y + H) < 1e-6: return x + H
    if abs(x - H) < 1e-6: return W + (y + H)
    if abs(y - H) < 1e-6: return 2 * W + (H - x)
    return 3 * W + (H - y)

def s_to_xy(s):
    s = s % (4 * W)
    k = int(s // W); f = s - k * W
    return [(-H + f, -H), (H, -H + f), (H - f, H), (-H, H - f)][min(k, 3)]

P = 4 * W
ns_side = max(2, round(W / h_el))
s_uniform = [k * W / ns_side for k in range(4 * ns_side)]
s_core = sorted({round(perim_s(q), 5) % P for sg in segs for q in sg if on_wall(q)})
# snap core attachment points to the uniform grid when close, else insert
s_list = list(s_uniform)
snap = {}
for sc in s_core:
    k = min(range(len(s_uniform)), key=lambda i: min(abs(s_uniform[i] - sc), P - abs(s_uniform[i] - sc)))
    d = min(abs(s_uniform[k] - sc), P - abs(s_uniform[k] - sc))
    if d < 0.35 * W / ns_side:
        snap[sc] = s_uniform[k]
    else:
        snap[sc] = sc; s_list.append(sc)
s_list = sorted(set(round(s, 6) for s in s_list))

# ---------------- 2D points and lines ----------------
pts2 = {}      # key -> index
coords2 = []
def pid(xy):
    k = (round(xy[0], 4), round(xy[1], 4))
    if k not in pts2:
        pts2[k] = len(coords2); coords2.append(xy)
    return pts2[k]

tube_ring = [pid(s_to_xy(s)) for s in s_list]
tube_edges = [(tube_ring[i], tube_ring[(i + 1) % len(tube_ring)]) for i in range(len(tube_ring))]

core_edges = []
core_len = 0.0
for q0, q1 in segs:
    q0 = s_to_xy(snap[round(perim_s(q0), 5) % P]) if on_wall(q0) else q0
    q1 = s_to_xy(snap[round(perim_s(q1), 5) % P]) if on_wall(q1) else q1
    Ls = math.dist(q0, q1)
    if Ls < 0.3 * h_el: continue
    core_len += Ls
    nseg = max(1, round(Ls / h_el))
    ids = [pid((q0[0] + (q1[0] - q0[0]) * k / nseg, q0[1] + (q1[1] - q0[1]) * k / nseg)) for k in range(nseg + 1)]
    core_edges += list(zip(ids[:-1], ids[1:]))

n2 = len(coords2)
def nid(i2, j):
    return j * n2 + i2 + 1
nodes = [(nid(i, j), x, y, j * hz) for j in range(nz + 1) for i, (x, y) in enumerate(coords2)]

def extrude(edges, eid0):
    out = []; e = eid0
    for j in range(nz):
        for (i0, i1) in edges:
            e += 1
            out.append((e, nid(i0, j), nid(i1, j), nid(i1, j + 1), nid(i0, j + 1)))
    return out
sh_tube = extrude(tube_edges, 0)
sh_core = extrude(core_edges, len(sh_tube)) if core_edges else []
bottom = sorted({nid(i, 0) for e in tube_edges + core_edges for i in e})
wall_node = len(nodes) + 1
zw = L + 0.5

rho, E, nu = 2.70e-3, 70000.0, 0.33
A, B, n, C = 324.0, 114.0, 0.42, 0.002
eps0 = 1.0e-3
mass_tube = rho * P * T * L
mass_core = rho * core_len * a.tc * L
v = a.v; mass_g = a.mass_kg * 1000.0
tend = a.tend if a.tend > 0 else 1.15 * a.stroke_frac * L / v   # ms (speed ~ constant: heavy wall)

def f20(*vals): return "".join(f"{x:20.10g}" for x in vals)
def i10(*vals): return "".join(f"{int(x):10d}" for x in vals)
B10, B20 = " " * 10, " " * 20
def idlist(ids): return "\n".join(i10(*ids[k:k + 10]) for k in range(0, len(ids), 10))
def prop(pid_, title, thick):
    return (f"/PROP/SHELL/{pid_}\n{title}\n" + i10(a.ishell, 0, 0, 0, 0) + B10 + f20(0) + "\n" +
            f20(0, 0, 0, 0, 0) + "\n" + i10(5) + B10 + f20(thick, 0) + B10 + i10(1, 1, 0))

deck = []
deck.append("#RADIOSS STARTER\n/BEGIN\n" + f"{a.name:<80s}\n" + i10(2024) + "\n" +
            f"{'g':>20s}{'mm':>20s}{'ms':>20s}\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}")
deck.append(f"/TITLE\nAA6061-T6 crash box, core={a.core}")
deck.append("/ANALY\n" + i10(0, 0, 1, 0))
deck.append("/RANDOM\n" + f20(1.0e-4))
deck.append("/MAT/PLAS_JOHNS/1\nAA6061-T6\n" + f20(rho) + "\n" + f20(E, nu) + i10(0) + "\n" +
            f20(A, B, n, 0.0, 0.0) + "\n" + f20(C, eps0) + i10(1, 1) + f20(10.0, 0.0) + "\n" +
            f20(0, 0, 0, 0))
deck.append(prop(1, "TUBE_SHELL", T))
deck.append("/PART/1\nTUBE\n" + i10(1, 1, 0))
if sh_core:
    deck.append(prop(2, "CORE_SHELL", a.tc))
    deck.append("/PART/2\nCORE\n" + i10(2, 1, 0))
deck.append("/NODE\n" + "\n".join(i10(i) + f20(x, y, z) for i, x, y, z in nodes) +
            "\n" + i10(wall_node) + f20(0.0, 0.0, zw))
deck.append("/SHELL/1\n" + "\n".join(i10(*e) for e in sh_tube))
if sh_core:
    deck.append("/SHELL/2\n" + "\n".join(i10(*e) for e in sh_core))
deck.append("/GRNOD/NODE/1\nBOTTOM\n" + idlist(bottom))
deck.append("/BCS/1\nCLAMP_BOTTOM\n   111 111" + i10(0, 1))
parts = "1" if not sh_core else "1 2"
deck.append("/GRNOD/PART/2\nALL_NODES\n" + i10(*parts.split()))
deck.append("/SURF/PART/3\nALL_SURF\n" + i10(*parts.split()))
deck.append("/INTER/TYPE7/1\nSELF_CONTACT\n" + i10(2, 3, 4, 0, 0) + B10 + i10(0, 0, 0, 0) + "\n" +
            f20(0, 0, 0) + B20 + i10(0) + "\n" + f20(0, 0, 0, 0) + i10(0, 0) + "\n" +
            f20(1, 0.15, 0.5 * min(T, a.tc if sh_core else T), 0, 0) + "\n" +
            "       000" + B20 + i10(0) + f20(0, 0, 0) + "\n" +
            i10(0, 0) + f20(0) + i10(0, 0, 0) + f20(0) + i10(0))
deck.append("/RWALL/PLANE/1\nIMPACTOR\n" + i10(wall_node, 0, 2, 1) + "\n" + f20(0, 0, 0) + "\n" +
            f20(mass_g, 0, 0, -v) + "\n" + f20(0.0, 0.0, zw - 1.0))
deck.append("/TH/RWALL/1\nWALL\nDEF\n" + i10(1))
deck.append("/TH/NODE/2\nWALLNODE\nDEF\n" + i10(wall_node, 0))
deck.append("/TH/PART/3\nPART_ENERGY\nDEF\n" + "\n".join(i10(k) for k in parts.split()))
if a.anim:
    deck.append("/ATH/NODE/4\nANIM_NODES\nDX        DY        DZ\n" +
                "\n".join(i10(i, 0) for i, *_ in nodes))
deck.append("/END")

os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, f"{a.name}_0000.rad"), "w") as fh:
    fh.write("\n".join(deck) + "\n")
dtcard = f"\n/DT/NODA/CST/0\n0.9 {a.dt_cst}" if a.dt_cst > 0 else ""
with open(os.path.join(a.out, f"{a.name}_0001.rad"), "w") as fh:
    fh.write(f"/RUN/{a.name}/1\n{tend:.3f}\n/VERS/2024\n/TFILE/4\n0.005\n/ATFILE/4\n{tend/40:.4f}\n"
             f"/PRINT/-5000{dtcard}\n/RFILE/0\n10000000\n")

info = dict(vars(a), n_nodes=len(nodes) + 1, n_shell_tube=len(sh_tube), n_shell_core=len(sh_core),
            mass_tube_g=mass_tube, mass_core_g=mass_core, mass_total_g=mass_tube + mass_core,
            core_rib_length_mm=core_len, KE_kJ=0.5 * mass_g * v ** 2 / 1e6, tend_ms=tend, **core_info)
with open(os.path.join(a.out, f"{a.name}_info.json"), "w") as fh:
    json.dump(info, fh, indent=2)
with open(os.path.join(a.out, f"{a.name}_section.json"), "w") as fh:
    json.dump(dict(points=coords2, tube=tube_edges, core=core_edges), fh)
print(json.dumps({k: info[k] for k in ("n_nodes", "n_shell_tube", "n_shell_core", "mass_tube_g",
                                       "mass_core_g", "mass_total_g", "tend_ms")}))

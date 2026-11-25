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
p.add_argument("--core", choices=["none", "reentrant", "horsetail"], default="none")
p.add_argument("--inner_frac", type=float, default=0.5)  # horsetail: inner tube width / outer width
p.add_argument("--ti", type=float, default=0.0)        # horsetail: inner tube thickness (0 = tc)
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
p.add_argument("--wall_slide", type=int, default=1)  # RWALL: 0 = sliding, 1 = tied (nodes stick to the impactor face)
p.add_argument("--ishell", type=int, default=24)    # 24 = QEPH
p.add_argument("--car", type=int, default=0)          # 1 = simplified half-car: box + rail spring + car mass into a FIXED rigid wall
p.add_argument("--rail_kN", type=float, default=180.0) # car mode: crush force plateau of the lumped rest-of-front structure (per side)
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
Hi = a.inner_frac * H if a.core == "horsetail" else 0.0
if a.core in ("reentrant", "horsetail"):
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

    def lb(p0, p1, Hs):  # Liang-Barsky parameters of the segment inside the square of half-width Hs
        x0, y0 = p0; dx = p1[0] - x0; dy = p1[1] - y0
        t0, t1 = 0.0, 1.0
        for pp, q in ((-dx, x0 + Hs), (dx, Hs - x0), (-dy, y0 + Hs), (dy, Hs - y0)):
            if abs(pp) < 1e-12:
                if q < 0: return None
            else:
                r = q / pp
                if pp < 0: t0 = max(t0, r)
                else: t1 = min(t1, r)
        if t1 - t0 < 1e-9: return None
        return t0, t1

    def clip_all(p0, p1):  # pieces inside the outer square and (horsetail) outside the inner square
        o = lb(p0, p1, H)
        if o is None: return []
        pieces = [o]
        if Hi > 0:
            i = lb(p0, p1, Hi)
            if i is not None:
                pieces = [(o[0], min(o[1], i[0])), (max(o[0], i[1]), o[1])]
                pieces = [q for q in pieces if q[1] - q[0] > 1e-9]
        dx = p1[0] - p0[0]; dy = p1[1] - p0[1]
        return [((p0[0] + t0 * dx, p0[1] + t0 * dy), (p0[0] + t1 * dx, p0[1] + t1 * dy)) for t0, t1 in pieces]

    def clip(p0, p1):  # unused legacy
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
      for q0, q1 in clip_all(p0, p1):
        # drop ribs lying on a tube wall itself
        if any((abs(q0[0] - q1[0]) < 1e-6 and abs(abs(q0[0]) - Hs) < 1e-6) or
               (abs(q0[1] - q1[1]) < 1e-6 and abs(abs(q0[1]) - Hs) < 1e-6) for Hs in (H, Hi) if Hs > 0):
            continue
        key = tuple(sorted([(round(q0[0], 5), round(q0[1], 5)), (round(q1[0], 5), round(q1[1], 5))]))
        if key in seen or key[0] == key[1]: continue
        seen.add(key)
        segs.append(key)

# ---------------- tube rings (outer, and inner for horsetail) ----------------
def on_sq(pt, Hs):
    x, y = pt
    return Hs > 0 and (abs(abs(x) - Hs) < 1e-6 and abs(y) <= Hs + 1e-6 or
                       abs(abs(y) - Hs) < 1e-6 and abs(x) <= Hs + 1e-6)

def perim_s(pt, Hs):  # arc-length position on the square, CCW from (-Hs,-Hs)
    x, y = pt; Ws = 2 * Hs
    if abs(y + Hs) < 1e-6: return x + Hs
    if abs(x - Hs) < 1e-6: return Ws + (y + Hs)
    if abs(y - Hs) < 1e-6: return 2 * Ws + (Hs - x)
    return 3 * Ws + (Hs - y)

def s_to_xy(s, Hs):
    Ws = 2 * Hs; s = s % (4 * Ws)
    k = int(s // Ws); f = s - k * Ws
    return [(-Hs + f, -Hs), (Hs, -Hs + f), (Hs - f, Hs), (-Hs, Hs - f)][min(k, 3)]

pts2 = {}      # key -> index
coords2 = []
def pid(xy):  # merge points closer than 1e-4 mm (rounding alone can split a node in two)
    gx, gy = math.floor(xy[0] / 0.01), math.floor(xy[1] / 0.01)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for i in pts2.get((gx + dx, gy + dy), ()):
                if math.dist(coords2[i], xy) < 1e-4:
                    return i
    pts2.setdefault((gx, gy), []).append(len(coords2)); coords2.append(xy)
    return len(coords2) - 1

def ring(Hs):
    """Uniform ring on square Hs with core attachment points snapped/inserted. Returns edges, snap(pt)."""
    Ws = 2 * Hs; Pp = 4 * Ws
    ns_side = max(2, round(Ws / h_el))
    s_uniform = [k * Ws / ns_side for k in range(4 * ns_side)]
    s_core = sorted({round(perim_s(q, Hs), 5) % Pp for sg in segs for q in sg if on_sq(q, Hs)})
    s_list = list(s_uniform); snap = {}
    for sc in s_core:
        k = min(range(len(s_uniform)), key=lambda i: min(abs(s_uniform[i] - sc), Pp - abs(s_uniform[i] - sc)))
        d = min(abs(s_uniform[k] - sc), Pp - abs(s_uniform[k] - sc))
        if d < 0.35 * Ws / ns_side: snap[sc] = s_uniform[k]
        else: snap[sc] = sc; s_list.append(sc)
    s_list = sorted(set(round(x, 6) for x in s_list))
    ids = [pid(s_to_xy(x, Hs)) for x in s_list]
    edges = [(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
    return edges, (lambda q: s_to_xy(snap[round(perim_s(q, Hs), 5) % Pp], Hs))

P = 4 * W
tube_edges, snap_out = ring(H)
inner_edges, snap_in = ring(Hi) if Hi > 0 else ([], None)

def snap_pt(q):
    if on_sq(q, H): return snap_out(q)
    if Hi > 0 and on_sq(q, Hi): return snap_in(q)
    return q

core_edges = []
core_len = 0.0
for q0, q1 in segs:
    q0 = snap_pt(q0); q1 = snap_pt(q1)
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
sh_in = extrude(inner_edges, len(sh_tube) + len(sh_core)) if inner_edges else []
TI = a.ti if a.ti > 0 else a.tc
bottom = sorted({nid(i, 0) for e in tube_edges + core_edges + inner_edges for i in e})
wall_node = len(nodes) + 1
zw = L + 0.5

# AA6061-T6 from the tensile tests of Aktas et al. (2023), Table 1 and Fig. 8b: rho 2700 kg/m3,
# E 68.85 GPa, nu 0.33, yield 285 MPa. Johnson-Cook B, n fitted to values read off Fig. 8b
# (true plastic strain 0-0.06; beyond that the power law is an extrapolation).
# Strain-rate term C = 0.002 (eps0 = 1/s) from Lesuer-type AA6061-T6 data (assumed; weak rate effect).
rho, E, nu = 2.70e-3, 68850.0, 0.33
A, B, n, C = 285.0, 199.0, 0.532, 0.002
eps0 = 1.0e-3
mass_tube = rho * P * T * L
mass_core = rho * core_len * a.tc * L + rho * 8 * Hi * TI * L
v = a.v; mass_g = a.mass_kg * 1000.0
tend = a.tend if a.tend > 0 else (100.0 if a.car else 1.15 * a.stroke_frac * L / v)   # ms (speed ~ constant: heavy wall)

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
if sh_in:
    deck.append(prop(3, "INNER_TUBE_SHELL", TI))
    deck.append("/PART/3\nINNER_TUBE\n" + i10(3, 1, 0))
B_node, C_node = wall_node, wall_node + 1   # car mode: box mount (rigid body master) and car mass node
extra = (i10(B_node) + f20(0.0, 0.0, -10.0) + "\n" + i10(C_node) + f20(0.0, 0.0, -1000.0)) if a.car else (i10(wall_node) + f20(0.0, 0.0, zw))
deck.append("/NODE\n" + "\n".join(i10(i) + f20(x, y, z) for i, x, y, z in nodes) + "\n" + extra)
deck.append("/SHELL/1\n" + "\n".join(i10(*e) for e in sh_tube))
if sh_core:
    deck.append("/SHELL/2\n" + "\n".join(i10(*e) for e in sh_core))
if sh_in:
    deck.append("/SHELL/3\n" + "\n".join(i10(*e) for e in sh_in))
deck.append("/GRNOD/NODE/1\nBOTTOM\n" + idlist(bottom))
if not a.car:
    deck.append("/BCS/1\nCLAMP_BOTTOM\n   111 111" + i10(0, 1))
else:
    # Simplified half-car (ASSUMED values): 5 kg box mount (rigid body), rail = elastic-plastic spring with
    # 15 mm elastic ramp to a constant crush force rail_kN, 50% critical damping on the mount, car mass lumped at C.
    car_g = a.mass_kg * 1000.0
    F = a.rail_kN * 1000.0
    deck.append("/RBODY/1\nBOX_MOUNT\n" + i10(B_node, 0, 0, 0) + f20(5000.0) + i10(1, 0, 0, 0) + "\n" +
                f20(0, 0, 0) + "\n" + f20(0, 0, 0) + "\n" + i10(0, 0, 0))
    deck.append("/GRNOD/NODE/5\nCAR_NODES\n" + i10(B_node, C_node))
    deck.append("/BCS/2\nCAR_1D\n   110 111" + i10(0, 5))
    deck.append("/GRNOD/NODE/6\nCAR_MASS\n" + i10(C_node))
    deck.append("/ADMAS/0/1\nCAR_MASS\n" + f20(car_g) + i10(6))
    deck.append("/FUNCT/9001\nRAIL_FORCE_VS_DEFLECTION\n" + "\n".join(f20(x, y) for x, y in
                ((-5000, -F), (-15, -F), (0, 0), (15, F), (5000, F))))
    deck.append("/PROP/TYPE4/9\nRAIL_SPRING\n" + f20(10.0) + B20 + i10(0, 0, 0) + "\n" + f20(F / 15.0, 0.5 * 2.0 * math.sqrt(F / 15.0 * 5000.0), 1, 0, 1) + "\n" +
                i10(9001, 1, 0, 0, 0) + B20 + f20(0, 0) + "\n" + f20(0, 0, 0, 0))
    deck.append("/PART/9\nRAIL\n" + i10(9, 0, 0))
    deck.append("/SPRING/9\n" + i10(9000001, B_node, C_node))
    deck.append("/INIVEL/TRA/1\nCAR_SPEED_BOX\n" + f20(0, 0, v) + i10(2, 0))
    deck.append("/INIVEL/TRA/2\nCAR_SPEED_NODES\n" + f20(0, 0, v) + i10(5, 0))
parts = "1" + (" 2" if sh_core else "") + (" 3" if sh_in else "")
deck.append("/GRNOD/PART/2\nALL_NODES\n" + i10(*parts.split()))
deck.append("/SURF/PART/3\nALL_SURF\n" + i10(*parts.split()))
deck.append("/INTER/TYPE7/1\nSELF_CONTACT\n" + i10(2, 3, 4, 0, 0) + B10 + i10(0, 0, 0, 0) + "\n" +
            f20(0, 0, 0) + B20 + i10(0) + "\n" + f20(0, 0, 0, 0) + i10(0, 0) + "\n" +
            f20(1, 0.15, 0.5 * min(T, a.tc if sh_core else T, TI if sh_in else T), 0, 0) + "\n" +
            "       000" + B20 + i10(0) + f20(0, 0, 0) + "\n" +
            i10(0, 0) + f20(0) + i10(0, 0, 0) + f20(0) + i10(0))
if a.car:
    deck.append("/RWALL/PLANE/1\nFIXED_WALL\n" + i10(0, a.wall_slide, 2, 0) + "\n" + f20(0, 0, 0) + "\n" +
                f20(0.0, 0.0, zw) + "\n" + f20(0.0, 0.0, zw - 1.0))
else:
    deck.append("/RWALL/PLANE/1\nIMPACTOR\n" + i10(wall_node, a.wall_slide, 2, 1) + "\n" + f20(0, 0, 0) + "\n" +
                f20(mass_g, 0, 0, -v) + "\n" + f20(0.0, 0.0, zw - 1.0))
deck.append("/TH/RWALL/1\nWALL\nDEF\n" + i10(1))
if a.car:
    deck.append("/TH/NODE/2\nCAR_NODES\nDEF\n" + i10(B_node, 0) + "\n" + i10(C_node, 0))
    deck.append("/TH/SPRING/5\nRAIL\nDEF\n" + i10(9000001))
else:
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

info = dict(vars(a), n_nodes=len(nodes) + (2 if a.car else 1), n_shell_tube=len(sh_tube), n_shell_core=len(sh_core) + len(sh_in), n_shell_inner=len(sh_in),
            mass_tube_g=mass_tube, mass_core_g=mass_core, mass_total_g=mass_tube + mass_core,
            core_rib_length_mm=core_len, KE_kJ=0.5 * mass_g * v ** 2 / 1e6, tend_ms=tend, **core_info)
with open(os.path.join(a.out, f"{a.name}_info.json"), "w") as fh:
    json.dump(info, fh, indent=2)
with open(os.path.join(a.out, f"{a.name}_section.json"), "w") as fh:
    json.dump(dict(points=coords2, tube=tube_edges + inner_edges, core=core_edges), fh)
print(json.dumps({k: info[k] for k in ("n_nodes", "n_shell_tube", "n_shell_core", "mass_tube_g",
                                       "mass_core_g", "mass_total_g", "tend_ms")}))

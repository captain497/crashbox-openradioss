#!/usr/bin/env python3
"""Generate an OpenRadioss (Radioss block format, 2024 syntax) deck for a square
thin-walled AA6061-T6 tube crushed axially by a moving rigid wall with mass.

Unit system: g, mm, ms  ->  force N, stress MPa, energy mJ (N.mm), velocity m/s.

All geometry, impact and material values are PLACEHOLDER ASSUMPTIONS taken from
typical literature ranges; they are not yet fixed by the project.
"""
import argparse, math, os

p = argparse.ArgumentParser()
p.add_argument("--name", default="BASE")
p.add_argument("--width", type=float, default=60.0)    # mid-surface width (mm)
p.add_argument("--length", type=float, default=180.0)  # tube length (mm)
p.add_argument("--t", type=float, default=2.0)         # wall thickness (mm)
p.add_argument("--esize", type=float, default=2.5)     # target element size (mm)
p.add_argument("--mass_kg", type=float, default=100.0) # impactor mass (kg)
p.add_argument("--v", type=float, default=10.0)        # impact speed (m/s)
p.add_argument("--tend", type=float, default=30.0)     # end time (ms)
p.add_argument("--trigger", type=float, default=0.0)   # optional dent depth (mm), 0 = none
p.add_argument("--istf", type=int, default=0)        # TYPE7 contact stiffness option
p.add_argument("--dtmin_cst", type=float, default=0.0) # if >0: /DT/NODA/CST target time step (ms), adds small mass
p.add_argument("--mat", choices=["aa6061", "simpson_steel", "simpson_tab"], default="aa6061")
p.add_argument("--fric", type=float, default=0.15)     # self-contact friction
p.add_argument("--wall_slide", type=int, default=0)    # RWALL: 0 sliding, 1 tied, 2 friction
p.add_argument("--wall_fric", type=float, default=0.0)
p.add_argument("--rc", type=float, default=0.0)        # corner radius of the mid-surface (mm), 0 = sharp corners
p.add_argument("--anim", type=int, default=1)
p.add_argument("--out", default=".")
a = p.parse_args()

W, L, T = a.width, a.length, a.t
ns = max(2, round(W / a.esize))          # elements per side
nz = max(2, round(L / a.esize))          # elements along length
hs, hz = W / ns, L / nz

per = []  # perimeter points, counter-clockwise: (x, y, side, frac)
if a.rc <= 0:
    corners = [(-W/2, -W/2), (W/2, -W/2), (W/2, W/2), (-W/2, W/2)]
    for k in range(4):
        x0, y0 = corners[k]
        x1, y1 = corners[(k + 1) % 4]
        for i in range(ns):
            f = i / ns
            per.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0), k, f))
else:  # rounded corners: straight flat of W - 2 rc per side + 90 deg arc
    R, Hh = a.rc, W / 2
    nf = max(2, round((W - 2 * R) / a.esize)); na = max(2, round(0.5 * math.pi * R / a.esize))
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    starts = [(-Hh + R, -Hh), (Hh, -Hh + R), (Hh - R, Hh), (-Hh, Hh - R)]
    centres = [(Hh - R, -Hh + R), (Hh - R, Hh - R), (-Hh + R, Hh - R), (-Hh + R, -Hh + R)]
    for k in range(4):
        (x0, y0), (dx, dy) = starts[k], dirs[k]
        for i in range(nf):
            f = i / nf
            per.append((x0 + dx * f * (W - 2 * R), y0 + dy * f * (W - 2 * R), k, f))
        cx, cy = centres[k]; a0 = -math.pi / 2 + k * math.pi / 2
        for i in range(na):
            ang = a0 + 0.5 * math.pi * i / na
            per.append((cx + R * math.cos(ang), cy + R * math.sin(ang), -1, 0.0))
npr = len(per)

def nid(i, j):
    return j * npr + (i % npr) + 1

node_list = []
for j in range(nz + 1):
    z = j * hz
    for i, (x, y, side, f) in enumerate(per):
        yy = y
        if a.trigger > 0 and side in (0, 2) and 0 < f < 1:
            zc, half = L - 0.15 * W, 0.25 * W
            if abs(z - zc) < half:
                d = a.trigger * math.sin(math.pi * f) * math.cos(math.pi * (z - zc) / (2 * half))
                yy = y + (d if side == 0 else -d)
        node_list.append((nid(i, j), x, yy, z))
shells = []
eid = 0
for j in range(nz):
    for i in range(npr):
        eid += 1
        shells.append((eid, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))
bottom = [nid(i, 0) for i in range(npr)]
wall_node = len(node_list) + 1
zw = L + 0.5   # wall starts 0.5 mm above the tube

def f20(*v):
    return "".join(f"{x:20.10g}" for x in v)
def i10(*v):
    return "".join(f"{int(x):10d}" for x in v)
B10 = " " * 10
B20 = " " * 20
def idlist(ids):
    return "\n".join(i10(*ids[k:k + 10]) for k in range(0, len(ids), 10))

# AA6061-T6 Johnson-Cook (Lesuer-type values; thermal softening off)
rho, E, nu = 2.70e-3, 70000.0, 0.33
A, B, n, C = 324.0, 114.0, 0.42, 0.002
eps0 = 1.0e-3   # 1 /s in 1/ms
if a.mat in ("simpson_steel", "simpson_tab"):
    # Simpson & Kazanci (2020) mild steel tube: E=200 GPa, rho=7132 kg/m3, nu=0.28, sy=265 MPa,
    # post-yield power law K=565 MPa, n=0.192. Johnson-Cook A,B,n fitted to max(265, 565*ep^0.192)
    # over ep 0-0.6 (within 5% above ep=0.05). Rate effects off (quasi-static test).
    rho, E, nu = 7.132e-3, 200000.0, 0.28
    A, B, n, C = 265.0, 339.4, 0.541, 0.0
mass_g = a.mass_kg * 1000.0
v = a.v         # m/s == mm/ms

deck = []
deck.append("#RADIOSS STARTER\n/BEGIN\n" + f"{a.name:<80s}\n" + i10(2024) + "\n" +
            f"{'g':>20s}{'mm':>20s}{'ms':>20s}\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}")
deck.append("/TITLE\nSquare tube axial crush")
deck.append("/ANALY\n#    N2D3D     INTEG   IPARITH      ISUB\n" + i10(0, 0, 1, 0))
deck.append("/RANDOM\n" + f20(1.0e-4))
if a.mat == "simpson_tab":
    # exact Simpson & Kazanci curve as a table: sigma = max(265, 565 * ep^0.192) MPa
    eps = [0.0, 0.0194, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    deck.append("/FUNCT/10\nSIMPSON_STEEL_HARDENING\n" + "\n".join(f20(e, max(265.0, 565.0 * e ** 0.192)) for e in eps))
    deck.append("/MAT/PLAS_TAB/1\nTUBE_MAT\n" + f20(rho) + "\n" + f20(E, nu, 0, 0, 0) + "\n" +
                i10(1, 0) + f20(0, 0, 0) + B10 + i10(0) + "\n" + i10(0) + f20(0) + i10(0) + f20(0, 0) + "\n" +
                i10(10) + "\n" + f20(1.0) + "\n" + f20(0.0))
elif True: deck.append("/MAT/PLAS_JOHNS/1\nTUBE_MAT\n" + f20(rho) + "\n" + f20(E, nu) + i10(0) + "\n" +
            f20(A, B, n, 0.0, 0.0) + "\n" + f20(C, eps0) + i10(1, 1) + f20(10.0, 0.0) + "\n" +  # Fsmooth=1, Fcut=10 kHz
            f20(0, 0, 0, 0))
deck.append("/PROP/SHELL/1\nTUBE_SHELL\n" + i10(24, 0, 0, 0, 0) + B10 + f20(0) + "\n" +
            f20(0, 0, 0, 0, 0) + "\n" + i10(5) + B10 + f20(T, 0) + B10 + i10(1, 1, 0))
deck.append("/PART/1\nTUBE\n" + i10(1, 1, 0))
deck.append("/NODE\n" + "\n".join(i10(i) + f20(x, y, z) for i, x, y, z in node_list) +
            "\n" + i10(wall_node) + f20(0.0, 0.0, zw))
deck.append("/SHELL/1\n" + "\n".join(i10(*e) for e in shells))
deck.append("/GRNOD/NODE/1\nBOTTOM\n" + idlist(bottom))
deck.append("/BCS/1\nCLAMP_BOTTOM\n#   trarot   Skew_id   Gnod_id\n   111 111" + i10(0, 1))
deck.append("/GRNOD/PART/2\nTUBE_NODES\n" + i10(1))
deck.append("/SURF/PART/3\nTUBE_SURF\n" + i10(1))
deck.append("/INTER/TYPE7/1\nSELF_CONTACT\n" + i10(2, 3, a.istf, 0, 0) + B10 + i10(0, 0, 0, 0) + "\n" +
            f20(0, 0, 0) + B20 + i10(0) + "\n" +
            f20(0, 0, 0, 0) + i10(0, 0) + "\n" +
            f20(1, a.fric, 0.5 * T, 0, 0) + "\n" +
            "       000" + B20 + i10(0) + f20(0, 0, 0) + "\n" +
            i10(0, 0) + f20(0) + i10(0, 0, 0) + f20(0) + i10(0))
deck.append("/RWALL/PLANE/1\nIMPACTOR\n" + i10(wall_node, a.wall_slide, 2, 1) + "\n" + f20(0, a.wall_fric, 0) + "\n" +
            f20(mass_g, 0, 0, -v) + "\n" + f20(0.0, 0.0, zw - 1.0))
deck.append("/TH/RWALL/1\nWALL\nDEF\n" + i10(1))
deck.append("/TH/NODE/2\nWALLNODE\nDEF\n" + i10(wall_node, 0))
deck.append("/TH/PART/3\nTUBE_ENERGY\nDEF\n" + i10(1))
# coarse-in-time nodal displacements of every tube node, written to the A-file, for animation
if a.anim: deck.append("/ATH/NODE/4\nANIM_NODES\nDX        DY        DZ\n" +
            "\n".join(i10(i, 0) for i, *_ in node_list))
deck.append("/END")

os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, f"{a.name}_0000.rad"), "w") as fh:
    fh.write("\n".join(deck) + "\n")

dtcard = f"\n/DT/NODA/CST/0\n0.9 {a.dtmin_cst}" if a.dtmin_cst > 0 else ""
eng = f"""/RUN/{a.name}/1
{a.tend}
/VERS/2024
/TFILE/4
0.01
/ATFILE/4
0.5
/ANIM/DT
0. 0.5
/ANIM/ELEM/EPSP
/ANIM/ELEM/VONM
/ANIM/ELEM/THIC
/PRINT/-2000{dtcard}
/RFILE/0
1000000
"""
with open(os.path.join(a.out, f"{a.name}_0001.rad"), "w") as fh:
    fh.write(eng)

print(f"nodes={len(node_list)+1} shells={len(shells)} h_side={hs:.3f} h_len={hz:.3f} "
      f"tube_mass={4*W*L*T*rho:.1f} g  KE={0.5*mass_g*v**2/1e6:.3f} kJ")

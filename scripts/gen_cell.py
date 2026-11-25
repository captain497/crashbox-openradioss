#!/usr/bin/env python3
"""Re-entrant (auxetic) lattice block in-plane compression, to show the negative Poisson's ratio
of the cell used in the crash box core. Same AA6061-T6 card and shell formulation; ribs are shells
extruded through the depth. Crushed between a fixed rigid wall and a heavy moving rigid wall.
Units g, mm, ms.
"""
import argparse, json, math, os
p = argparse.ArgumentParser()
p.add_argument("--name", default="CELL"); p.add_argument("--out", default=".")
p.add_argument("--theta", type=float, default=20.0); p.add_argument("--l", type=float, default=10.0)
p.add_argument("--beta", type=float, default=2.0); p.add_argument("--tc", type=float, default=1.0)
p.add_argument("--nx", type=int, default=5); p.add_argument("--ny", type=int, default=6)
p.add_argument("--depth", type=float, default=10.0); p.add_argument("--h_el", type=float, default=2.0)
p.add_argument("--v", type=float, default=1.0); p.add_argument("--strain", type=float, default=0.4)
p.add_argument("--anim", type=int, default=1)
a = p.parse_args()
th = math.radians(a.theta); l = a.l; h = a.beta * l
px = 2 * l * math.cos(th); pitch = h - l * math.sin(th)
segs = set()
def key(pt): return (round(pt[0], 5), round(pt[1], 5))
for j in range(a.ny):
    y0 = j * pitch; off = (j % 2) * px / 2
    for i in range(a.nx):
        x0 = i * px + off
        P1 = (x0, y0 - h / 2); P2 = (x0, y0 + h / 2); P3 = (x0 + px / 2, y0 + h / 2 - l * math.sin(th))
        P4 = (x0 + px, y0 + h / 2); P5 = (x0 + px, y0 - h / 2); P6 = (x0 + px / 2, y0 - h / 2 + l * math.sin(th))
        for s in ((P1, P2), (P2, P3), (P3, P4), (P4, P5), (P5, P6), (P6, P1)):
            k = tuple(sorted((key(s[0]), key(s[1]))))
            segs.add(k)
pts = {}; coords = []
def pid(xy):
    k = key(xy)
    if k not in pts: pts[k] = len(coords); coords.append(xy)
    return pts[k]
edges = []
for q0, q1 in sorted(segs):
    Ls = math.dist(q0, q1); ns = max(1, round(Ls / a.h_el))
    ids = [pid((q0[0] + (q1[0] - q0[0]) * k / ns, q0[1] + (q1[1] - q0[1]) * k / ns)) for k in range(ns + 1)]
    edges += list(zip(ids[:-1], ids[1:]))
nz = max(1, round(a.depth / a.h_el)); hz = a.depth / nz; n2 = len(coords)
nid = lambda i, k: k * n2 + i + 1
nodes = [(nid(i, k), x, y, k * hz) for k in range(nz + 1) for i, (x, y) in enumerate(coords)]
shells = []; e = 0
for k in range(nz):
    for i0, i1 in edges:
        e += 1; shells.append((e, nid(i0, k), nid(i1, k), nid(i1, k + 1), nid(i0, k + 1)))
ys = [c[1] for c in coords]; ymin, ymax = min(ys), max(ys); H0 = ymax - ymin
xs = [c[0] for c in coords]
# side nodes at mid height, mid depth: leftmost and rightmost points within the middle band
band = [i for i, (x, y) in enumerate(coords) if abs(y - (ymin + ymax) / 2) < 0.2 * H0]
iL = min(band, key=lambda i: coords[i][0]); iR = max(band, key=lambda i: coords[i][0])
kz = nz // 2
side_nodes = [nid(iL, kz), nid(iR, kz)]
f20 = lambda *v: "".join(f"{x:20.10g}" for x in v)
i10 = lambda *v: "".join(f"{int(x):10d}" for x in v)
B10, B20 = " " * 10, " " * 20
rho, E, nu = 2.70e-3, 68850.0, 0.33
A, B, n, C, eps0 = 285.0, 199.0, 0.532, 0.002, 1.0e-3
W = len(nodes) + 1
tend = a.strain * H0 / a.v
d = ["#RADIOSS STARTER\n/BEGIN\n" + f"{a.name:<80s}\n" + i10(2024) + "\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}",
     "/TITLE\nRe-entrant AA6061-T6 lattice, in-plane compression",
     "/MAT/PLAS_JOHNS/1\nAA6061-T6\n" + f20(rho) + "\n" + f20(E, nu) + i10(0) + "\n" + f20(A, B, n, 0.0, 0.0) + "\n" +
     f20(C, eps0) + i10(1, 1) + f20(10.0, 0.0) + "\n" + f20(0, 0, 0, 0),
     "/PROP/SHELL/1\nRIBS\n" + i10(24, 0, 0, 0, 0) + B10 + f20(0) + "\n" + f20(0, 0, 0, 0, 0) + "\n" + i10(5) + B10 + f20(a.tc, 0) + B10 + i10(1, 1, 0),
     "/PART/1\nLATTICE\n" + i10(1, 1, 0),
     "/NODE\n" + "\n".join(i10(i) + f20(x, y, z) for i, x, y, z in nodes) + "\n" + i10(W) + f20(sum(xs) / len(xs), ymax + 0.5, a.depth / 2),
     "/SHELL/1\n" + "\n".join(i10(*s) for s in shells),
     "/GRNOD/PART/2\nALL\n" + i10(1),
     "/SURF/PART/3\nALL\n" + i10(1),
     "/INTER/TYPE7/1\nSELF_CONTACT\n" + i10(2, 3, 4, 0, 0) + B10 + i10(0, 0, 0, 0) + "\n" + f20(0, 0, 0) + B20 + i10(0) + "\n" +
     f20(0, 0, 0, 0) + i10(0, 0) + "\n" + f20(1, 0.15, 0.5 * a.tc, 0, 0) + "\n" + "       000" + B20 + i10(0) + f20(0, 0, 0) + "\n" +
     i10(0, 0) + f20(0) + i10(0, 0, 0) + f20(0) + i10(0),
     # all nodes: no out-of-plane (z) motion -> plane-strain-like lattice
     "/BCS/1\nNO_Z\n   001 000" + i10(0, 2),
     "/RWALL/PLANE/1\nBOTTOM_FIXED\n" + i10(0, 0, 2, 0) + "\n" + f20(0, 0.15, 0) + "\n" + f20(0.0, ymin - 0.5, 0.0) + "\n" + f20(0.0, ymin + 0.5, 0.0),
     "/RWALL/PLANE/2\nTOP_MOVING\n" + i10(W, 0, 2, 0) + "\n" + f20(0, 0.15, 0) + "\n" + f20(1.0e9, 0, -a.v, 0) + "\n" + f20(sum(xs) / len(xs), ymax - 0.5, a.depth / 2),
     "/TH/RWALL/1\nWALLS\nDEF\n" + i10(1, 2),
     "/TH/NODE/2\nSIDES_AND_WALL\nDEF\n" + "\n".join(i10(k, 0) for k in side_nodes + [W])]
if a.anim:
    d.append("/ATH/NODE/4\nANIM_NODES\nDX        DY        DZ\n" + "\n".join(i10(i, 0) for i, *_ in nodes))
d.append("/END")
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, f"{a.name}_0000.rad"), "w").write("\n".join(d) + "\n")
open(os.path.join(a.out, f"{a.name}_0001.rad"), "w").write(f"/RUN/{a.name}/1\n{tend:.3f}\n/VERS/2024\n/TFILE/4\n{tend/500:.4f}\n/ATFILE/4\n{tend/40:.4f}\n/PRINT/-5000\n/RFILE/0\n10000000\n")
json.dump(dict(vars(a), H0=H0, W0=coords[iR][0] - coords[iL][0], side_nodes=side_nodes, wall_node=W, tend=tend,
               n_shells=len(shells), cell_width=px, row_pitch=pitch), open(os.path.join(a.out, f"{a.name}_info.json"), "w"), indent=1)
print(f"shells {len(shells)} H0 {H0:.1f} W0 {coords[iR][0]-coords[iL][0]:.1f} tend {tend:.1f}")

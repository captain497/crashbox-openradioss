#!/usr/bin/env python3
"""Virtual uniaxial tensile test of the AA6061-T6 material card used in the crash box model.

Shell strip (gauge 50 x 12.5 x 2 mm), left end held, right end pulled by a rigid body at a
constant slow speed. Same /MAT/PLAS_JOHNS card and shell formulation as gen_crashbox.py.
Units g, mm, ms.
"""
import argparse, os
p = argparse.ArgumentParser()
p.add_argument("--name", default="COUPON"); p.add_argument("--out", default=".")
p.add_argument("--speed", type=float, default=0.1)   # mm/ms (= m/s); 0.1 -> strain rate 2 /s over 50 mm
p.add_argument("--disp", type=float, default=6.0)    # mm pulled (12% nominal strain)
p.add_argument("--h", type=float, default=1.25)
a = p.parse_args()
Lg, Wd, T = 50.0, 12.5, 2.0
nx, ny = round(Lg / a.h), round(Wd / a.h)
def nid(i, j): return j * (nx + 1) + i + 1
nodes = [(nid(i, j), i * Lg / nx, j * Wd / ny, 0.0) for j in range(ny + 1) for i in range(nx + 1)]
shells = []; e = 0
for j in range(ny):
    for i in range(nx):
        e += 1; shells.append((e, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))
left = [nid(0, j) for j in range(ny + 1)]; right = [nid(nx, j) for j in range(ny + 1)]
M = len(nodes) + 1   # rigid body main node at the right end
f20 = lambda *v: "".join(f"{x:20.10g}" for x in v)
i10 = lambda *v: "".join(f"{int(x):10d}" for x in v)
B10, B20 = " " * 10, " " * 20
idl = lambda ids: "\n".join(i10(*ids[k:k + 10]) for k in range(0, len(ids), 10))
rho, E, nu = 2.70e-3, 68850.0, 0.33
A, B, n, C, eps0 = 285.0, 199.0, 0.532, 0.002, 1.0e-3
tend = a.disp / a.speed + 1.0
d = []
d.append("#RADIOSS STARTER\n/BEGIN\n" + f"{a.name:<80s}\n" + i10(2024) + "\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}\n" + f"{'g':>20s}{'mm':>20s}{'ms':>20s}")
d.append("/TITLE\nAA6061-T6 virtual tensile coupon")
d.append("/MAT/PLAS_JOHNS/1\nAA6061-T6\n" + f20(rho) + "\n" + f20(E, nu) + i10(0) + "\n" + f20(A, B, n, 0.0, 0.0) + "\n" +
         f20(C, eps0) + i10(1, 1) + f20(10.0, 0.0) + "\n" + f20(0, 0, 0, 0))
d.append("/PROP/SHELL/1\nCOUPON\n" + i10(24, 0, 0, 0, 0) + B10 + f20(0) + "\n" + f20(0, 0, 0, 0, 0) + "\n" + i10(5) + B10 + f20(T, 0) + B10 + i10(1, 1, 0))
d.append("/PART/1\nCOUPON\n" + i10(1, 1, 0))
d.append("/NODE\n" + "\n".join(i10(i) + f20(x, y, z) for i, x, y, z in nodes) + "\n" + i10(M) + f20(Lg, Wd / 2, 0))
d.append("/SHELL/1\n" + "\n".join(i10(*s) for s in shells))
d.append("/GRNOD/NODE/1\nLEFT\n" + idl(left))
d.append("/BCS/1\nLEFT_HELD_X\n   100 000" + i10(0, 1))
d.append("/GRNOD/NODE/4\nLEFT_MID\n" + i10(nid(0, ny // 2)))
d.append("/BCS/4\nLEFT_MID_HELD\n   111 000" + i10(0, 4))
d.append("/GRNOD/NODE/2\nRIGHT\n" + idl(right))
d.append("/RBODY/1\nGRIP\n" + i10(M, 0, 0, 0) + f20(0.0) + i10(2, 0, 0, 0) + "\n" + f20(0, 0, 0) + "\n" + f20(0, 0, 0) + "\n" + i10(0, 0, 0))
d.append("/GRNOD/NODE/3\nGRIP_MAIN\n" + i10(M))
d.append("/BCS/3\nGRIP_GUIDE\n   011 111" + i10(0, 3))
d.append("/FUNCT/1\nRAMP\n" + f20(0, 0) + "\n" + f20(1.0, 1.0) + "\n" + f20(1e6, 1.0))
d.append("/IMPVEL/1\nPULL\n" + i10(1) + "         X" + i10(0, 0, 3, 0, 0) + "\n" + f20(0, a.speed, 0, 0))
d.append("/TH/RBODY/1\nGRIP\nDEF\n" + i10(1))
d.append("/TH/NODE/2\nGRIP_NODE\nDEF\n" + i10(M, 0))
mid = shells[(ny // 2) * nx + nx // 2][0]
d.append("/TH/SHEL/3\nMID_ELEMENT\nDEF\n" + i10(mid))
d.append("/END")
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, f"{a.name}_0000.rad"), "w").write("\n".join(d) + "\n")
open(os.path.join(a.out, f"{a.name}_0001.rad"), "w").write(f"/RUN/{a.name}/1\n{tend}\n/VERS/2024\n/TFILE/4\n{tend/600:.4f}\n/PRINT/-5000\n/ANIM/DT\n0 {tend/30:.4f}\n/ANIM/ELEM/EPSP\n/ANIM/ELEM/VONM\n/RFILE/0\n10000000\n")
print(f"nodes {len(nodes)+1} shells {len(shells)} tend {tend} ms")

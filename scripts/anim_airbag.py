#!/usr/bin/env python3
"""Side view (x-z, car frame) animation of the occupant block and the cushion from the solver's nodal displacements."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
p = argparse.ArgumentParser(); p.add_argument("case_dir"); p.add_argument("--case", required=True); p.add_argument("--label", default="")
a = p.parse_args()
nodes, ath, sec = {}, [], None
part_of, shells = {}, {}
for line in open(os.path.join(a.case_dir, f"{a.case}_0000.rad")):
    if line.startswith("/"): sec = line.strip(); continue
    if line.startswith("#") or not line.strip(): continue
    if sec == "/NODE": nodes[int(line[:10])] = np.array([float(line[10:30]), float(line[30:50]), float(line[50:70])])
    elif sec in ("/SHELL/1", "/SHELL/2"):
        v = [int(line[i:i + 10]) for i in range(0, 50, 10)]; shells.setdefault(sec[-1], []).append(v[1:])
    elif sec and sec.startswith("/ATH/NODE") and line[:10].strip().isdigit(): ath.append(int(line[:10]))
idx = {n: k for k, n in enumerate(ath)}
X0 = np.array([nodes[n] for n in ath])
t, blocks = th.time_series(os.path.join(a.case_dir, f"{a.case}T01a"))
frames = [np.asarray(b[-1]).reshape(-1, 3) for b in blocks]
info = json.load(open(os.path.join(a.case_dir, f"{a.case}_info.json")))
occ = sorted({n for q in shells["1"] for n in q}); bag = sorted({n for q in shells.get("2", []) for n in q})
io = [idx[n] for n in occ]; ib = [idx[n] for n in bag]
# mid-plane slice of the cushion (|y| small) for a clear outline
yb = X0[ib, 1]; sl = [k for k, y in zip(ib, yb) if abs(y) < 20.0]
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=110)
ax.set_xlim(-300, 900); ax.set_ylim(0, 700); ax.set_aspect("equal"); ax.set_xlabel("x, forward (mm, car frame)"); ax.set_ylabel("z (mm)")
ax.axvline(info["gap"] + 480.0 * 0.925, color="#555", lw=3)   # steering-wheel side mount line
po, = ax.fill([], [], color="#e08a1e", alpha=.8, label="occupant (rigid, 40 kg)")
pb = ax.scatter([], [], s=4, c="#2a6fdb", label="cushion, mid-plane nodes")
ax.legend(loc="upper right", frameon=False, fontsize=7); ttl = ax.set_title("", fontsize=8)
def draw(k):
    X = X0 + frames[k]
    xo, zo = X[io, 0], X[io, 2]
    po.set_xy(np.array([[xo.min(), zo.min()], [xo.max(), zo.min()], [xo.max(), zo.max()], [xo.min(), zo.max()]]))
    if sl: pb.set_offsets(np.column_stack([X[sl, 0], X[sl, 2]]))
    ttl.set_text(f"{a.label or a.case}   t = {t[k]:.0f} ms")
    return po, pb
anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
anim.save(os.path.join(a.case_dir, f"{a.case}_airbag.gif"), writer=PillowWriter(fps=10))
k = int(np.argmax([f[io, 0].mean() for f in frames])); draw(k); fig.savefig(os.path.join(a.case_dir, f"{a.case}_max_excursion.png"))

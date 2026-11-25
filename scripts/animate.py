#!/usr/bin/env python3
"""Animation of the deforming tube from real solver output.

Nodal displacements DX DY DZ of every tube node are written by the solver to the
<case>T01a file (/ATH/NODE, every 0.5 ms). This script adds them to the initial
coordinates from the deck and renders the shell mesh frame by frame (GIF).
Colour = local fold depth proxy: radial inward/outward displacement of each node.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th

p = argparse.ArgumentParser()
p.add_argument("case_dir"); p.add_argument("--case", default="BASE")
p.add_argument("--out", default=None); p.add_argument("--fps", type=int, default=8)
a = p.parse_args()
out = a.out or a.case_dir

# --- initial nodes and shells from the deck ---
nodes, shells, sec = {}, [], None
with open(os.path.join(a.case_dir, f"{a.case}_0000.rad")) as f:
    for line in f:
        if line.startswith("/"):
            sec = line.strip().split("/")[1]; continue
        if line.startswith("#"):
            continue
        if sec == "NODE":
            nodes[int(line[:10])] = (float(line[10:30]), float(line[30:50]), float(line[50:70]))
        elif sec == "SHELL":
            v = [int(line[i:i + 10]) for i in range(0, 50, 10)]
            shells.append(v[1:5])
ath_ids = sorted(i for i in nodes if i != max(nodes))   # all tube nodes, deck order
idx = {nid: k for k, nid in enumerate(ath_ids)}
X0 = np.array([nodes[i] for i in ath_ids])
Q = np.array([[idx[n] for n in s] for s in shells])

t, blocks = th.time_series(os.path.join(a.case_dir, f"{a.case}T01a"))
frames = []
for b in blocks:
    D = b[-1].reshape(-1, 3)
    frames.append(X0 + D)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter

r0 = np.maximum(np.abs(X0[:, 0]), np.abs(X0[:, 1]))
fig = plt.figure(figsize=(4.2, 5.6), dpi=110)
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
ax = fig.add_subplot(111, projection="3d")
cmap = plt.get_cmap("coolwarm")

def draw(k):
    ax.cla()
    X = frames[k]
    r = np.maximum(np.abs(X[:, 0]), np.abs(X[:, 1])) - r0
    c = cmap(np.clip(0.5 + r[Q].mean(axis=1) / 20.0, 0, 1))
    pc = Poly3DCollection(X[Q], facecolors=c, edgecolors=(0, 0, 0, 0.15), linewidths=0.1)
    ax.add_collection3d(pc)
    ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_zlim(0, 190)
    ax.set_box_aspect((90, 90, 190), zoom=1.1); ax.view_init(elev=12, azim=35)
    ax.set_axis_off()
    ax.set_title(f"{a.case}  t = {t[k]:.1f} ms\ncrush = {-(X[:, 2].max() - X0[:, 2].max()):.0f} mm", fontsize=9)

anim = FuncAnimation(fig, draw, frames=len(frames))
anim.save(os.path.join(out, f"{a.case}_crush.gif"), writer=PillowWriter(fps=a.fps))
# final deformed shape as a still
draw(len(frames) - 1); fig.savefig(os.path.join(out, f"{a.case}_final_shape.png"), dpi=150)
print(f"{len(frames)} frames, t = {t[0]:.2f} to {t[-1]:.2f} ms")

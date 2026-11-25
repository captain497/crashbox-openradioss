#!/usr/bin/env python3
"""2D front-view animation of the re-entrant lattice compression from the solver's nodal displacements."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
p = argparse.ArgumentParser(); p.add_argument("case_dir"); p.add_argument("--case", default="CELL"); a = p.parse_args()
nodes, shells, ath, sec = {}, [], [], None
for line in open(os.path.join(a.case_dir, f"{a.case}_0000.rad")):
    if line.startswith("/"): sec = line.strip(); continue
    if line.startswith("#"): continue
    if sec == "/NODE": nodes[int(line[:10])] = (float(line[10:30]), float(line[30:50]), float(line[50:70]))
    elif sec == "/SHELL/1": v = [int(line[i:i + 10]) for i in range(0, 50, 10)]; shells.append(v[1:])
    elif sec and sec.startswith("/ATH/NODE") and line[:10].strip().isdigit(): ath.append(int(line[:10]))
idx = {n: k for k, n in enumerate(ath)}
segs = [(s[0], s[1]) for s in shells if abs(nodes[s[0]][2]) < 1e-6 and abs(nodes[s[1]][2]) < 1e-6]
X0 = np.array([nodes[n][:2] for n in ath])
t, blocks = th.time_series(os.path.join(a.case_dir, f"{a.case}T01a"))
frames = [np.asarray(b[-1]).reshape(-1, 3)[:, :2] for b in blocks]
info = json.load(open(os.path.join(a.case_dir, f"{a.case}_info.json")))
fig, ax = plt.subplots(figsize=(5, 5.4), dpi=110)
lc = LineCollection([], colors="#2a6fdb", linewidths=1.6); ax.add_collection(lc)
xmin, ymin = X0.min(0) - 15; xmax, ymax = X0.max(0) + 15
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
top = ax.axhline(X0[:, 1].max(), color="#555", lw=3); ax.axhline(X0[:, 1].min(), color="#555", lw=3)
for xv in (X0[:, 0].min(), X0[:, 0].max()): ax.axvline(xv, color="#d9534f", ls=":", lw=1)
ttl = ax.set_title("")
def draw(k):
    X = X0 + frames[k]
    lc.set_segments([[X[idx[i]], X[idx[j]]] for i, j in segs if i in idx and j in idx])
    top.set_ydata([X[:, 1].max()] * 2)
    ttl.set_text(f"Re-entrant lattice, t = {t[k]:.1f} ms\nred dotted = original width (sides pull inward: auxetic)")
    return lc,
anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
anim.save(os.path.join(a.case_dir, f"{a.case}_compression.gif"), writer=PillowWriter(fps=8))
draw(len(frames) - 1); fig.savefig(os.path.join(a.case_dir, f"{a.case}_final_2d.png"))

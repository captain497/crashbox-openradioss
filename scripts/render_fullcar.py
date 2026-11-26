#!/usr/bin/env python3
"""3D animation of the full-car run from the solver's nodal displacements (/ATH). Left doors and windscreen are drawn
see-through so the occupant and the airbag are visible."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p = argparse.ArgumentParser(); p.add_argument("case_dir"); p.add_argument("--case", required=True); p.add_argument("--label", default="")
p.add_argument("--view", default="iso", choices=["iso", "side", "top", "structure"]); p.add_argument("--still", action="store_true"); p.add_argument("--azim", type=float); p.add_argument("--elev", type=float); a = p.parse_args()
nodes, sh, sec = {}, {}, None
for line in open(os.path.join(a.case_dir, f"{a.case}_0000.rad")):
    if line.startswith("/"): sec = line.strip(); continue
    if line.startswith("#") or not line.strip(): continue
    if sec == "/NODE": nodes[int(line[:10])] = (float(line[10:30]), float(line[30:50]), float(line[50:70]))
    elif sec and sec.startswith("/SHELL/"):
        v = [int(line[k:k + 10]) for k in range(0, 50, 10)]; sh.setdefault(int(sec.split("/")[2]), []).append(v[1:])
N = len(nodes); X0 = np.array([nodes[k] for k in range(1, N + 1)])
t, blocks = th.time_series(os.path.join(a.case_dir, f"{a.case}T01a"))
frames = [np.asarray(b[-1]).reshape(-1, 3) for b in blocks]
style = {1: ("#2a6fdb", 1.0), 2: ("#e08a1e", 1.0), 5: ("#555555", 1.0), 6: ("#8c564b", 1.0), 7: ("#c0392b", 1.0), 9: ("#b03a2e", 0.9),
         10: ("#b03a2e", 0.06), 11: ("#9ecae1", 0.12), 12: ("#222222", 1.0), 13: ("#f2b880", 1.0), 14: ("#f7f7f7", 0.9)}
order = [9, 10, 11, 12, 6, 5, 1, 2, 7, 13, 14] if a.view != "structure" else [5, 6, 1, 2]
fig = plt.figure(figsize=(10, 6), dpi=100); ax = fig.add_subplot(projection="3d")
def draw(k):
    ax.cla(); X = X0 + frames[k]
    for pp in order:
        if pp not in sh: continue
        q = np.array(sh[pp]) - 1; col, al = style[pp]
        pc = Poly3DCollection(X[q], facecolor=col, edgecolor=(0, 0, 0, 0.08 if al > 0.5 else 0.0), linewidths=0.1, alpha=al)
        ax.add_collection3d(pc)
    yy, zz = (np.meshgrid([-700, 700], [250, 750]) if a.view == "structure" else np.meshgrid([-900, 900], [0, 1500])); ax.plot_surface(np.full_like(yy, 4.0), yy, zz, color="#777", alpha=0.35)
    if a.view == "structure":
        ax.set_xlim(-950, 100); ax.set_ylim(-700, 700); ax.set_zlim(250, 750); ax.set_box_aspect((1050, 1400, 500), zoom=1.3)
    else:
        ax.set_xlim(-3900, 300); ax.set_ylim(-1300, 1300); ax.set_zlim(-200, 1700); ax.set_box_aspect((4200, 2600, 1900), zoom=1.45)
    el, az = {"iso": (16, -60), "side": (3, 90), "top": (89, 90), "structure": (35, 60)}[a.view]; ax.view_init(a.elev if a.elev is not None else el, a.azim if a.azim is not None else az); ax.set_axis_off()
    ax.set_title(f"{a.label or a.case}   t = {t[k]:.0f} ms", fontsize=10)
if not a.still:
  anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
  anim.save(os.path.join(a.case_dir, f"{a.case}_{a.view}.gif"), writer=PillowWriter(fps=8))
for k in (0, len(frames) // 2, len(frames) - 1):
    draw(k); fig.savefig(os.path.join(a.case_dir, f"{a.case}_{a.view}_t{t[k]:03.0f}ms.png"))

#!/usr/bin/env python3
"""Vertical (9:16) MP4 crash video of a full-car run, in the style of commercial crash-simulation clips: dark background,
plastic-strain contours on the deforming structure, several camera angles, slow motion.
Every frame is drawn from the solver's own output: nodal displacements (/ATH, *T01a) and shell plastic strain (/BTH, *T01b).
No frames are interpolated; slow motion comes only from playing the saved solver states slowly."""
import argparse, os, struct, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

p = argparse.ArgumentParser(); p.add_argument("case_dir"); p.add_argument("--case", required=True)
p.add_argument("--title", default="Full car into rigid wall, 55 km/h"); p.add_argument("--subtitle", default="")
p.add_argument("--fps", type=float, default=12.0); p.add_argument("--vmax", type=float, default=0.3)
p.add_argument("--out"); p.add_argument("--stills", action="store_true"); p.add_argument("--shots", default="side,iso,front,cabin"); a = p.parse_args()

nodes, sh, eids, sec = {}, {}, {}, None
for line in open(os.path.join(a.case_dir, f"{a.case}_0000.rad")):
    if line.startswith("/"): sec = line.strip(); continue
    if line.startswith("#") or not line.strip(): continue
    if sec == "/NODE": nodes[int(line[:10])] = (float(line[10:30]), float(line[30:50]), float(line[50:70]))
    elif sec and sec.startswith("/SHELL/"):
        v = [int(line[k:k + 10]) for k in range(0, 50, 10)]; pp = int(sec.split("/")[2])
        sh.setdefault(pp, []).append(v[1:]); eids.setdefault(pp, []).append(v[0])
X0 = np.array([nodes[k] for k in range(1, len(nodes) + 1)])
t, blocks = th.time_series(os.path.join(a.case_dir, f"{a.case}T01a"))
frames = [np.asarray(b[-1]).reshape(-1, 3) for b in blocks]
# shell results: element ids in the solver's output order come from the file header
fb = os.path.join(a.case_dir, f"{a.case}T01b"); recs = list(th.records(fb))
g = next(k for k, r in enumerate(recs) if len(r) == 60 and struct.unpack(">5i", r[:20])[1:3] == (3, 0) and struct.unpack(">5i", r[:20])[4] == 10)
ng = struct.unpack(">5i", recs[g][:20])[3]; ids = [struct.unpack(">i", recs[g + 1 + i][:4])[0] for i in range(ng)]; pos = {e: i for i, e in enumerate(ids)}
tb, bb = th.time_series(fb); eps = np.array([np.asarray(blk[-1]).reshape(-1, 10)[:, 9] for blk in bb])
nf = min(len(frames), len(eps)); frames, eps, t = frames[:nf], np.maximum.accumulate(eps[:nf], axis=0), t[:nf]
DEF = [1, 2, 5, 6, 7]; field = {pp: eps[:, [pos[e] for e in eids[pp]]] for pp in DEF if pp in eids}
cmap = plt.get_cmap("jet"); norm = matplotlib.colors.Normalize(0, a.vmax)
style = {9: ("#c9ccd1", 0.55), 10: ("#c9ccd1", 0.07), 11: ("#9ecae1", 0.15), 12: ("#1a1a1a", 1.0), 13: ("#f2b880", 1.0), 14: ("#ffffff", 0.85)}
order = [9, 10, 11, 12, 6, 5, 7, 1, 2, 13, 14]
# shots: (label, elev, azim, xlim, ylim, zlim, zoom, parts)
ALL = order; FRONT = [5, 6, 1, 2, 7]
SHOTS = {"side": ("Side view", 2, -90, (-3900, 400), (-1300, 1300), (-300, 1700), 1.15, ALL),
         "iso": ("Front three-quarter view", 18, -50, (-3900, 400), (-1300, 1300), (-300, 1700), 1.1, ALL),
         "front": ("Front structure: bumper, crash boxes, rails", 30, -130, (-1000, 150), (-750, 750), (250, 850), 1.0, [5, 6, 1, 2]),
         "cabin": ("Occupant, belt and airbag (doors hidden)", 10, -75, (-2400, -300), (-600, 1000), (200, 1300), 1.1, [9, 11, 13, 14, 7, 5, 6, 1, 2])}
BG = "#0d1117"
fig = plt.figure(figsize=(5.4, 9.6), dpi=200, facecolor=BG)
ax = fig.add_axes([0.0, 0.18, 1.0, 0.64], projection="3d"); ax.set_facecolor(BG)
cax = fig.add_axes([0.15, 0.12, 0.7, 0.018])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.set_label("Effective plastic strain", color="w", fontsize=9); cb.ax.tick_params(colors="w", labelsize=8)
txt = dict(color="w", ha="center")
fig.text(0.5, 0.95, a.title, fontsize=13, weight="bold", **txt)
if a.subtitle: fig.text(0.5, 0.925, a.subtitle, fontsize=9, **txt)
h_shot = fig.text(0.5, 0.875, "", fontsize=10, color="#9fd3ff", ha="center")
h_time = fig.text(0.5, 0.835, "", fontsize=16, weight="bold", **txt)
fig.text(0.5, 0.045, f"OpenRadioss explicit FE run (own solver build). Saved solver states every {np.median(np.diff(t)):.0f} ms,\n"
         "played in slow motion, no interpolated frames. Simplified car body, not a validated vehicle model.", fontsize=6.5, color="#aaaaaa", ha="center")

def draw(shot, k):
    lab, el, az, xl, yl, zl, zoom, parts = SHOTS[shot]
    ax.cla(); ax.set_facecolor(BG); X = X0 + frames[k]
    for pp in [q for q in order if q in parts]:
        if pp not in sh: continue
        if shot == "cabin" and pp == 9: continue
        q = np.array(sh[pp]) - 1
        if pp in field: pc = Poly3DCollection(X[q], facecolors=cmap(norm(field[pp][k])), edgecolor=(0, 0, 0, 0.25), linewidths=0.08)
        else:
            col, al = style[pp]; pc = Poly3DCollection(X[q], facecolor=col, edgecolor=(0, 0, 0, 0.1 if al > 0.5 else 0.0), linewidths=0.08, alpha=al)
        ax.add_collection3d(pc)
    yy, zz = np.meshgrid(np.linspace(*yl, 2), np.linspace(zl[0], zl[1], 2)); ax.plot_surface(np.full_like(yy, 4.0), yy, zz, color="#6b6f76", alpha=0.3)
    gx, gy = np.meshgrid(np.linspace(xl[0], 4, 2), np.linspace(*yl, 2))
    if shot == "side": ax.plot_surface(gx, gy, np.full_like(gx, -300.0), color="#2a2f36", alpha=0.9)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl); ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0]), zoom=zoom)
    ax.view_init(el, az); ax.set_axis_off()
    h_shot.set_text(lab); h_time.set_text(f"t = {t[k]:5.1f} ms")

if a.stills:
    for shot in a.shots.split(","):
        draw(shot, len(frames) - 1); fig.savefig(os.path.join(a.case_dir, f"{a.case}_still_{shot}.png"), facecolor=BG, dpi=60)
    sys.exit()
tmp = tempfile.mkdtemp(); n = 0
for shot in a.shots.split(","):
    seq = list(range(len(frames))) + [len(frames) - 1] * int(a.fps)       # hold the last state for 1 s
    for k in seq:
        draw(shot, k); fig.savefig(os.path.join(tmp, f"f{n:05d}.png"), facecolor=BG); n += 1
out = a.out or os.path.join(a.case_dir, f"{a.case}_crash_video.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%05d.png"), "-vf", "scale=1080:1920:flags=lanczos,format=yuv420p",
                "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart", out], check=True)
print(out, n, "frames")

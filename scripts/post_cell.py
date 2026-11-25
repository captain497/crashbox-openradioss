#!/usr/bin/env python3
"""Re-entrant lattice compression: nominal stress-strain and Poisson's ratio from the solver output."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
p = argparse.ArgumentParser(); p.add_argument("case_dir"); p.add_argument("--case", default="CELL"); a = p.parse_args()
info = json.load(open(os.path.join(a.case_dir, f"{a.case}_info.json")))
t, b = th.time_series(os.path.join(a.case_dir, f"{a.case}T01"))
Wl = np.array([x[1] for x in b]); N = np.array([x[2] for x in b])
dxL, dxR, dyW = N[:, 0], N[:, 6], N[:, 13]
H0, W0 = info["H0"], info["W0"]
ey = -dyW / H0                                   # nominal compressive strain (positive)
ex = (dxR - dxL) / W0                            # lateral strain (negative = contracts)
F = np.abs(np.gradient(Wl[:, 7], t))             # top wall normal force, N (impulse differentiated)
area = W0 * info["depth"]
s = F / area
nu = np.where(ey > 1e-4, ex / np.maximum(ey, 1e-9), np.nan)   # nu = -(lateral)/(axial), axial strain negative in compression
def at(x): return float(np.interp(x, ey, nu))
out = dict(case=a.case, theta=info["theta"], beta=info["beta"], tc=info["tc"], H0=H0, W0=W0,
           poisson_at_2pct=at(0.02), poisson_at_5pct=at(0.05), poisson_at_10pct=at(0.10), poisson_at_20pct=at(0.20),
           lateral_strain_at_20pct=float(np.interp(0.2, ey, ex)), peak_stress_MPa=float(s[ey < 0.2].max()),
           plateau_stress_10_30pct_MPa=float(s[(ey > 0.1) & (ey < 0.3)].mean()) if (ey > 0.3).any() else None)
json.dump(out, open(os.path.join(a.case_dir, f"{a.case}_auxetic_check.json"), "w"), indent=1)
np.savetxt(os.path.join(a.case_dir, f"{a.case}_curves.csv"), np.column_stack([t, ey, ex, s]), delimiter=",",
           header="time_ms,axial_strain,lateral_strain,nominal_stress_MPa", comments="")
print(json.dumps(out, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)
axs[0].plot(ey * 100, s, color="#2a6fdb"); axs[0].set_xlabel("Compressive strain (%)"); axs[0].set_ylabel("Nominal stress (MPa)"); axs[0].grid(alpha=.3)
axs[1].plot(ey * 100, ex * 100, color="#2a6fdb", label="lateral strain at mid-height")
axs[1].axhline(0, color="#888", lw=0.8); axs[1].set_xlabel("Compressive strain (%)"); axs[1].set_ylabel("Lateral strain (%)"); axs[1].grid(alpha=.3)
axs[1].set_title(f"Poisson's ratio: {out['poisson_at_5pct']:.2f} at 5%, {out['poisson_at_20pct']:.2f} at 20%", fontsize=8)
fig.suptitle(f"Re-entrant AA6061-T6 lattice in compression (theta {info['theta']:.0f} deg, beta {info['beta']}, rib {info['tc']} mm)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(a.case_dir, f"{a.case}_auxetic_check.png"))

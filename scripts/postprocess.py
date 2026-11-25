#!/usr/bin/env python3
"""Crash metrics from an OpenRadioss crash box run (real solver output only).

Reads <case>T01 (binary TH). Rigid-wall forces are stored as impulses and are
differentiated in time. Layout per step, from the deck generator:
  global(22), part TUBE(9), rigid wall(6: FNX FNY FNZ FTX FTY FTZ),
  wall node(6: DX DY DZ VX VY VZ).
Units: g-mm-ms -> force N, displacement mm, energy mJ.

Outputs <case>_force_disp.csv, <case>_metrics.json and plots.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th

p = argparse.ArgumentParser()
p.add_argument("case_dir")
p.add_argument("--case", default="BASE")
p.add_argument("--tube_mass_g", type=float, required=True)
p.add_argument("--length", type=float, default=180.0)
p.add_argument("--stroke_frac", type=float, default=0.7,
               help="effective stroke cap for CFE/MCF when crush runs further")
p.add_argument("--out", default=None)
a = p.parse_args()
out = a.out or a.case_dir
os.makedirs(out, exist_ok=True)

t, blocks = th.time_series(os.path.join(a.case_dir, f"{a.case}T01"))
G = np.array([b[0] for b in blocks]); P = np.array([b[1] for b in blocks])
W = np.array([b[2] for b in blocks]); N = np.array([b[3] for b in blocks])

# Radioss writes rigid-wall forces to TH as cumulative impulses (N*ms);
# force is the time derivative. Differencing over each 0.01 ms output interval
# gives the interval-averaged force (no further filtering).
I_z = np.abs(W[:, 2])
F = np.gradient(I_z, t) / 1000.0           # wall normal force, kN
d = -N[:, 2]                               # crush displacement, mm
v = -N[:, 5]                               # wall speed, m/s
ie_tube = P[:, 0] / 1e3                    # tube internal energy, J
he_tube = P[:, 6] / 1e3 if P.shape[1] > 6 else np.zeros_like(t)

# stop at the point of maximum crush (wall rebound) if it happens
k_end = int(np.argmax(d))
sl = slice(0, k_end + 1)
F, d, v, tt = F[sl], d[sl], v[sl], t[sl]
EA = np.concatenate([[0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(d))])  # kN*mm = J
PCF = float(F.max()); d_pcf = float(d[F.argmax()])
dmax = float(d[-1])
MCF = float(EA[-1] / dmax) if dmax > 0 else float("nan")
SEA = float(EA[-1] / (a.tube_mass_g / 1000.0)) / 1000.0   # kJ/kg
CFE = MCF / PCF

m = dict(case=a.case, sim_end_time_ms=float(t[-1]), time_at_max_crush_ms=float(tt[-1]),
         rebound_detected=bool(k_end < len(t) - 1),
         max_crush_mm=dmax, crush_ratio=dmax / a.length,
         final_wall_speed_ms=float(v[-1]),
         PCF_kN=PCF, disp_at_PCF_mm=d_pcf, MCF_kN=MCF, EA_J=float(EA[-1]),
         tube_mass_g=a.tube_mass_g, SEA_kJ_per_kg=SEA, CFE=CFE,
         tube_internal_energy_J=float(ie_tube[sl][-1]),
         tube_hourglass_energy_J=float(he_tube[sl][-1]),
         hourglass_ratio=float(he_tube[sl][-1] / max(ie_tube[sl][-1], 1e-9)),
         n_samples=int(len(tt)))
with open(os.path.join(out, f"{a.case}_metrics.json"), "w") as f:
    json.dump(m, f, indent=2)
np.savetxt(os.path.join(out, f"{a.case}_force_disp.csv"),
           np.column_stack([tt, d, F, EA, v]), delimiter=",",
           header="time_ms,crush_mm,force_kN,EA_J,wall_speed_ms", comments="")
print(json.dumps(m, indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
ax[0].plot(d, F, lw=1.0, color="#2a6fdb")
ax[0].axhline(MCF, ls="--", lw=1, color="#d9534f", label=f"MCF = {MCF:.1f} kN")
ax[0].plot([d_pcf], [PCF], "o", color="#333", ms=4, label=f"PCF = {PCF:.1f} kN")
ax[0].set_xlabel("Crush displacement (mm)"); ax[0].set_ylabel("Rigid-wall force (kN)")
ax[0].set_title("Force vs displacement"); ax[0].legend(frameon=False); ax[0].grid(alpha=.3)
ax[1].plot(d, EA, lw=1.2, color="#2a6fdb")
ax[1].set_xlabel("Crush displacement (mm)"); ax[1].set_ylabel("Absorbed energy (J)")
ax[1].set_title(f"Energy absorbed, EA = {EA[-1]:.0f} J"); ax[1].grid(alpha=.3)
fig.suptitle(f"OpenRadioss baseline {a.case}: AA6061-T6 square tube (placeholder inputs)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, f"{a.case}_force_disp.png"))

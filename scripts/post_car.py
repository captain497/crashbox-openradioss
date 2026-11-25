#!/usr/bin/env python3
"""Post-process a car-mode crash box run (gen_crashbox.py --car 1): car crash pulse and crush history.

Reads <case>T01 in <dir>. TH blocks in car mode (file order): PART energies, RWALL, SPRING (rail), NODE (B mount, C car).
Writes <case>_car_pulse.csv, <case>_car_metrics.json and <case>_car.png.
The pulse csv (time_ms, car_decel_g with CFC60) is the input of gen_airbag.py.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
from sae_filter import cfc

p = argparse.ArgumentParser(); p.add_argument("dir"); p.add_argument("--case", required=True); p.add_argument("--plot", type=int, default=1)
a = p.parse_args()
info = json.load(open(os.path.join(a.dir, f"{a.case}_info.json")))
t, b = th.time_series(os.path.join(a.dir, f"{a.case}T01"))
t = np.asarray(t, float)
wall_imp = np.array([s[2][2] for s in b])            # RWALL normal impulse, z (N ms)
spr_F = np.array([s[3][1] for s in b])                # rail spring force (N, negative = compression)
spr_dl = np.array([s[3][7] for s in b])               # rail spring elongation (mm)
dzB = np.array([s[4][2] for s in b]); vzC = np.array([s[4][11] for s in b]); dzC = np.array([s[4][8] for s in b])


dt_s = float(np.median(np.diff(t))) * 1e-3
acc = -np.gradient(vzC, t) / 9.81e-3                  # car deceleration, g (positive = slowing)
acc60 = cfc(acc, dt_s, 60)
Fw = np.gradient(wall_imp, t) / 1000.0                # wall force, kN
Fw600 = cfc(Fw, dt_s, 600)
box_crush = np.maximum(dzB - 0.5, 0.0)                # box tip starts 0.5 mm from the wall
rail_crush = -spr_dl
v0 = info["v"]
i_stop = int(np.argmax(vzC <= 0)) if (vzC <= 0).any() else len(t) - 1
E0 = 0.5 * (info.get("car_kg", 557.0)) * v0 ** 2      # J per crash box side
Ebox = float(np.trapezoid(np.abs(Fw[:i_stop + 1]) * 1000.0, box_crush[:i_stop + 1]) / 1000.0)
Erail = float(np.trapezoid(np.abs(spr_F[:i_stop + 1]), rail_crush[:i_stop + 1]) / 1000.0)
m = dict(case=a.case, core=info.get("core"), v0_m_s=v0, t_stop_ms=float(t[i_stop]), stopped=bool((vzC <= 0).any()),
         car_peak_decel_CFC60_g=float(acc60[:i_stop + 1].max()), car_mean_decel_g=float(v0 / (t[i_stop] * 1e-3) / 9.81) if (vzC <= 0).any() else None,
         car_max_dynamic_crush_mm=float(dzC.max()), box_crush_max_mm=float(box_crush.max()), rail_crush_max_mm=float(rail_crush.max()),
         wall_PCF_CFC600_kN=float(Fw600.max()), rebound_speed_m_s=float(-vzC.min()) if vzC.min() < 0 else 0.0,
         box_energy_J=Ebox, rail_energy_J=Erail, impact_energy_J=E0, rail_kN=info.get("rail_kN"))
json.dump(m, open(os.path.join(a.dir, f"{a.case}_car_metrics.json"), "w"), indent=1)
hdr = "time_ms,car_decel_g,car_decel_raw_g,car_speed_m_s,car_disp_mm,wall_force_CFC600_kN,box_crush_mm,rail_crush_mm,rail_force_kN"
np.savetxt(os.path.join(a.dir, f"{a.case}_car_pulse.csv"),
           np.column_stack([t, acc60, acc, vzC, dzC, Fw600, box_crush, rail_crush, -spr_F / 1000.0]), delimiter=",", header=hdr, comments="", fmt="%.6g")
print(json.dumps(m, indent=1))
if a.plot:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), dpi=140)
    ax[0].plot(t, acc, lw=0.4, c="#bbb", label="raw"); ax[0].plot(t, acc60, c="#2a6fdb", label="CFC60")
    ax[0].set_xlabel("Time (ms)"); ax[0].set_ylabel("Car deceleration (g)"); ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(t, vzC, c="#2a6fdb"); ax[1].set_xlabel("Time (ms)"); ax[1].set_ylabel("Car speed (m/s)")
    ax[2].plot(t, box_crush, label="crash box"); ax[2].plot(t, rail_crush, label="rail (spring)"); ax[2].plot(t, dzC, label="car total")
    ax[2].set_xlabel("Time (ms)"); ax[2].set_ylabel("Crush (mm)"); ax[2].legend(frameon=False, fontsize=7)
    for x in ax: x.grid(alpha=.3)
    fig.suptitle(f"{a.case}: simplified car into rigid wall, 55 km/h, 557 kg per box (OpenRadioss)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.dir, f"{a.case}_car.png")); plt.close(fig)

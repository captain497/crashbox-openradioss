#!/usr/bin/env python3
"""Post-process an occupant/belt/airbag run from gen_airbag.py.

TH blocks (file order): [INTER x2 (bag runs only)], SPRING (belt strap 1), NODE (occupant main), [MONV (bag runs only)].
Occupant motion is relative to the car (car frame). Absolute occupant deceleration = car pulse - relative acceleration.
Restraint force on the occupant = m * absolute deceleration; airbag force = restraint - 2 x belt strap force.
Writes <case>_occupant.csv and <case>_occupant_metrics.json.
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
from sae_filter import cfc

p = argparse.ArgumentParser(); p.add_argument("dir"); p.add_argument("--case", required=True)
a = p.parse_args()
info = json.load(open(os.path.join(a.dir, f"{a.case}_info.json")))
t, b = th.time_series(os.path.join(a.dir, f"{a.case}T01")); t = np.asarray(t, float)
bag = not info["no_bag"]
o = 2 if bag else 1
belt1 = np.array([s[o][1] for s in b]); dx = np.array([s[o + 1][0] for s in b]); vx = np.array([s[o + 1][3] for s in b])
if bag:
    vol = np.array([s[o + 2][1] for s in b]) / 1e6; pg = (np.array([s[o + 2][2] for s in b]) - 0.101325) * 1000.0
pulse = np.loadtxt(info["pulse"], delimiter=",", skiprows=1, usecols=(0, 1))
car_g = np.interp(t, pulse[:, 0], pulse[:, 1], right=0.0)
dt_s = float(np.median(np.diff(t))) * 1e-3
a_rel_g = np.gradient(vx, t) / 9.81e-3
occ_g = cfc(car_g - a_rel_g, dt_s, 180)                       # absolute occupant deceleration, CFC180
F_tot = info["occ_kg"] * occ_g * 9.81 / 1000.0                # kN
F_belt = 2 * belt1 / 1000.0
F_bag = F_tot - F_belt if bag else np.zeros_like(t)
def clip3(x):  # highest level sustained for 3 ms
    n = max(1, int(round(3.0 / np.median(np.diff(t)))))
    return float(max(x[i:i + n].min() for i in range(len(x) - n)))
out = open(os.path.join(a.dir, f"{a.case}_0001.out")).read()
err = re.findall(r"^\s+\d+\s+[\d.]+\s+\S+\s+\S+\s+\d+\s+(-?[\d.]+)%", out, re.M)
m = dict(case=a.case, pulse=os.path.basename(info["pulse"]), airbag=bag, p0_kPa=info["p0_kPa"] if bag else None,
         occ_kg=info["occ_kg"], occ_peak_decel_CFC180_g=float(occ_g.max()), occ_decel_3ms_g=clip3(occ_g),
         max_forward_excursion_mm=float(dx.max()), t_max_excursion_ms=float(t[int(dx.argmax())]),
         peak_belt_force_kN=float(F_belt.max()), peak_airbag_force_kN=float(F_bag.max()) if bag else None,
         peak_bag_pressure_kPa=float(pg.max()) if bag else None, min_bag_volume_L=float(vol.min()) if bag else None,
         rebound_speed_m_s=float(-vx.min()) if vx.min() < 0 else 0.0,
         energy_error_max_abs_pct=float(max(abs(float(e)) for e in err)) if err else None,
         energy_error_final_pct=float(err[-1]) if err else None,
         normal_termination="NORMAL TERMINATION" in out)
json.dump(m, open(os.path.join(a.dir, f"{a.case}_occupant_metrics.json"), "w"), indent=1)
cols = [t, car_g, occ_g, dx, vx, F_belt, F_bag] + ([pg, vol] if bag else [])
hdr = "time_ms,car_decel_g,occupant_decel_CFC180_g,occupant_excursion_mm,occupant_rel_speed_m_s,belt_force_kN,airbag_force_kN" + (",bag_pressure_kPa,bag_volume_L" if bag else "")
np.savetxt(os.path.join(a.dir, f"{a.case}_occupant.csv"), np.column_stack(cols), delimiter=",", header=hdr, comments="", fmt="%.6g")
print(json.dumps(m))

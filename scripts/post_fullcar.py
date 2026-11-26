#!/usr/bin/env python3
"""Post-process a full-car run (gen_fullcar.py): car pulse, crush, wall force, occupant, belt and airbag.
TH blocks (file order): PART energies, RWALL, SPRING (belt strap 1), NODE (cabin main, occupant main), MONV."""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
from sae_filter import cfc
p = argparse.ArgumentParser(); p.add_argument("dir"); p.add_argument("--case", required=True); a = p.parse_args()
info = json.load(open(os.path.join(a.dir, f"{a.case}_info.json")))
t, b = th.time_series(os.path.join(a.dir, f"{a.case}T01")); t = np.asarray(t, float); dt_s = float(np.median(np.diff(t))) * 1e-3
wall = np.array([s[2][0] for s in b]); belt1 = np.array([s[3][1] for s in b])
ic, io = (0, 6) if info["cab_main"] < info["occ_main"] else (6, 0)     # TH nodes come out sorted by node id
cab = np.array([s[4][ic:ic + 6] for s in b]); occ = np.array([s[4][io:io + 6] for s in b]); mv = np.array([s[5] for s in b])
Fw = cfc(np.gradient(np.abs(wall), t) / 1000.0, dt_s, 600)
car_g = cfc(-np.gradient(cab[:, 3], t) / 9.81e-3, dt_s, 60)
occ_g = cfc(-np.gradient(occ[:, 3], t) / 9.81e-3, dt_s, 180)
crush = cab[:, 0]; rel = occ[:, 0] - cab[:, 0]; pg = (mv[:, 2] - 0.101325) * 1000.0; vol = mv[:, 1] / 1e6
v0 = info["v"]; stop = (cab[:, 3] <= 0)
i_stop = int(np.argmax(stop)) if stop.any() else len(t) - 1
n3 = max(1, int(round(3.0 / np.median(np.diff(t)))))
clip3 = lambda x: float(max(x[i:i + n3].min() for i in range(len(x) - n3)))
out = open(os.path.join(a.dir, f"{a.case}_0001.out")).read()
err = [float(e) for e in re.findall(r"^\s+\d+\s+[\d.]+\s+\S+\s+\S+\s+\d+\s+(-?[\d.]+)%", out, re.M)]
m = dict(case=a.case, crash_box=info["box_args"], car_kg=info["car_kg"], v0_m_s=v0, stopped=bool(stop.any()), t_stop_ms=float(t[i_stop]),
         car_max_crush_mm=float(crush.max()), car_peak_decel_CFC60_g=float(car_g.max()), car_mean_decel_g=float(v0 / (t[i_stop] * 1e-3) / 9.81) if stop.any() else None,
         wall_peak_force_CFC600_kN=float(Fw.max()), occupant_peak_decel_CFC180_g=float(occ_g.max()), occupant_decel_3ms_g=clip3(occ_g),
         occupant_max_forward_travel_in_car_mm=float(rel.max()), peak_belt_force_kN=float(2 * belt1.max() / 1000.0),
         peak_bag_pressure_kPa=float(pg.max()), min_bag_volume_L=float(vol.min()), energy_error_max_abs_pct=max(abs(e) for e in err) if err else None,
         normal_termination="NORMAL TERMINATION" in out)
json.dump(m, open(os.path.join(a.dir, f"{a.case}_metrics.json"), "w"), indent=1)
np.savetxt(os.path.join(a.dir, f"{a.case}_history.csv"), np.column_stack([t, car_g, cab[:, 3], crush, Fw, occ_g, rel, 2 * belt1 / 1000.0, pg, vol]), delimiter=",",
           header="time_ms,car_decel_CFC60_g,car_speed_m_s,car_crush_mm,wall_force_CFC600_kN,occupant_decel_CFC180_g,occupant_travel_in_car_mm,belt_force_kN,bag_pressure_kPa,bag_volume_L",
           comments="", fmt="%.6g")
print(json.dumps(m, indent=1))

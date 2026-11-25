#!/usr/bin/env python3
"""Package the material, cell, car-into-wall and airbag results (solver output only) into one folder."""
import argparse, glob, json, os, shutil
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
p = argparse.ArgumentParser(); p.add_argument("--runs", default="/home/user/runs"); p.add_argument("--out", required=True); a = p.parse_args()
R, O = a.runs, a.out
for s in ("1_material_coupon", "2_auxetic_cell", "3_car_into_wall", "4_airbag_pressure_study"): os.makedirs(os.path.join(O, s), exist_ok=True)
cp = lambda pat, sub: [shutil.copy(f, os.path.join(O, sub)) for f in glob.glob(os.path.join(R, pat))]
cp("COUPON/COUPON_material_check.*", "1_material_coupon"); cp("COUPON/COUPON_curves.csv", "1_material_coupon")
for f in ("CELL_auxetic_check.json", "CELL_auxetic_check.png", "CELL_curves.csv", "CELL_compression.gif", "CELL_crush.gif", "CELL_final_2d.png"): cp(f"CELL/{f}", "2_auxetic_cell")
# car
rows = []
for d in sorted(glob.glob(os.path.join(R, "CAR_*"))):
    for f in glob.glob(os.path.join(d, "*_car_metrics.json")):
        m = json.load(open(f)); rows.append(m); c = m["case"]
        for suf in ("_car_pulse.csv", "_car.png", "_crush.gif", "_final_shape.png", "_car_metrics.json"): cp(f"{os.path.basename(d)}/{c}{suf}", "3_car_into_wall")
if rows: pd.DataFrame(rows).to_csv(os.path.join(O, "3_car_into_wall", "car_summary.csv"), index=False)
# airbag
rows = []
for f in sorted(glob.glob(os.path.join(R, "airbag", "AB_*", "*_occupant_metrics.json"))):
    m = json.load(open(f)); rows.append(m); c = m["case"]; d = os.path.dirname(f)
    for suf in ("_occupant.csv", "_airbag.gif", "_max_excursion.png"):
        if os.path.exists(os.path.join(d, c + suf)): shutil.copy(os.path.join(d, c + suf), os.path.join(O, "4_airbag_pressure_study"))
if rows:
    S = pd.DataFrame(rows); S.to_csv(os.path.join(O, "4_airbag_pressure_study", "airbag_summary.csv"), index=False)
    for tag in sorted({c.split("_")[1] for c in S.case}):
        cases = [c for c in S.case if c.split("_")[1] == tag]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), dpi=140)
        for c in sorted(cases, key=lambda c: (c.endswith("BELT"), int(c.split("_P")[-1]) if "_P" in c else 0)):
            D = pd.read_csv(os.path.join(R, "airbag", c, f"{c}_occupant.csv"))
            lab = "belt only" if c.endswith("BELT") else f"airbag p0 = {c.split('_P')[-1]} kPa"
            ax[0].plot(D.time_ms, D.occupant_decel_CFC180_g, lw=1, label=lab)
            ax[1].plot(D.time_ms, D.occupant_excursion_mm, lw=1, label=lab)
            if "bag_pressure_kPa" in D: ax[2].plot(D.time_ms, D.bag_pressure_kPa, lw=1, label=lab)
        ax[0].plot(D.time_ms, D.car_decel_g, "k--", lw=0.8, label="car pulse (CFC60)")
        ax[0].set_ylabel("Occupant deceleration, CFC180 (g)"); ax[1].set_ylabel("Forward excursion vs car (mm)"); ax[2].set_ylabel("Cushion gauge pressure (kPa)")
        for x in ax: x.set_xlabel("Time (ms)"); x.grid(alpha=.3)
        ax[0].legend(frameon=False, fontsize=6)
        fig.suptitle(f"Occupant + belt + airbag driven by the {tag} car crash pulse (OpenRadioss)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(O, "4_airbag_pressure_study", f"airbag_sweep_{tag}.png")); plt.close(fig)
print(sorted(os.listdir(O)))

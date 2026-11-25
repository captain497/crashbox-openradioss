#!/usr/bin/env python3
"""Compare a Simpson & Kazanci (2020) empty-tube validation run with their test data (Table 9)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
p = argparse.ArgumentParser(); p.add_argument("case_dir"); p.add_argument("--case", required=True)
p.add_argument("--crush", type=float, default=120.0); a = p.parse_args()
t, blocks = th.time_series(os.path.join(a.case_dir, f"{a.case}T01"))
W = np.array([b[2] for b in blocks]); N = np.array([b[3] for b in blocks])
F = np.gradient(np.abs(W[:, 2]), t) / 1000.0; d = -N[:, 2]
k = int(np.argmax(d >= a.crush)) if d.max() >= a.crush else len(d) - 1
F_, d_ = F[:k + 1], d[:k + 1]
EA = float(np.sum(0.5 * (F_[1:] + F_[:-1]) * np.diff(d_))) / 1000.0  # kJ
exp = dict(PCF_kN=[82.78, 84.03, 83.56], MCF_kN=[29.04, 25.32, 24.41], EA_kJ=[3.49, 3.04, 2.93])
fe = dict(PCF_kN=float(F_.max()), MCF_kN=EA * 1000 / d_[-1], EA_kJ=EA)
out = dict(case=a.case, crush_mm=float(d_[-1]), rows=[])
for kk in exp:
    m = float(np.mean(exp[kk])); out["rows"].append(dict(metric=kk, test_mean=m, test_min=min(exp[kk]), test_max=max(exp[kk]),
                                                       FE=fe[kk], error_pct=(fe[kk] - m) / m * 100))
json.dump(out, open(os.path.join(a.case_dir, f"{a.case}_validation.json"), "w"), indent=2)
np.savetxt(os.path.join(a.case_dir, f"{a.case}_force_disp.csv"), np.column_stack([t, d, F]), delimiter=",",
           header="time_ms,crush_mm,force_kN", comments="")
print(json.dumps(out, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.6), dpi=130)
ax.plot(d_, F_, lw=0.9, color="#2a6fdb", label=f"OpenRadioss ({a.case})")
ax.axhline(np.mean(exp["MCF_kN"]), ls="--", color="#555", lw=0.8, label="test mean force (avg of 3)")
ax.axhline(np.mean(exp["PCF_kN"]), ls=":", color="#555", lw=0.8, label="test peak force (avg of 3)")
ax.axhline(fe["MCF_kN"], ls="--", color="#2a6fdb", lw=0.8, label="FE mean force")
ax.set_xlabel("Crush (mm)"); ax.set_ylabel("Force (kN)"); ax.set_xlim(0, a.crush); ax.grid(alpha=.3)
ax.legend(frameon=False, fontsize=7); ax.set_title("Simpson & Kazanci (2020) empty mild steel tube: FE vs test", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(a.case_dir, f"{a.case}_validation.png"))

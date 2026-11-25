#!/usr/bin/env python3
"""Post-process the virtual tensile test: true stress vs plastic strain (mid element) and
engineering stress-strain (grip force), compared with values read off Aktas et al. (2023) Fig. 8."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th
p = argparse.ArgumentParser(); p.add_argument("case_dir"); p.add_argument("--case", default="COUPON"); a = p.parse_args()
t, b = th.time_series(os.path.join(a.case_dir, f"{a.case}T01"))
N = np.array([x[1] for x in b]); S = np.array([x[2] for x in b]); R = np.array([x[3] for x in b])
dx = N[:, 0]; sig = S[:, 0]; epsp = S[:, 9]
F = np.abs(np.gradient(R[:, 0], t))                     # N (rigid-body impulse differentiated)
A0, L0 = 12.5 * 2.0, 50.0
e_eng = dx / L0; s_eng = F / A0
# values read by eye off Aktas et al. (2023) Fig. 8b (true stress vs true effective plastic strain), MPa
ak_ep = np.array([0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
ak_s = np.array([285, 293, 297, 303, 309, 315.5, 321, 326.5, 329])
fe_s = np.interp(ak_ep, epsp, sig)
rows = [dict(plastic_strain=float(e), Aktas_fig8b_MPa=float(s), FE_true_stress_MPa=float(f), diff_pct=float((f - s) / s * 100))
        for e, s, f in zip(ak_ep[1:], ak_s[1:], fe_s[1:])]
k = np.argmax(s_eng)
out = dict(case=a.case, rows=rows, FE_yield_MPa=float(np.interp(0.002, e_eng - s_eng / 68850.0, s_eng)),
           FE_UTS_eng_MPa=float(s_eng[k]), FE_strain_at_UTS=float(e_eng[k]), Aktas_yield_MPa=285.0,
           note="Aktas curve values are read by eye from the published figure (approximate).")
json.dump(out, open(os.path.join(a.case_dir, f"{a.case}_material_check.json"), "w"), indent=2)
np.savetxt(os.path.join(a.case_dir, f"{a.case}_curves.csv"), np.column_stack([t, e_eng, s_eng, epsp, sig]), delimiter=",",
           header="time_ms,eng_strain,eng_stress_MPa,plastic_strain_mid,true_stress_mid_MPa", comments="")
print(json.dumps(out, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
axs[0].plot(epsp, sig, color="#2a6fdb", lw=1.5, label="OpenRadioss coupon (mid element)")
axs[0].plot(ak_ep, ak_s, "o", color="#d9534f", ms=4, label="Aktas et al. (2023) Fig. 8b (read off)")
axs[0].set_xlabel("True plastic strain"); axs[0].set_ylabel("True stress (MPa)"); axs[0].set_xlim(0, 0.12); axs[0].grid(alpha=.3); axs[0].legend(frameon=False, fontsize=7)
axs[1].plot(e_eng * 100, s_eng, color="#2a6fdb", lw=1.5, label="OpenRadioss coupon (grip force / A0)")
axs[1].axhline(285, ls=":", color="#d9534f", lw=1, label="Aktas yield 285 MPa")
axs[1].set_xlabel("Engineering strain (%)"); axs[1].set_ylabel("Engineering stress (MPa)"); axs[1].grid(alpha=.3); axs[1].legend(frameon=False, fontsize=7)
fig.suptitle("AA6061-T6 material card: virtual tensile test", fontsize=10); fig.tight_layout()
fig.savefig(os.path.join(a.case_dir, f"{a.case}_material_check.png"))

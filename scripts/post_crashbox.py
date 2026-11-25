#!/usr/bin/env python3
"""Crash metrics for decks from gen_crashbox.py (real solver output only).

TH layout per step: global(22), parts (9 vars each: IE KE XMOM YMOM ZMOM MASS HE
TURBKE ...), rigid wall (FNX FNY FNZ FTX FTY FTZ, cumulative impulses), wall
node (DX DY DZ VX VY VZ). Force = d(impulse)/dt over each output interval.

Metrics are evaluated up to the evaluation stroke s_eval = stroke_frac * L
(the heavy wall would otherwise keep crushing the box solid), or up to the
maximum crush if the wall stops earlier.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread as th

p = argparse.ArgumentParser()
p.add_argument("case_dir"); p.add_argument("--case", required=True)
p.add_argument("--stroke_frac", type=float, default=0.70)
p.add_argument("--plot", type=int, default=1)
a = p.parse_args()
info = json.load(open(os.path.join(a.case_dir, f"{a.case}_info.json")))
L = info["length"]; mass_kg = info["mass_total_g"] / 1000.0
npart = 2 if info["n_shell_core"] > 0 else 1

t, blocks = th.time_series(os.path.join(a.case_dir, f"{a.case}T01"))
P = np.array([b[1] for b in blocks]).reshape(len(t), npart, -1)
W = np.array([b[2] for b in blocks]); N = np.array([b[3] for b in blocks])
G = np.array([b[0] for b in blocks])

F = np.gradient(np.abs(W[:, 2]), t) / 1000.0     # kN
d = -N[:, 2]                                      # mm
s_eval = a.stroke_frac * L
k_max = int(np.argmax(d))
reached = d[k_max] >= s_eval
k_end = int(np.argmax(d >= s_eval)) if reached else k_max
sl = slice(0, k_end + 1)
F_, d_, t_ = F[sl], d[sl], t[sl]
EA = np.concatenate([[0], np.cumsum(0.5 * (F_[1:] + F_[:-1]) * np.diff(d_))])  # J
PCF = float(F_.max()); stroke = float(d_[-1])
MCF = float(EA[-1] / stroke)
IE = P[k_end, :, 0].sum() / 1e3; HE = P[k_end, :, 6].sum() / 1e3
# energy balance error reported by the engine at this step: (IE+KE - KE0 - Wext)/... use globals
cored = info["core"] != "none"
m = dict(case=a.case, core=info["core"], theta=info["theta"] if cored else None,
         ncell=info["ncell"] if cored else None, beta=info["beta"] if cored else None,
         tc=info["tc"] if cored else None, t=info["t"],
         mass_g=info["mass_total_g"], eval_stroke_mm=stroke, reached_eval_stroke=bool(reached),
         time_at_eval_ms=float(t_[-1]), sim_end_ms=float(t[-1]),
         PCF_kN=PCF, MCF_kN=MCF, EA_J=float(EA[-1]), SEA_kJ_per_kg=float(EA[-1] / mass_kg / 1000.0),
         CFE=MCF / PCF, IE_parts_J=float(IE), HE_parts_J=float(HE),
         hourglass_ratio=float(HE / max(IE, 1e-9)),
         EA_vs_IE_mismatch=float((EA[-1] - IE) / max(IE, 1e-9)))
# occupant-relevant: deceleration of the impact mass (wall force / impactor mass), in g
m_imp = info["mass_kg"]
acc_g = F_ * 1000.0 / m_imp / 9.81
def clip_peak(ms):  # highest average over any window of `ms` milliseconds (like the 3 ms clip)
    cum = np.concatenate([[0], np.cumsum(0.5 * (acc_g[1:] + acc_g[:-1]) * np.diff(t_))])
    if t_[-1] - t_[0] < ms: return float("nan")
    k = np.searchsorted(t_, t_ + ms)
    ok = k < len(t_)
    return float(np.max((cum[k[ok]] - cum[ok]) / (t_[k[ok]] - t_[ok])))
m["impactor_peak_decel_g"] = float(acc_g.max())
m["impactor_peak_decel_3ms_g"] = clip_peak(3.0)
m["impactor_mean_decel_g"] = float(MCF * 1000.0 / m_imp / 9.81)
m["impactor_speed_at_eval_ms"] = float(-N[k_end, 5])
# engine energy-balance error (column ERROR in the listing) up to the evaluation time
import re
errs = []
outf = os.path.join(a.case_dir, f"{a.case}_0001.out")
if os.path.exists(outf):
    for line in open(outf, errors="ignore"):
        mm = re.match(r"^\s+\d+\s+([0-9.]+)\s+[0-9.E+-]+\s+\w+\s+\d+\s+(-?[0-9.]+)%", line)
        if mm and float(mm.group(1)) <= m["time_at_eval_ms"] + 1e-6:
            errs.append(float(mm.group(2)))
m["energy_error_max_abs_pct"] = max((abs(e) for e in errs), default=None)
m["valid"] = bool(m["energy_error_max_abs_pct"] is not None and m["energy_error_max_abs_pct"] < 5.0
                  and abs(m["EA_vs_IE_mismatch"]) < 0.05)
json.dump(m, open(os.path.join(a.case_dir, f"{a.case}_metrics.json"), "w"), indent=2)
np.savetxt(os.path.join(a.case_dir, f"{a.case}_force_disp.csv"), np.column_stack([t, d, F]),
           delimiter=",", header="time_ms,crush_mm,force_kN", comments="")
print(json.dumps(m, indent=2))

if a.plot:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=130)
    ax.plot(d, F, lw=0.9, color="#bbb")
    ax.plot(d_, F_, lw=1.0, color="#2a6fdb")
    ax.axhline(MCF, ls="--", lw=1, color="#d9534f", label=f"MCF {MCF:.1f} kN")
    ax.axvline(stroke, ls=":", lw=1, color="#555", label=f"eval stroke {stroke:.0f} mm")
    ax.set_xlabel("Crush (mm)"); ax.set_ylabel("Wall force (kN)")
    ax.set_title(f"{a.case}: PCF {PCF:.0f} kN, EA {EA[-1]/1000:.2f} kJ, SEA {m['SEA_kJ_per_kg']:.1f} kJ/kg", fontsize=9)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(a.case_dir, f"{a.case}_force_disp.png"))

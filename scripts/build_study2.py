#!/usr/bin/env python3
"""Package design study 2 (aim: higher SEA and lower PCF than the conventional box) from solver output only."""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
p = argparse.ArgumentParser()
p.add_argument("--table", required=True); p.add_argument("--surr", nargs=4, required=True, help="dirs: AA6061, AA7075, AA6061 equal mass, AA7075 equal mass")
p.add_argument("--runs", default="/home/user/runs"); p.add_argument("--out", required=True)
a = p.parse_args()
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = a.out; F = os.path.join(O, "figures"); os.makedirs(F, exist_ok=True)
df = pd.read_csv(a.table, dtype={"gh_run_id": str}); D = df.set_index("case"); b = D.loc["B000"]
cols = ["case", "mat", "core", "theta", "tc", "ncell", "beta", "t", "trig", "core_gap", "core_bevel", "mass_g", "EA_J", "SEA_kJ_per_kg",
        "PCF_CFC600_kN", "PCF_kN", "MCF_kN", "CFE_CFC600", "impactor_mean_decel_g", "energy_error_max_abs_pct", "valid", "engine_wall_s", "gh_run_id"]
def pct(T):
    T = T.copy()
    for k in ("mass_g", "EA_J", "SEA_kJ_per_kg", "PCF_CFC600_kN", "MCF_kN"): T[f"{k}_vs_B000_pct"] = (T[k] / b[k] - 1) * 100
    return T
# 1 design comparison
names = {"B000": "Conventional AA6061-T6 box, 2.0 mm (baseline)", "E_BT": "Baseline + 1.5 mm crush trigger (reference only)",
         "V_RE": "Study 1 re-entrant core, 2.0 mm wall", "Q6M_S": "RECOMMENDED AA6061-T6: 1.0 mm wall + re-entrant core, equal mass",
         "Q6M_K": "AA6061-T6 equal mass, knee", "Q6_K": "AA6061-T6 knee (lighter)", "Q7M_S": "AA7075-T6 equal mass, max SEA",
         "Q7_K": "AA7075-T6 knee (lighter)", "Q7M_K": "AA7075-T6 equal mass, knee"}
T1 = pct(D.loc[[c for c in names if c in D.index]].reset_index())[[*cols, "mass_g_vs_B000_pct", "EA_J_vs_B000_pct", "SEA_kJ_per_kg_vs_B000_pct", "PCF_CFC600_kN_vs_B000_pct", "MCF_kN_vs_B000_pct"]]
T1.insert(1, "design", [names[c] for c in T1.case]); T1.to_csv(os.path.join(O, "table1_design_comparison.csv"), index=False)
# 2 validation (Simpson & Kazanci 2020)
vrows = []
desc = {"SK_V2": "sharp corners, 2.0 mm", "SK_M15": "sharp corners, 1.5 mm", "SK_M125": "sharp corners, 1.25 mm",
        "SK_R3": "corner radius 2.25 mm, 3.0 mm mesh", "SK_R25": "corner radius, 2.5 mm", "SK_R": "corner radius, 2.0 mm", "SK_R15": "corner radius, 1.5 mm",
        "SK_R125": "corner radius, 1.25 mm", "SK_T25": "0.15 mm imperfection, 2.5 mm", "SK_T": "0.15 mm imperfection, 2.0 mm",
        "SK_T15": "0.15 mm imperfection, 1.5 mm", "SK_RT": "radius + imperfection, 2.0 mm", "SK_F1": "friction 0.1 (sensitivity), 2.0 mm"}
for c, dsc in desc.items():
    f = os.path.join(a.runs, c, f"{c}_validation.json")
    if not os.path.exists(f): continue
    v = json.load(open(f)); r = {x["metric"]: x for x in v["rows"]}
    vrows.append(dict(run=c, model=dsc, PCF_FE_kN=r["PCF_kN"]["FE"], PCF_test_kN=r["PCF_kN"]["test_mean"], PCF_err_pct=r["PCF_kN"]["error_pct"],
                      MCF_FE_kN=r["MCF_kN"]["FE"], MCF_test_kN=r["MCF_kN"]["test_mean"], MCF_err_pct=r["MCF_kN"]["error_pct"],
                      EA_FE_kJ=r["EA_kJ"]["FE"], EA_err_pct=r["EA_kJ"]["error_pct"], within_10pct=abs(r["PCF_kN"]["error_pct"]) <= 10 and abs(r["MCF_kN"]["error_pct"]) <= 10))
    for g in glob.glob(os.path.join(a.runs, c, f"{c}_validation.png")): shutil.copy(g, os.path.join(F, "validation_" + os.path.basename(g)))
T2 = pd.DataFrame(vrows); T2.to_csv(os.path.join(O, "table2_validation_simpson2020.csv"), index=False)
# 3-5 design data
X = df[df.case.str.match(r"^(E_|X\d|H0|M\d)")][cols]; X.to_csv(os.path.join(O, "table3_exploratory_21_runs.csv"), index=False)
G = df[df.case.str.match(r"^G\d+$")][cols]; G.to_csv(os.path.join(O, "table4_design_study_AA6061_80_runs.csv"), index=False)
K = df[df.case.str.match(r"^K\d+$")][cols]; K.to_csv(os.path.join(O, "table5_design_study_AA7075_50_runs.csv"), index=False)
# 6 surrogates, 7 optimum check
labels = ["AA6061-T6", "AA7075-T6", "AA6061-T6 equal mass", "AA7075-T6 equal mass"]; pre = ["Q6", "Q7", "Q6M", "Q7M"]
sc = []; chk = []
for d, lab, pr in zip(a.surr, labels, pre):
    if "equal" not in lab:
        s = pd.read_csv(os.path.join(d, "surrogate_scores.csv")); s.insert(0, "material", lab); sc.append(s)
        for g in glob.glob(os.path.join(d, "parity_*.png")): shutil.copy(g, os.path.join(F, f"parity_{lab.split()[0]}_" + os.path.basename(g)[7:]))
    shutil.copy(os.path.join(d, "pareto.png"), os.path.join(F, f"pareto_{lab.replace(' ', '_')}.png"))
    o = json.load(open(os.path.join(d, "optimum.json")))
    for k, v in o["picks"].items():
        c = f"{pr}_{k}"
        if c not in D.index: continue
        r = D.loc[c]
        chk.append(dict(case=c, study=lab, theta=v["theta"], tc=v["tc"], t=v["t"], trig=v["trig"], core_gap=v["core_gap"], core_bevel=v["core_bevel"],
                        surrogate=f"GP/{o['best_models']}", mass_g=r.mass_g, SEA_pred=v["SEA_pred"], SEA_FE=r.SEA_kJ_per_kg, SEA_err_pct=(v["SEA_pred"] / r.SEA_kJ_per_kg - 1) * 100,
                        PCF_pred=v["PCF_pred"], PCF_FE=r.PCF_CFC600_kN, PCF_err_pct=(v["PCF_pred"] / r.PCF_CFC600_kN - 1) * 100,
                        EA_J=r.EA_J, EA_vs_B000_pct=(r.EA_J / b.EA_J - 1) * 100, SEA_vs_B000_pct=(r.SEA_kJ_per_kg / b.SEA_kJ_per_kg - 1) * 100,
                        PCF_vs_B000_pct=(r.PCF_CFC600_kN / b.PCF_CFC600_kN - 1) * 100, CFE=r.CFE_CFC600, valid=r.valid))
pd.concat(sc).to_csv(os.path.join(O, "table6_surrogate_accuracy.csv"), index=False)
T7 = pd.DataFrame(chk); T7.to_csv(os.path.join(O, "table7_optimum_prediction_vs_FE.csv"), index=False)
with pd.ExcelWriter(os.path.join(O, "study2_results.xlsx")) as xw:
    for n, T in (("1 Design comparison", T1), ("2 Validation", T2), ("3 Exploratory", X), ("4 DOE AA6061", G), ("5 DOE AA7075", K),
                 ("6 Surrogates", pd.concat(sc)), ("7 Optimum check", T7)): T.to_excel(xw, sheet_name=n, index=False)
# figures
def curve(c):
    d = tempfile.mkdtemp()
    for f in (f"{c}T01", f"{c}_info.json", f"{c}_0001.out"):
        r = subprocess.run(["git", "-C", repo, "show", f"origin/results:campaign/{c}/{f}"], capture_output=True)
        if r.returncode: return None
        open(os.path.join(d, f), "wb").write(r.stdout)
    subprocess.run([sys.executable, os.path.join(repo, "scripts/post_crashbox.py"), d, "--case", c, "--plot", "0"], capture_output=True)
    return pd.read_csv(os.path.join(d, f"{c}_force_disp.csv"))
fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=150)
for c, col in (("B000", "#888"), ("Q6M_S", "#2a6fdb"), ("Q6M_K", "#7fb0ff"), ("Q7M_S", "#e08a1e"), ("Q7_K", "#2ca02c")):
    cv = curve(c)
    if cv is None: continue
    cv = cv[cv.crush_mm <= D.loc[c, "eval_stroke_mm"] + 0.01]
    ax.plot(cv.crush_mm, cv.force_CFC600_kN, lw=1.2, color=col, label=f"{c}: {names[c]}")
ax.axhline(b.PCF_CFC600_kN, ls=":", color="#888", lw=0.8)
ax.set_xlabel("Crush (mm)"); ax.set_ylabel("Wall force, CFC600 (kN)"); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=6.5)
ax.set_title("Force-crush, 55 km/h, 557 kg per box (OpenRadioss)", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(F, "fig1_force_crush_recommended.png")); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.5, 4.3), dpi=150)
for T, col, lab in ((G, "#2a6fdb", "AA6061-T6 design study (80)"), (K, "#e08a1e", "AA7075-T6 design study (50)")):
    ax.scatter(T.PCF_CFC600_kN, T.SEA_kJ_per_kg, s=14, c=col, alpha=.6, label=lab)
for c, mk in (("B000", "s"), ("Q6M_S", "*"), ("Q7M_S", "*"), ("Q7_K", "*")):
    ax.scatter(D.loc[c, "PCF_CFC600_kN"], D.loc[c, "SEA_kJ_per_kg"], marker=mk, s=110, c="#d9534f" if c != "B000" else "k", zorder=5)
    ax.annotate(c, (D.loc[c, "PCF_CFC600_kN"], D.loc[c, "SEA_kJ_per_kg"]), fontsize=7, xytext=(5, 3), textcoords="offset points")
ax.axvline(b.PCF_CFC600_kN, ls="--", c="k", lw=.8); ax.axhline(b.SEA_kJ_per_kg, ls="--", c="k", lw=.8)
ax.text(b.PCF_CFC600_kN * 0.98, ax.get_ylim()[1] * 0.97, "aim: up-left of the baseline", ha="right", va="top", fontsize=7)
ax.set_xlabel("PCF, CFC600 (kN)"); ax.set_ylabel("SEA (kJ/kg)"); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig2_SEA_vs_PCF_all_runs.png")); plt.close(fig)
print(T1[["case", "mass_g", "EA_J", "SEA_kJ_per_kg", "PCF_CFC600_kN", "CFE_CFC600", "EA_J_vs_B000_pct", "PCF_CFC600_kN_vs_B000_pct"]].round(1).to_string(index=False))
print(T2.round(1).to_string(index=False))

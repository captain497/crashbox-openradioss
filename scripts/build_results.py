#!/usr/bin/env python3
"""Assemble the results package (tables, plots, animations) from solver output only.

Inputs: the collect.py table, the surrogate output folder, the results branch, and the
local validation run folders. Output: one folder with CSV tables, an Excel workbook and figures.
"""
import argparse, glob, json, os, shutil, subprocess, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--table", required=True); p.add_argument("--surr", required=True)
p.add_argument("--val", nargs="*", default=[], help="local validation run dirs (with <case>_validation.json)")
p.add_argument("--opt_local", nargs="*", default=[], help="local optimum-check run dirs")
p.add_argument("--out", required=True)
a = p.parse_args()
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(a.out, exist_ok=True)
fig_dir = os.path.join(a.out, "figures"); os.makedirs(fig_dir, exist_ok=True)

def show(path):
    r = subprocess.run(["git", "-C", repo, "show", f"origin/results:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None

df = pd.read_csv(a.table, dtype={"gh_run_id": str})
# local optimum checks override / add rows
for d in a.opt_local:
    for f in glob.glob(os.path.join(d, "*_metrics.json")):
        m = json.load(open(f)); m["gh_run_id"] = "local"
        rt = os.path.join(d, "runtime.txt")
        if os.path.exists(rt): m["engine_wall_s"] = open(rt).read().strip().split("=")[-1]
        df = df[df.case != m["case"]]; df = pd.concat([df, pd.DataFrame([m])], ignore_index=True)

cols = ["case", "core", "theta", "ncell", "beta", "tc", "t", "mass_g", "EA_J", "SEA_kJ_per_kg", "PCF_kN", "PCF_CFC600_kN",
        "MCF_kN", "CFE", "CFE_CFC600", "impactor_peak_decel_CFC600_g", "impactor_peak_decel_3ms_g", "impactor_mean_decel_g",
        "eval_stroke_mm", "energy_error_max_abs_pct", "hourglass_ratio", "valid", "engine_wall_s", "gh_run_id"]
cols = [c for c in cols if c in df.columns]
names = {"B000": "Baseline empty tube, t = 2.0 mm", "V_BEQ": "Empty tube, equal mass (t = 3.23 mm)",
         "V_RE": "Re-entrant core (theta 20, n 3, beta 2, tc 0.8)", "V_HT": "Horsetail double wall + re-entrant (tc 0.63)"}

# ---------- Table 1: design comparison ----------
cmp_cases = ["B000", "V_BEQ", "V_RE", "V_HT"] + sorted(c for c in df.case if str(c).startswith("O58_"))
T1 = df[df.case.isin(cmp_cases)].set_index("case").loc[[c for c in cmp_cases if c in set(df.case)]].reset_index()
T1.insert(1, "design", [names.get(c, "NSGA-II optimum check" if c.startswith("O58") else c) for c in T1.case])
base = T1[T1.case == "B000"].iloc[0]
for k in ("EA_J", "SEA_kJ_per_kg", "PCF_CFC600_kN", "impactor_mean_decel_g"):
    T1[f"{k}_vs_baseline_pct"] = (T1[k] / base[k] - 1) * 100
T1[["design"] + cols].to_csv(os.path.join(a.out, "table1_design_comparison.csv"), index=False)

# ---------- Table 2: mesh convergence ----------
mesh = [("M_B33", 3.333, "empty"), ("B000", 2.5, "empty"), ("M_B17", 1.667, "empty"), ("M_B125", 1.25, "empty"),
        ("V_RE", 2.5, "re-entrant"), ("M_RE20", 2.0, "re-entrant")]
T2 = pd.DataFrame([dict(case=c, model=mdl, element_mm=h, **df[df.case == c][["SEA_kJ_per_kg", "PCF_CFC600_kN", "MCF_kN",
                   "energy_error_max_abs_pct", "engine_wall_s"]].iloc[0].to_dict()) for c, h, mdl in mesh if c in set(df.case)])
T2.to_csv(os.path.join(a.out, "table2_mesh_convergence.csv"), index=False)

# ---------- Table 3: validation ----------
rows = []
for d in a.val:
    for f in glob.glob(os.path.join(d, "*_validation.json")):
        v = json.load(open(f))
        for r in v["rows"]:
            rows.append(dict(run=v["case"], **r))
T3 = pd.DataFrame(rows)
if len(T3): T3.to_csv(os.path.join(a.out, "table3_validation_simpson2020.csv"), index=False)

# ---------- Table 4: design study ----------
T4 = df[df.case.str.match(r"^D\d+$")].sort_values("case")[cols]
T4.to_csv(os.path.join(a.out, "table4_design_study_60_runs.csv"), index=False)

# ---------- Table 5/6: surrogates and optimum ----------
T5 = pd.read_csv(os.path.join(a.surr, "surrogate_scores.csv"))
opt = json.load(open(os.path.join(a.surr, "optimum.json")))
r6 = []
for n, r in opt["picks"].items():
    c = "O58_" + n.split("_")[1]
    fe = df[df.case == c]
    row = dict(pick=n, case=c, theta=r["theta"], tc=r["tc"], ncell=int(r["ncell"]), beta=r["beta"],
               SEA_pred=r["SEA_pred"], PCF_pred=r["PCF_pred"])
    if len(fe):
        row.update(SEA_FE=fe.SEA_kJ_per_kg.iloc[0], PCF_FE=fe.PCF_CFC600_kN.iloc[0])
        row["SEA_err_pct"] = (row["SEA_pred"] / row["SEA_FE"] - 1) * 100
        row["PCF_err_pct"] = (row["PCF_pred"] / row["PCF_FE"] - 1) * 100
    r6.append(row)
T6 = pd.DataFrame(r6); T6.to_csv(os.path.join(a.out, "table6_optimum_prediction_vs_FE.csv"), index=False)
T5.to_csv(os.path.join(a.out, "table5_surrogate_accuracy.csv"), index=False)

try:
    with pd.ExcelWriter(os.path.join(a.out, "crashbox_results.xlsx")) as xw:
        T1[["design"] + cols].to_excel(xw, sheet_name="1 Design comparison", index=False)
        T2.to_excel(xw, sheet_name="2 Mesh convergence", index=False)
        if len(T3): T3.to_excel(xw, sheet_name="3 Validation", index=False)
        T4.to_excel(xw, sheet_name="4 Design study", index=False)
        T5.to_excel(xw, sheet_name="5 Surrogate accuracy", index=False)
        T6.to_excel(xw, sheet_name="6 Optimum check", index=False)
except Exception as e:
    print("xlsx skipped:", e)

# ---------- Figures ----------
def curve(case):
    for d in a.opt_local:
        f = os.path.join(d, f"{case}_force_disp.csv")
        if os.path.exists(f): return pd.read_csv(f)
    import tempfile  # regenerate from the raw TH file with the current post-processor
    d = tempfile.mkdtemp()
    for f in (f"{case}T01", f"{case}_info.json", f"{case}_0001.out"):
        b = show(f"campaign/{case}/{f}")
        if b is None: return None
        open(os.path.join(d, f), "wb").write(b)
    subprocess.run([sys.executable, os.path.join(repo, "scripts/post_crashbox.py"), d, "--case", case, "--plot", "0"], capture_output=True)
    return pd.read_csv(os.path.join(d, f"{case}_force_disp.csv"))

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
cl = {"B000": "#888", "V_BEQ": "#444", "V_RE": "#2a6fdb", "V_HT": "#e08a1e", "O58_K": "#2ca02c"}
for c in [c for c in cmp_cases if c in cl]:
    cv = curve(c)
    if cv is None: continue
    s = df[df.case == c].eval_stroke_mm.iloc[0]
    cv = cv[cv.crush_mm <= s + 0.01]
    ax.plot(cv.crush_mm, cv.force_CFC600_kN, lw=1.2, color=cl[c], label=f"{c}: {names.get(c, 'NSGA-II optimum')}")
ax.set_xlabel("Crush (mm)"); ax.set_ylabel("Wall force, CFC600 (kN)"); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=7)
ax.set_title("Force-crush curves at 55 km/h, 557 kg per crash box (OpenRadioss)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig1_force_crush_comparison.png")); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150)
lab = T1.case.tolist(); x = np.arange(len(lab))
for ax, k, yl in zip(axs, ("SEA_kJ_per_kg", "PCF_CFC600_kN", "impactor_mean_decel_g"),
                     ("SEA (kJ/kg)", "PCF, CFC600 (kN)", "Mean deceleration of 557 kg (g)")):
    ax.bar(x, T1[k], color=[cl.get(c, "#2ca02c") for c in lab])
    for i, v in enumerate(T1[k]): ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=45, fontsize=7); ax.set_ylabel(yl); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig2_metrics_comparison.png")); plt.close(fig)

D = T4[T4.valid.astype(str) == "True"]
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
sc = ax.scatter(D.PCF_CFC600_kN, D.SEA_kJ_per_kg, c=D.tc, cmap="viridis", s=22)
plt.colorbar(sc, label="core rib thickness tc (mm)")
for c in ("B000", "V_BEQ", "O58_K"):
    r = df[df.case == c]
    if len(r): ax.scatter(r.PCF_CFC600_kN, r.SEA_kJ_per_kg, marker="*", s=120, c="#d9534f"); ax.annotate(c, (r.PCF_CFC600_kN.iloc[0], r.SEA_kJ_per_kg.iloc[0]), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("PCF, CFC600 (kN)"); ax.set_ylabel("SEA (kJ/kg)"); ax.grid(alpha=.3)
ax.set_title(f"Design study: {len(D)} valid FE runs", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig3_design_study_SEA_vs_PCF.png")); plt.close(fig)

for f in glob.glob(os.path.join(a.surr, "*.png")):
    shutil.copy(f, os.path.join(fig_dir, "fig4_" + os.path.basename(f)))
for d in a.val:
    for f in glob.glob(os.path.join(d, "*_validation.png")):
        shutil.copy(f, os.path.join(fig_dir, "fig5_" + os.path.basename(f)))

anim_dir = os.path.join(a.out, "animations"); os.makedirs(anim_dir, exist_ok=True)
for c in ("A_B000", "A_V_RE", "A_V_HT", "O58_K", "O58_S", "O58_P"):
    for suf in ("_crush.gif", "_final_shape.png"):
        b = show(f"campaign/{c}/{c}{suf}")
        if b: open(os.path.join(anim_dir, c.replace("A_", "") + suf), "wb").write(b)
    for d in a.opt_local:
        f = os.path.join(d, f"{c}{suf}")
        if os.path.exists(f): shutil.copy(f, os.path.join(anim_dir, c + suf))
print("tables:", sorted(os.listdir(a.out)))
print(T1[["case", "mass_g", "EA_J", "SEA_kJ_per_kg", "PCF_CFC600_kN", "MCF_kN", "CFE_CFC600", "impactor_mean_decel_g", "energy_error_max_abs_pct"]].round(2).to_string(index=False))
print(T6.round(2).to_string(index=False))

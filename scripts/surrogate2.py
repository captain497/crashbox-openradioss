#!/usr/bin/env python3
"""Surrogates (GP, RF, NN) and constrained NSGA-II for design study 2 (continuous design variables).

Targets: SEA (kJ/kg, maximise) and PCF (kN, CFC600, minimise), optional constraint PCF <= --pcf_max
(the conventional 2 mm AA6061-T6 box). Writes scores, parity plots, pareto.csv/png, optimum.json and a DOE csv
of picks to re-check in FE.
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error

p = argparse.ArgumentParser()
p.add_argument("table"); p.add_argument("--out", required=True)
p.add_argument("--case_re", required=True); p.add_argument("--runs", default="")
p.add_argument("--xcols", default="theta,tc,t,trig,core_gap,core_bevel")
p.add_argument("--pcf", default="PCF_CFC600_kN"); p.add_argument("--pcf_max", type=float, default=0.0)
p.add_argument("--fixed", default="core=reentrant,ncell=3,beta=2.0,mat=aa6061")
p.add_argument("--prefix", default="Q"); p.add_argument("--doe_out", required=True)
p.add_argument("--mass_lo", type=float, default=0.0); p.add_argument("--mass_hi", type=float, default=0.0)  # optional mass window (g)
p.add_argument("--seed", type=int, default=2026)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.table, dtype={"gh_run_id": str})
df = df[df["case"].str.match(a.case_re) & (df["valid"].astype(str) == "True")].copy()
if a.runs: df = df[df["gh_run_id"].astype(str).isin(a.runs.split(","))]
XC = a.xcols.split(","); X = df[XC].astype(float).values
TARGETS = {"SEA_kJ_per_kg": "SEA (kJ/kg)", a.pcf: "PCF (kN)"}
print(f"{len(df)} valid runs used")

def models(seed):
    k = Ck(1.0) * Matern(length_scale=np.ones(len(XC)), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(1e-2, (1e-6, 1.0))
    gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=8, random_state=seed))
    rf = RandomForestRegressor(n_estimators=500, random_state=seed)
    nn = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16, 16), activation="tanh",
                                    solver="lbfgs", alpha=1e-2, max_iter=20000, random_state=seed)), transformer=StandardScaler())
    return {"GP": gp, "RF": rf, "NN": nn}

rows, best, fitted = [], {}, {}
for tgt, label in TARGETS.items():
    y = df[tgt].astype(float).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=a.seed)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), dpi=130)
    for ax, (name, mdl) in zip(axs, models(a.seed).items()):
        mdl.fit(Xtr, ytr); pte = mdl.predict(Xte)
        r2cv = [r2_score(y, cross_val_predict(models(a.seed + r)[name], X, y, cv=KFold(5, shuffle=True, random_state=a.seed + r))) for r in range(5)]
        rows.append(dict(target=tgt, model=name, n_train=len(ytr), n_test=len(yte), R2_test=r2_score(yte, pte),
                         MAPE_test_pct=mean_absolute_percentage_error(yte, pte) * 100, R2_cv_mean=np.mean(r2cv), R2_cv_std=np.std(r2cv)))
        ax.scatter(ytr, mdl.predict(Xtr), s=12, c="#9bbcf0", label="train"); ax.scatter(yte, pte, s=18, c="#d9534f", label="test")
        lo, hi = min(y.min(), pte.min()), max(y.max(), pte.max()); ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{name}: test R² {rows[-1]['R2_test']:.3f}, CV R² {np.mean(r2cv):.3f}", fontsize=8)
        ax.set_xlabel(f"FE {label}"); ax.set_ylabel(f"Predicted {label}"); ax.grid(alpha=.3)
    axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(a.out, f"parity_{tgt}.png")); plt.close(fig)
    sc = [r for r in rows if r["target"] == tgt]
    best[tgt] = max(sc, key=lambda r: r["R2_test"] + r["R2_cv_mean"])["model"]
    m = models(a.seed)[best[tgt]]; m.fit(X, y); fitted[tgt] = m
scores = pd.DataFrame(rows); scores.to_csv(os.path.join(a.out, "surrogate_scores.csv"), index=False)
print(scores.round(3).to_string(index=False)); print("best:", best)

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
xl = df[XC].astype(float).min().values; xu = df[XC].astype(float).max().values   # stay inside the sampled box
sea_m, pcf_m = fitted["SEA_kJ_per_kg"], fitted[a.pcf]
mass_m = None
if a.mass_hi > 0:   # mass is a deterministic function of geometry; a GP reproduces it closely
    mass_m = models(a.seed)["GP"].fit(X, df["mass_g"].astype(float).values)
    print("mass GP R2 (train):", r2_score(df["mass_g"], mass_m.predict(X)))
ncon = (1 if a.pcf_max > 0 else 0) + (2 if mass_m is not None else 0)
class P(Problem):
    def __init__(s): super().__init__(n_var=len(XC), n_obj=2, n_ieq_constr=ncon, xl=xl, xu=xu)
    def _evaluate(s, x, out, *args, **kw):
        sp, pp = sea_m.predict(x), pcf_m.predict(x)
        out["F"] = np.column_stack([-sp, pp])
        G = []
        if a.pcf_max > 0: G.append(pp - a.pcf_max)
        if mass_m is not None:
            mm = mass_m.predict(x); G += [a.mass_lo - mm, mm - a.mass_hi]
        if G: out["G"] = np.column_stack(G)
res = minimize(P(), NSGA2(pop_size=150), ("n_gen", 250), seed=a.seed, verbose=False)
par = pd.DataFrame(res.X, columns=XC); par["SEA_pred"] = -res.F[:, 0]; par["PCF_pred"] = res.F[:, 1]
par = par.sort_values("PCF_pred").reset_index(drop=True); par.to_csv(os.path.join(a.out, "pareto.csv"), index=False)
s_n = (par.SEA_pred.max() - par.SEA_pred) / (np.ptp(par.SEA_pred) + 1e-12); p_n = (par.PCF_pred - par.PCF_pred.min()) / (np.ptp(par.PCF_pred) + 1e-12)
picks = {"K": par.iloc[int(np.argmin(np.hypot(s_n, p_n)))], "S": par.iloc[int(par.SEA_pred.idxmax())], "P": par.iloc[int(par.PCF_pred.idxmin())]}
opt = {n: {c: float(r[c]) for c in par.columns} for n, r in picks.items()}
json.dump(dict(best_models=best, picks=opt, pcf_max=a.pcf_max, n_runs=len(df), xcols=XC), open(os.path.join(a.out, "optimum.json"), "w"), indent=2)
fx = dict(kv.split("=") for kv in a.fixed.split(","))
with open(a.doe_out, "w") as fh:
    fh.write("case," + ",".join(fx) + "," + ",".join(XC) + "\n")
    for n, r in opt.items():
        fh.write(f"{a.prefix}_{n}," + ",".join(fx.values()) + "," + ",".join(f"{r[c]:.4f}" for c in XC) + "\n")
print(json.dumps(opt, indent=1))
fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
ax.scatter(df[a.pcf], df["SEA_kJ_per_kg"], s=14, c="#bbb", label="FE design study")
ax.plot(par.PCF_pred, par.SEA_pred, "k.-", lw=1, ms=3, label="NSGA-II Pareto front (surrogate)")
if a.pcf_max > 0: ax.axvline(a.pcf_max, color="#d9534f", ls="--", lw=1, label="PCF limit (baseline box)")
for n, r in opt.items(): ax.annotate(n, (r["PCF_pred"], r["SEA_pred"]), fontsize=7, xytext=(4, -10), textcoords="offset points")
ax.set_xlabel("PCF, CFC600 (kN)"); ax.set_ylabel("SEA (kJ/kg)"); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "pareto.png"))

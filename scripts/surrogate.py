#!/usr/bin/env python3
"""Surrogate models (GP, RF, NN) on the FE design study and NSGA-II optimisation.

Input: the table from collect.py (only valid D-cases are used).
Design variables: theta (deg), tc (mm), ncell (-), beta (-).
Targets: SEA (kJ/kg, maximise) and PCF (kN, CFC600-filtered, minimise).

Outputs (in --out): surrogate_scores.csv, parity plots, pareto.csv/png, optimum.json,
and doe/opt.csv with the designs to re-check in FE.
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
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error

p = argparse.ArgumentParser()
p.add_argument("table"); p.add_argument("--out", default="surrogate_out")
p.add_argument("--pcf", default="PCF_CFC600_kN")
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--runs", default="", help="comma-separated GitHub run ids to accept (empty = all)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

df = pd.read_csv(a.table)
df = df[df["case"].str.match(r"^D\d+$") & (df["valid"].astype(str) == "True")].copy()
if a.runs:
    df = df[df["gh_run_id"].astype(str).isin(a.runs.split(","))]
X_COLS = ["theta", "tc", "ncell", "beta"]
TARGETS = {"SEA_kJ_per_kg": "SEA (kJ/kg)", a.pcf: "PCF (kN)"}
X = df[X_COLS].astype(float).values
print(f"{len(df)} valid design-study runs used")

def models(seed):
    k = Ck(1.0) * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(1e-2, (1e-6, 1.0))
    gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=8, random_state=seed))
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=seed)
    nn = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16, 16), activation="tanh", solver="lbfgs",
                                                               alpha=1e-2, max_iter=20000, random_state=seed)),
        transformer=StandardScaler())
    return {"GP": gp, "RF": rf, "NN": nn}

rows, best, fitted = [], {}, {}
for tgt, label in TARGETS.items():
    y = df[tgt].astype(float).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=a.seed)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), dpi=130)
    for ax, (name, mdl) in zip(axs, models(a.seed).items()):
        mdl.fit(Xtr, ytr); pte = mdl.predict(Xte)
        r2_test = r2_score(yte, pte); mape = mean_absolute_percentage_error(yte, pte) * 100
        # repeated 5-fold CV (out-of-fold R2), 5 repeats
        r2cv = []
        for rep in range(5):
            oof = cross_val_predict(models(a.seed + rep)[name], X, y, cv=KFold(5, shuffle=True, random_state=a.seed + rep))
            r2cv.append(r2_score(y, oof))
        rows.append(dict(target=tgt, model=name, n_train=len(ytr), n_test=len(yte), R2_test=r2_test,
                         MAPE_test_pct=mape, R2_cv_mean=np.mean(r2cv), R2_cv_std=np.std(r2cv)))
        ax.scatter(ytr, mdl.predict(Xtr), s=12, c="#9bbcf0", label="train")
        ax.scatter(yte, pte, s=18, c="#d9534f", label="test")
        lo, hi = min(y.min(), pte.min()), max(y.max(), pte.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{name}: test R² {r2_test:.3f}, CV R² {np.mean(r2cv):.3f}", fontsize=8)
        ax.set_xlabel(f"FE {label}"); ax.set_ylabel(f"Predicted {label}"); ax.grid(alpha=.3)
    axs[0].legend(frameon=False, fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"parity_{tgt}.png")); plt.close(fig)
    sc = [r for r in rows if r["target"] == tgt]
    b = max(sc, key=lambda r: r["R2_cv_mean"])["model"]
    best[tgt] = b
    m = models(a.seed)[b]; m.fit(X, y); fitted[tgt] = m
scores = pd.DataFrame(rows); scores.to_csv(os.path.join(a.out, "surrogate_scores.csv"), index=False)
print(scores.round(3).to_string(index=False)); print("best:", best)

# ---------------- NSGA-II on the best surrogates ----------------
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
lo = df[["theta", "tc", "beta"]].min().values; hi = df[["theta", "tc", "beta"]].max().values
bounds = dict(theta=(10, 30), tc=(0.4, 1.2), beta=(1.5, 3.0))
xl = np.array([bounds[k][0] for k in ("theta", "tc", "beta")]); xu = np.array([bounds[k][1] for k in ("theta", "tc", "beta")])
sea_m, pcf_m = fitted["SEA_kJ_per_kg"], fitted[a.pcf]
fronts = []
for nc in sorted(df["ncell"].astype(int).unique()):
    class P(Problem):
        def __init__(s): super().__init__(n_var=3, n_obj=2, xl=xl, xu=xu)
        def _evaluate(s, x, out, *args, **kw):
            Z = np.column_stack([x[:, 0], x[:, 1], np.full(len(x), nc), x[:, 2]])
            out["F"] = np.column_stack([-sea_m.predict(Z), pcf_m.predict(Z)])
    res = minimize(P(), NSGA2(pop_size=100), ("n_gen", 200), seed=a.seed, verbose=False)
    for xx, ff in zip(res.X, res.F):
        fronts.append(dict(theta=xx[0], tc=xx[1], ncell=nc, beta=xx[2], SEA_pred=-ff[0], PCF_pred=ff[1]))
fr = pd.DataFrame(fronts)
# global non-dominated set over the three ncell values
F = np.column_stack([-fr.SEA_pred, fr.PCF_pred])
nd = [i for i in range(len(F)) if not any((F[j] <= F[i]).all() and (F[j] < F[i]).any() for j in range(len(F)))]
par = fr.iloc[nd].sort_values("PCF_pred").reset_index(drop=True)
par.to_csv(os.path.join(a.out, "pareto.csv"), index=False)
# compromise (knee): closest to the ideal point in normalised objective space
s_n = (par.SEA_pred.max() - par.SEA_pred) / (par.SEA_pred.max() - par.SEA_pred.min() + 1e-12)
p_n = (par.PCF_pred - par.PCF_pred.min()) / (par.PCF_pred.max() - par.PCF_pred.min() + 1e-12)
k = int(np.argmin(np.hypot(s_n, p_n)))
picks = {"OPT_K": par.iloc[k], "OPT_S": par.iloc[int(par.SEA_pred.idxmax())], "OPT_P": par.iloc[int(par.PCF_pred.idxmin())]}
opt = {n: {c: float(r[c]) for c in par.columns} for n, r in picks.items()}
json.dump(dict(best_models=best, picks=opt), open(os.path.join(a.out, "optimum.json"), "w"), indent=2)
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "doe", "opt.csv"), "w") as fh:
    fh.write("case,core,theta,tc,ncell,beta\n")
    for n, r in opt.items():
        fh.write(f"{n},reentrant,{r['theta']:.3f},{r['tc']:.4f},{int(r['ncell'])},{r['beta']:.4f}\n")
print(json.dumps(opt, indent=1))

fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
ax.scatter(df[a.pcf], df["SEA_kJ_per_kg"], s=14, c="#bbb", label="FE design study")
for nc, mk in zip((2, 3, 4), "o^s"):
    q = fr[fr.ncell == nc]; ax.scatter(q.PCF_pred, q.SEA_pred, s=6, marker=mk, alpha=.35, label=f"NSGA-II front, n={nc}")
ax.plot(par.PCF_pred, par.SEA_pred, "k-", lw=1, label="global Pareto front (surrogate)")
for n, r in opt.items(): ax.annotate(n, (r["PCF_pred"], r["SEA_pred"]), fontsize=7, xytext=(4, -10), textcoords="offset points")
ax.set_xlabel("PCF, CFC600 (kN)"); ax.set_ylabel("SEA (kJ/kg)"); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "pareto.png"))

#!/usr/bin/env python3
"""Collect the alternative-material runs (campaign/AM_*) and the AA6061-T6 references (B000, Q6M_S)
from the results branch, write the comparison table and figures.
Usage: build_altmat.py <out_dir>"""
import csv, io, json, os, subprocess, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1]; os.makedirs(os.path.join(out, "figures"), exist_ok=True)
ref = "origin/results"
REFDIR = os.environ.get("REFDIR", "")  # re-post-processed B000 / Q6M_S (older metrics files lack CFC600 fields)
def show(p):
    loc = os.path.join(REFDIR, *p.split("/")[1:]) if REFDIR else ""
    if loc and p.split("/")[1] in ("B000", "Q6M_S") and os.path.exists(loc) and not p.endswith("RUN_INFO.txt"): return open(loc).read()
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{p}"], capture_output=True, text=True).stdout
cases = [c for c in show("campaign/").split() if c.startswith("AM_")]
cases = ["B000", "Q6M_S"] + sorted(c.rstrip("/") for c in cases)

MAT = {"aa6061": "AA6061-T6", "az31b": "Mg AZ31B", "ti64": "Ti-6Al-4V", "steel_sk": "Mild steel"}
COL = {"aa6061": "#257de0", "az31b": "#f47230", "ti64": "#00b894", "steel_sk": "#f2a600"}
SIZING = {"T": "same thickness", "M": "equal mass", "S": "equal strength (EA)"}
rows = []
for c in cases:
    m = json.loads(show(f"campaign/{c}/{c}_metrics.json")); m.setdefault("mat", "aa6061"); m.setdefault("core", "none")
    run = show(f"campaign/{c}/RUN_INFO.txt").split()
    geom = "auxetic (Q6M_S)" if m["core"] == "reentrant" else "conventional tube"
    if c in ("B000", "Q6M_S"):
        sizing, fail = "reference", "none"
    else:
        tag = c.split("_")[3]
        sizing, fail = SIZING[tag[0]], ("Johnson-Cook fracture" if tag.endswith("F") else "none")
    refc = "Q6M_S" if m["core"] == "reentrant" else "B000"
    rows.append(dict(case=c, geometry=geom, material=MAT[m["mat"]], mat=m["mat"], sizing=sizing, fracture_model=fail,
                     t_mm=m["t"], tc_mm=m.get("tc") if m["core"] == "reentrant" else "",
                     mass_g=m["mass_g"], EA_J=m["EA_J"], SEA_kJ_per_kg=m["SEA_kJ_per_kg"], PCF_CFC600_kN=m["PCF_CFC600_kN"],
                     PCF_raw_kN=m["PCF_kN"], MCF_kN=m["MCF_kN"], CFE_CFC600=m["CFE_CFC600"],
                     eval_stroke_mm=m["eval_stroke_mm"], reached_140mm=m["reached_eval_stroke"],
                     energy_error_max_abs_pct=m["energy_error_max_abs_pct"], EA_vs_IE_mismatch=m["EA_vs_IE_mismatch"],
                     passes_validity_check=m["valid"], reference=refc, gh_run_id=run[1] if len(run) > 1 else ""))
byc = {r["case"]: r for r in rows}
for r in rows:
    b = byc[r["reference"]]
    for k in ("mass_g", "EA_J", "SEA_kJ_per_kg", "PCF_CFC600_kN", "MCF_kN"):
        r[f"{k}_vs_ref_pct"] = 100 * (r[k] / b[k] - 1)
keys = list(rows[0].keys())
with open(os.path.join(out, "table_B_FE_material_comparison.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
    for r in rows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#999",
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})

# Fig A1: SEA vs PCF, one panel per geometry
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
MK = {"reference": "*", "same thickness": "s", "equal mass": "o", "equal strength (EA)": "D"}
for ax, geom in zip(axs, ("conventional tube", "auxetic (Q6M_S)")):
    for r in rows:
        if r["geometry"] != geom: continue
        hollow = r["fracture_model"] != "none"
        ax.scatter(r["PCF_CFC600_kN"], r["SEA_kJ_per_kg"], s=110 if r["sizing"] == "reference" else 55, marker=MK[r["sizing"]],
                   facecolor="white" if hollow else COL[r["mat"]], edgecolor=COL[r["mat"]], linewidth=2, zorder=3)
        lab = r["case"].replace("AM_", "")
        ax.annotate(lab, (r["PCF_CFC600_kN"], r["SEA_kJ_per_kg"]), xytext=(5, 4), textcoords="offset points", fontsize=7, color="#333")
    b = byc["B000"]
    ax.axvline(b["PCF_CFC600_kN"], color="#999", lw=1, ls="--"); ax.text(b["PCF_CFC600_kN"] + 4, 2, "PCF limit\n(B000)", fontsize=7, color="#666")
    ax.set_title(geom.capitalize(), fontsize=10, loc="left"); ax.set_xlabel("PCF, CFC600 (kN)")
axs[0].set_ylabel("SEA to 140 mm (kJ/kg)")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=COL[k], label=v) for k, v in MAT.items()]
h += [Line2D([], [], marker=MK[k], ls="", color="#555", mfc="#555" if k else "w", label=k) for k in MK]
h += [Line2D([], [], marker="o", ls="", color="#555", mfc="white", label="hollow = with fracture")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Alternative materials: SEA vs PCF (OpenRadioss, 557 kg at 55 km/h per box)", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0.1, 1, 0.97)); fig.savefig(os.path.join(out, "figures", "figA1_SEA_vs_PCF.png"), dpi=200); plt.close(fig)

# Fig A2: force-crush curves, auxetic geometry, references + equal mass + lighter candidates
def curve(c):
    d = np.genfromtxt(io.StringIO(show(f"campaign/{c}/{c}_force_disp.csv")), delimiter=",", names=True)
    return d["crush_mm"], d["force_CFC600_kN"]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
sets = [("Equal mass to AA6061 auxetic box", ["Q6M_S", "AM_Q_MG_M", "AM_Q_MG_MF", "AM_Q_TI_M", "AM_Q_TI_MF", "AM_Q_ST_M"]),
        ("Lighter designs at similar EA", ["Q6M_S", "AM_Q_MG_T", "AM_Q_MG_TF", "AM_Q_TI_S", "AM_Q_TI_SF"])]
for ax, (title, cs) in zip(axs, sets):
    for c in cs:
        if c not in byc: continue
        x, f = curve(c); r = byc[c]
        ax.plot(x, f, color=COL[r["mat"]], lw=1.6, ls="--" if r["fracture_model"] != "none" else "-",
                label=f'{c.replace("AM_", "")} ({r["material"]}, {r["mass_g"]:.0f} g)')
    ax.set_xlim(0, 150); ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel("Crush (mm)")
    ax.legend(frameon=False, fontsize=7)
axs[0].set_ylabel("Force, CFC600 (kN)")
fig.suptitle("Force-crush, auxetic geometry (dashed = with Johnson-Cook fracture)", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(out, "figures", "figA2_force_crush_auxetic.png"), dpi=200); plt.close(fig)
print(len(rows), "rows")
for r in rows:
    print(f'{r["case"]:12s} {r["material"]:11s} {r["sizing"]:20s} {r["fracture_model"][:4]:4s} m={r["mass_g"]:6.1f} EA={r["EA_J"]/1e3:6.2f} SEA={r["SEA_kJ_per_kg"]:5.1f} '
          f'PCF={r["PCF_CFC600_kN"]:6.1f} MCF={r["MCF_kN"]:6.1f} CFE={r["CFE_CFC600"]:.2f} err={r["energy_error_max_abs_pct"]} ok={r["passes_validity_check"]} '
          f'dm={r["mass_g_vs_ref_pct"]:+.0f}% dEA={r["EA_J_vs_ref_pct"]:+.0f}% dPCF={r["PCF_CFC600_kN_vs_ref_pct"]:+.0f}%')

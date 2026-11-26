#!/usr/bin/env python3
"""Cost vs crush-force table for the alternative-material study (raw material cost only).
Usage: build_altmat_cost.py <alt_materials_dir>  (reads table_B_FE_material_comparison.csv there)"""
import csv, os, sys
d = sys.argv[1]
B = {r["case"]: r for r in csv.DictReader(open(os.path.join(d, "table_B_FE_material_comparison.csv")))}
# USD/kg. Sources (see README): metal basis and semi-finished basis; None = no sourced figure.
PRICE = {
 "AA6061-T6": dict(metal=2.94, metal_src="LME aluminium 3-month offer USD 2,940/t, 17 Nov 2026 (AlCircle, 18 Nov 2026); primary ingot, alloy/extrusion premium not included",
                   semi=None, semi_src="no sourced 6061/6063 extrusion price"),
 "Mild steel": dict(metal=1330 / 907.185, metal_src="US HR coil, Nucor spot USD 1,330 per short ton (Steel Market Update, 23 Nov 2026)",
                    semi=1330 / 907.185, semi_src="same (HR coil is already a semi-finished sheet product)"),
 "Mg AZ31B": dict(metal=2.63, metal_src="China Mg alloy FOB USD 2,590-2,670/t, midpoint (SMM weekly review, 13 Aug 2026)",
                  semi=16.0, semi_src="AZ31B/AZ91 finished sheet USD 12-20/kg, midpoint, indicative (Saber Alloy price guide, reviewed Oct 2026)"),
 "Ti-6Al-4V": dict(metal=8.29, metal_src="Grade 5 ingot, China, USD 8.22-8.36/kg, midpoint (HonTitan 14 Jul 2026, citing SMM Nov 2025)",
                   semi=30.5, semi_src="Grade 5 sheet/plate China export USD 25.5-35.5/kg, midpoint, undated (HonTitan citing Accio B2B)"),
}
PICK = [("B000", "your suggested: conventional AA6061-T6 extrusion"), ("AM_B_ST_T", "your suggested: steel, same wall"),
        ("AM_B_MG_T", "alternative"), ("AM_B_MG_TF", "alternative (with fracture)"), ("AM_B_TI_M", "alternative"), ("AM_B_TI_MF", "alternative (with fracture)"),
        ("Q6M_S", "recommended auxetic, AA6061-T6"), ("AM_Q_ST_T", "auxetic, steel same wall"),
        ("AM_Q_MG_T", "alternative"), ("AM_Q_MG_TF", "alternative (with fracture)"), ("AM_Q_TI_S", "alternative"), ("AM_Q_TI_SF", "alternative (with fracture)")]
rows = []
for c, role in PICK:
    r = B[c]; p = PRICE[r["material"]]; m = float(r["mass_g"]) / 1000; ea = float(r["EA_J"]) / 1000
    ref = B[r["reference"]]; pref = PRICE["AA6061-T6"]["metal"] * float(ref["mass_g"]) / 1000
    cm = p["metal"] * m
    rows.append(dict(case=c, role=role, geometry=r["geometry"], material=r["material"], sizing=r["sizing"], fracture_model=r["fracture_model"],
                     wall_t_mm=r["t_mm"], mass_per_box_g=round(float(r["mass_g"]), 1),
                     initial_peak_crush_force_PCF_kN=round(float(r["PCF_CFC600_kN"]), 1), mean_crush_force_MCF_kN=round(float(r["MCF_kN"]), 1),
                     energy_absorbed_EA_kJ=round(ea, 2), SEA_kJ_per_kg=round(float(r["SEA_kJ_per_kg"]), 1), CFE=round(float(r["CFE_CFC600"]), 2),
                     price_metal_USD_per_kg=round(p["metal"], 2), material_cost_metal_basis_USD_per_box=round(cm, 2),
                     material_cost_vs_AA6061_same_geometry=round(cm / pref, 2),
                     price_semi_USD_per_kg=(round(p["semi"], 2) if p["semi"] else ""),
                     material_cost_semi_basis_USD_per_box=(round(p["semi"] * m, 2) if p["semi"] else ""),
                     USD_per_kJ_metal_basis=round(cm / ea, 3), price_sources=p["metal_src"] + " | " + p["semi_src"]))
with open(os.path.join(d, "table_C_cost_vs_force.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r["case"].ljust(11), r["material"].ljust(10), r["mass_per_box_g"], r["initial_peak_crush_force_PCF_kN"], r["mean_crush_force_MCF_kN"], r["energy_absorbed_EA_kJ"], "$", r["material_cost_metal_basis_USD_per_box"], r["material_cost_vs_AA6061_same_geometry"], r["material_cost_semi_basis_USD_per_box"])

# ---- figures: crush force and cost per box ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
COL = {"AA6061-T6": "#257de0", "Mg AZ31B": "#f47230", "Ti-6Al-4V": "#00b894", "Mild steel": "#f2a600"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#999",
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
def lab(r): return f'{r["case"].replace("AM_", "")}\n{r["material"]}\n{r["mass_per_box_g"]:.0f} g'
fig, axs = plt.subplots(2, 1, figsize=(10, 6.6))
for ax, geom in zip(axs, ("conventional tube", "auxetic (Q6M_S)")):
    rs = [r for r in rows if r["geometry"] == geom]; x = np.arange(len(rs)); w = 0.38
    for i, r in enumerate(rs):
        hatch = "///" if r["fracture_model"] != "none" else None
        ax.bar(x[i] - w / 2 - 0.01, r["initial_peak_crush_force_PCF_kN"], w, color=COL[r["material"]], hatch=hatch, edgecolor="white", linewidth=0)
        ax.bar(x[i] + w / 2 + 0.01, r["mean_crush_force_MCF_kN"], w, color=COL[r["material"]], alpha=0.45, hatch=hatch, edgecolor="white", linewidth=0)
        ax.text(x[i] - w / 2, r["initial_peak_crush_force_PCF_kN"] + 3, f'{r["initial_peak_crush_force_PCF_kN"]:.0f}', ha="center", fontsize=7, color="#333")
        ax.text(x[i] + w / 2, r["mean_crush_force_MCF_kN"] + 3, f'{r["mean_crush_force_MCF_kN"]:.0f}', ha="center", fontsize=7, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([lab(r) for r in rs], fontsize=7); ax.set_ylabel("Force (kN)")
    ax.set_title(f"{geom[0].upper() + geom[1:]}: solid bar = peak force where crushing starts (PCF), light bar = mean crush force (MCF); hatched = with fracture",
                 fontsize=8.5, loc="left")
fig.suptitle("At what force each material crushes (OpenRadioss, 557 kg at 55 km/h per box)", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(d, "figures", "figA3_crush_force_by_material.png"), dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (key, title) in zip(axs, (("material_cost_metal_basis_USD_per_box", "Raw metal price basis"),
                                  ("material_cost_semi_basis_USD_per_box", "Semi-finished sheet basis (Al extrusion price not sourced)"))):
    rs = [r for r in rows if r["fracture_model"] == "none"]; x = np.arange(len(rs))
    for i, r in enumerate(rs):
        v = r[key]
        if v == "": ax.text(x[i], 0.3, "n/a", ha="center", fontsize=7, color="#666"); continue
        ax.bar(x[i], v, 0.7, color=COL[r["material"]]); ax.text(x[i], v + 0.15, f"{v:.2f}", ha="center", fontsize=7, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([f'{r["case"].replace("AM_", "")}\n{r["material"]}' for r in rs], fontsize=6.5, rotation=0)
    ax.set_title(title, fontsize=9, loc="left")
axs[0].set_ylabel("Material cost per crash box (USD)")
fig.suptitle("Material cost per crash box (mass from FE x sourced price; processing and forming not included)", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(d, "figures", "figA4_material_cost_per_box.png"), dpi=200); plt.close(fig)

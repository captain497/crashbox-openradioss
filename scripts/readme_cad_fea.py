#!/usr/bin/env python3
"""Write README_CAD_FEA_EXPORT.md for the export folder. usage: readme_cad_fea.py <out_dir>"""
import csv, json, os, sys

out = sys.argv[1]
S = {r["case"]: r for r in json.load(open(os.path.join(out, "export_summary.json")))}
V = {r["case"]: r for r in csv.DictReader(open(os.path.join(out, "verification.csv")))}
geo = lambda c: "conventional tube" if c == "B000" or c.startswith("AM_B") else "auxetic (Q6M_S)"
rows = []
for c, s in S.items():
    v = V[c]
    tc = f"{s['tc_mm']:g}" if s["tc_mm"] else "-"
    rows.append(f"| {c} | {geo(c)} | {s['material']} | {'yes' if s['fracture'] else 'no'} | {s['t_mm']:g} | {tc} | "
                f"{s['nodes']} | {s['shells']} | {s['run_mass_g']:.1f} | {float(v['cad_solid_mass_err_pct']):+.2f} | "
                f"{'pass' if v['all_ok'] == 'True' else 'FAIL'} |")
ok = all(v["all_ok"] == "True" for v in V.values())
txt = f"""# Crash box models for SolidWorks and ANSYS (27 Nov 2026)

These are the 22 crash box models of the alternative-materials study (table B in `../alt_materials/`),
converted from the OpenRadioss decks into files that SolidWorks and ANSYS open. Crash box only; no car models.

**These files are model inputs, not results.** Nothing in this folder has been run in LS-DYNA or ANSYS.
Every result in the project still comes from the OpenRadioss runs. If you want ANSYS numbers in the report,
run the `.k` decks in LS-DYNA and report them as a separate solver cross-check.

## Which file opens where
| File (one folder per case) | Opens in | What it holds |
|---|---|---|
| `<case>_solid.step` | **SolidWorks** (File > Open), ANSYS SpaceClaim / DesignModeler / Workbench Geometry | One solid body: tube wall and core ribs at their real thickness, mm |
| `<case>_midsurface.step` | **SolidWorks** (surface bodies), **ANSYS** SpaceClaim / Workbench (shell model, Explicit Dynamics or LS-DYNA Workbench) | Mid-surfaces of tube and ribs, split so shared edges match; named TUBE_t... and CORE_t... with the thickness in the name |
| `<case>_solid.igs`, `<case>_midsurface.igs` | Both, as a fallback if STEP fails | Same geometry as IGES. The solid comes in as closed faces: in SolidWorks tick "Try forming solid(s)" |
| `<case>.k` | **ANSYS LS-DYNA** (LS-Run, LS-PrePost, Workbench LS-DYNA) | Full crash set-up: mesh, Johnson-Cook material (with fracture where the run had it), shell sections, self contact, 557 kg moving rigid wall at 15.28 m/s, clamped base, outputs |
| `<case>.bdf` (Nastran) and `<case>.inp` (Abaqus) | **ANSYS Mechanical** via External Model | The same mesh with shell thickness and elastic constants only |
| `<case>_0000.rad`, `<case>_0001.rad` | OpenRadioss | The original deck, regenerated with the exact arguments of the run |
| `<case>_info.json` | any text editor | The design and run parameters |

SolidWorks cannot read solver decks (`.k`, `.bdf`, `.inp`). STEP is the format that opens in both programs.

Units: CAD in mm. Every FE file uses g, mm, ms (force N, stress MPa, energy mJ, density g/mm3).
In LS-DYNA, results come out in those units (e.g. rigid wall force in N, time in ms).

## What has to be re-run, and what will differ
- Run each `.k` in LS-DYNA (R9 or newer; shell ELFORM 16, MAT 098 or MAT 015). End time is the same as the OpenRadioss run (about 11.3 ms).
- Read the crush force from `rwforc`. Filter it with CFC 600 as in the OpenRadioss post-processing before comparing PCF.
- Expect small differences from OpenRadioss, because these items are mapped, not identical:
  - Shell element: OpenRadioss QEPH (Ishell 24) becomes LS-DYNA ELFORM 16 with hourglass type 8, 5 points through the thickness.
  - Material: `/MAT/PLAS_JOHNS` becomes `*MAT_SIMPLIFIED_JOHNSON_COOK` (no fracture). Where the run used `/FAIL/JOHNSON`, it becomes `*MAT_JOHNSON_COOK` with the same D1 to D5, thermal softening off, and element deletion when all 5 points fail.
  - The strain-rate smoothing of the OpenRadioss JC law (Fcut 10 kHz) has no direct equivalent and is left out.
  - Contact: `/INTER/TYPE7` becomes `*CONTACT_AUTOMATIC_SINGLE_SURFACE` (friction 0.15, IGNORE=1 for the rib-wall junctions).
  - Impactor: the tied moving `/RWALL` becomes `*RIGIDWALL_PLANAR_MOVING` with FRIC=1 (no sliding), same mass and speed, 0.5 mm initial gap.
- For an ANSYS Explicit Dynamics (Workbench) model built from the STEP files, you must enter the material, contact, wall and mesh yourself. Use the values in the `.k` file header cards or in `../alt_materials/table_A_screening.csv`.

## Simplifications in the CAD (the FE files are exact)
- The crush trigger dent of the auxetic design (0.0061 mm amplitude) is in the meshes but left out of the CAD.
- In the CAD each core rib stops at its exact recess height. The mesh stops at the nearest 2.5 mm element row.
- At rib-to-rib and rib-to-wall junctions the solid fuses the material, while the shell model overlaps it. So the solid mass is 0 to 0.8% below the run mass (column below).

## The 22 cases
| Case | Geometry | Material | Fracture | t tube mm | t core mm | Nodes | Shells | Run mass g | CAD solid mass vs run % | Checks |
|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

## How the files were checked
`verification.csv` lists every check. In short:
- Each OpenRadioss deck was regenerated from the run's recorded arguments. Node count, shell count, mass and the engine file all match the run on the `results` branch.
- `.k`: read back with the PyDyna (ansys-dyna-core 0.12.1) keyword reader. Every keyword is recognised; the node count, element count, wall mass and speed are right, and the mass from the deck matches the run to within 1e-6.
- `.bdf` and `.inp`: read back with meshio 5. Node and element counts match.
- STEP and IGES: re-imported with gmsh 4.15 (OpenCASCADE). Surface count, area and solid volume match what was written. Each solid STEP is one closed body, and each mid-surface STEP was meshed with 2.5 mm quads in gmsh.
- Not checked: opening in SolidWorks or ANSYS themselves (not available here), and running the `.k` decks in LS-DYNA.

Overall: **{'all checks passed' if ok else 'SOME CHECKS FAILED, see verification.csv'}**.

## Rebuild
On branch `claude/project-thread-7638su` of `captain497/crashbox-openradioss`:
`python3 scripts/export_cad_fea.py <dir with each run's _info.json and _0001.rad> <out>`, then
`python3 scripts/verify_cad_fea.py <out>` and `python3 scripts/readme_cad_fea.py <out>`.
Needs numpy, gmsh, meshio and ansys-dyna-core.
"""
open(os.path.join(out, "README_CAD_FEA_EXPORT.md"), "w").write(txt)
print("ok" if ok else "FAIL")

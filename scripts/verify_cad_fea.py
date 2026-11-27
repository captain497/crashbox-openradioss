#!/usr/bin/env python3
"""Check the files written by export_cad_fea.py with independent open-source readers:
  .k            ansys-dyna-core (PyDyna) keyword reader: every keyword must be recognised,
                node/element counts, part thicknesses, wall mass/speed and mass from the deck
  .bdf / .inp   meshio readers: node and quad counts
  .step / .igs  gmsh (OpenCASCADE) re-import: surface/volume count, area, volume
  midsurface    gmsh shell mesh of the mid-surface STEP (shows it meshes in a pre-processor)
usage: verify_cad_fea.py <out_dir>   -> writes <out_dir>/verification.csv
"""
import csv, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import meshio, gmsh
from ansys.dyna.core.lib.deck import Deck

out = sys.argv[1]
summ = {r["case"]: r for r in json.load(open(os.path.join(out, "export_summary.json")))}
rows = []
for c, s in summ.items():
    d = os.path.join(out, c); r = dict(case=c)
    deck = Deck(); deck.import_file(os.path.join(d, f"{c}.k"))
    kws = [k for k in deck.keywords if hasattr(k, "keyword")]
    r["k_unrecognised_keywords"] = len(deck.string_keywords) if hasattr(deck, "string_keywords") else 0
    node = [k for k in kws if k.keyword == "NODE"][0].nodes
    els = [k for k in kws if k.keyword == "ELEMENT" and k.subkeyword == "SHELL"][0].elements
    secs = {int(k.secid): float(k.t1) for k in kws if k.keyword == "SECTION"}
    parts = {int(p): int(sid) for k in kws if k.keyword == "PART" for p, sid in zip(k.parts["pid"], k.parts["secid"])}
    mat = [k for k in kws if k.keyword == "MAT"][0]
    rw = [k for k in kws if k.keyword == "RIGIDWALL"][0]
    xyz = node.set_index("nid")[["x", "y", "z"]]
    mass = 0.0
    for pid, grp in els.groupby("pid"):
        P = [xyz.loc[grp[f"n{i}"]].to_numpy() for i in (1, 2, 3, 4)]
        a = 0.5 * np.linalg.norm(np.cross(P[2] - P[0], P[3] - P[1]), axis=1)
        mass += float(mat.ro) * secs[parts[int(pid)]] * a.sum()
    r.update(k_keywords=len(kws), k_nodes=len(node), k_shells=len(els), k_material=mat.__class__.__name__,
             k_wall_mass_kg=float(rw.mass) / 1e3, k_wall_v0_m_s=float(rw.v0), k_mass_g=mass,
             run_mass_g=s["run_mass_g"], k_mass_err_pct=100 * (mass / s["run_mass_g"] - 1),
             k_ok=len(node) == s["nodes"] and len(els) == s["shells"] and abs(mass / s["run_mass_g"] - 1) < 1e-6)
    for ext in ("bdf", "inp"):
        m = meshio.read(os.path.join(d, f"{c}.{ext}"), file_format="nastran" if ext == "bdf" else "abaqus")
        nq = sum(len(b.data) for b in m.cells if b.type == "quad")
        r[f"{ext}_nodes"], r[f"{ext}_quads"] = len(m.points), nq
        r[f"{ext}_ok"] = len(m.points) == s["nodes"] and nq == s["shells"]
    for kind in ("midsurface", "solid"):
        for ext in ("step", "igs"):
            gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.occ.importShapes(os.path.join(d, f"{c}_{kind}.{ext}")); gmsh.model.occ.synchronize()
            ns, nv = len(gmsh.model.getEntities(2)), len(gmsh.model.getEntities(3))
            area = sum(gmsh.model.occ.getMass(2, t) for _, t in gmsh.model.getEntities(2))
            vol = sum(gmsh.model.occ.getMass(3, t) for _, t in gmsh.model.getEntities(3))
            if kind == "midsurface" and ext == "step":
                gmsh.option.setNumber("Mesh.MeshSizeMax", 2.5); gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.model.mesh.generate(2)
                r["mid_step_gmsh_shell_elements"] = sum(len(t) for t in gmsh.model.mesh.getElements(2)[1])
            gmsh.finalize()
            if kind == "midsurface":
                r[f"mid_{ext}_surfaces"], r[f"mid_{ext}_area_mm2"] = ns, round(area, 1)
                r[f"mid_{ext}_ok"] = ns == s["mid_surfaces"] and abs(area / (s["mid_area_tube_mm2"] + s["mid_area_core_mm2"]) - 1) < 1e-4
            elif ext == "step":
                r["solid_step_volumes"], r["solid_step_volume_mm3"], solid_area = nv, round(vol, 1), area
                r["solid_step_ok"] = nv == 1 and abs(vol / s["solid_volume_mm3"] - 1) < 1e-3
            else:  # IGES has no solid entity in gmsh's writer: the solid arrives as its closed set of boundary faces
                r["solid_igs_faces"], r["solid_igs_area_mm2"] = ns, round(area, 1)
                r["solid_igs_ok"] = abs(area / solid_area - 1) < 1e-4
    r["cad_solid_mass_err_pct"] = 100 * (s["cad_solid_mass_g"] / s["run_mass_g"] - 1)
    r["cad_mid_mass_err_pct"] = 100 * (s["cad_mid_mass_g"] / s["run_mass_g"] - 1)
    r["all_ok"] = all(v for k, v in r.items() if k.endswith("_ok")) and r["k_unrecognised_keywords"] == 0
    rows.append(r); print(c, "OK" if r["all_ok"] else "FAIL", {k: v for k, v in r.items() if k.endswith("_ok") and not v})
with open(os.path.join(out, "verification.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("all files OK" if all(r["all_ok"] for r in rows) else "SOME CHECKS FAILED")

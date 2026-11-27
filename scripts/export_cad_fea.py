#!/usr/bin/env python3
"""Export crash box models (OpenRadioss decks from gen_crashbox.py) to files for
SolidWorks and ANSYS.

For each case (an *_info.json written by gen_crashbox.py for a real run) this:
  1. regenerates the OpenRadioss deck with exactly the recorded arguments and checks
     node / shell counts and mass against the recorded run;
  2. converts the deck to
       <case>.k      LS-DYNA keyword deck (ANSYS LS-DYNA / Workbench LS-DYNA / LS-PrePost)
       <case>.bdf    Nastran bulk data mesh (ANSYS Mechanical "External Model", SolidWorks Simulation import is NOT supported)
       <case>.inp    Abaqus mesh (ANSYS Mechanical "External Model")
  3. writes CAD from the exact design parameters (not from the mesh):
       <case>_midsurface.step / .igs   surface bodies on the shell mid-surfaces (ANSYS SpaceClaim/Workbench shell models, SolidWorks)
       <case>_solid.step / .igs        thickened solid of tube + core (SolidWorks part / ANSYS solid models)

Units of every FE file: g, mm, ms (force N, stress MPa, energy mJ). CAD files: mm.
The converted decks are INPUTS. They have not been run; no result in this project comes from them.

usage: export_cad_fea.py <info_dir> <out_dir> [case ...]
"""
import json, math, os, re, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_KEYS = ["width", "length", "t", "core", "inner_frac", "ti", "theta", "ncell", "beta", "tc", "esize",
            "mass_kg", "v", "stroke_frac", "tend", "dt_cst", "wall_slide", "ishell", "car", "rail_kN",
            "trig", "trig_z", "trig_w", "core_gap", "core_bevel", "mat", "fail"]
MAT_NAME = {"aa6061": "AA6061-T6", "aa7075": "AA7075-T6", "aa2024": "AA2024-T3", "az31b": "Mg AZ31B",
            "ti64": "Ti-6Al-4V", "steel_sk": "Mild steel (Simpson & Kazanci 2020)"}


# ---------------------------------------------------------------- OpenRadioss deck
def regenerate(info, workdir):
    args = [sys.executable, os.path.join(HERE, "gen_crashbox.py"), "--name", info["name"], "--anim", "0", "--out", workdir]
    for k in GEN_KEYS:
        if k in info and info[k] is not None:
            args += [f"--{k}", str(info[k])]
    subprocess.run(args, check=True, capture_output=True)
    return os.path.join(workdir, f"{info['name']}_0000.rad")


def parse_rad(path):
    blocks, cur = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("/"):
            cur = line.strip(); blocks[cur] = []
        elif cur is not None and not line.startswith("#"):
            blocks[cur].append(line)
    f20 = lambda s: [float(s[i:i + 20]) for i in range(0, len(s.rstrip()), 20)]
    i10 = lambda s: [int(s[i:i + 10]) for i in range(0, len(s.rstrip()), 10)]
    nodes = {}
    for ln in blocks["/NODE"]:
        nodes[int(ln[:10])] = tuple(float(ln[10 + 20 * k:30 + 20 * k]) for k in range(3))
    shells = {}
    for key, lines in blocks.items():
        m = re.fullmatch(r"/SHELL/(\d+)", key)
        if m:
            shells[int(m.group(1))] = [tuple(i10(ln)) for ln in lines]
    thick = {}
    for key, lines in blocks.items():
        m = re.fullmatch(r"/PROP/SHELL/(\d+)", key)
        if m:
            thick[int(m.group(1))] = float(lines[3][20:40])
    partname = {int(re.fullmatch(r"/PART/(\d+)", k).group(1)): v[0].strip() for k, v in blocks.items() if re.fullmatch(r"/PART/\d+", k)}
    mk = [k for k in blocks if k.startswith("/MAT/PLAS_JOHNS/")][0]
    ml = blocks[mk]
    rho = f20(ml[1])[0]; E, nu = f20(ml[2][:40])
    A, B, n = f20(ml[3])[:3]; C, eps0 = f20(ml[4][:40])
    mat = dict(rho=rho, E=E, nu=nu, A=A, B=B, n=n, C=C, eps0=eps0)
    fail = None
    if "/FAIL/JOHNSON/1" in blocks:
        D = f20(blocks["/FAIL/JOHNSON/1"][0]); feps0 = f20(blocks["/FAIL/JOHNSON/1"][1][:20])[0]
        fail = dict(D=D, eps0=feps0)
    bottom = [x for ln in blocks["/GRNOD/NODE/1"][1:] for x in i10(ln)]
    rw = blocks["/RWALL/PLANE/1"]
    wall_node = i10(rw[1])[0]
    mass_g, _, _, vz = f20(rw[3])
    zt = nodes[wall_node][2]; zh = f20(rw[4])[2]
    return dict(nodes=nodes, shells=shells, thick=thick, partname=partname, mat=mat, fail=fail,
                bottom=bottom, wall_node=wall_node, wall_mass=mass_g, wall_v=-vz, wall_zt=zt, wall_zh=zh,
                fric=0.15)


def quad_area(P):
    return 0.5 * np.linalg.norm(np.cross(P[2] - P[0], P[3] - P[1]))


def fe_mass(m):
    tot = 0.0
    for pid, els in m["shells"].items():
        for e in els:
            tot += m["mat"]["rho"] * m["thick"][pid] * quad_area([np.array(m["nodes"][k]) for k in e[1:]])
    return tot


# ---------------------------------------------------------------- writers
def fw(x, w=10):
    """Float in a fixed-width LS-DYNA / Nastran field."""
    if x == 0: return f"{0.0:>{w}.1f}"
    for p in range(w - 2, 0, -1):
        s = f"{x:.{p}g}"
        if "e" not in s and "." not in s: s += "."
        if len(s) <= w: return f"{s:>{w}}"
    raise ValueError(x)


def card(*vals, w=10):
    out = ""
    for v in vals:
        if v is None or v == "": out += " " * w
        elif isinstance(v, str): out += f"{v:<{w}}"
        elif isinstance(v, int): out += f"{v:>{w}d}"
        else: out += fw(float(v), w)
    return out


def write_k(m, info, tend, path):
    mt, nm = m["mat"], info["name"]
    struct = [n for n in m["nodes"] if n != m["wall_node"]]
    L = ["*KEYWORD", "*TITLE", f"{nm} {MAT_NAME[info.get('mat', 'aa6061')]} crash box (converted from OpenRadioss; units g mm ms)",
         "$ Converted by scripts/export_cad_fea.py from the OpenRadioss deck of the run with the same name.",
         "$ Units: mass g, length mm, time ms -> force N, stress MPa, energy mJ, density g/mm3.",
         "$ This deck is an INPUT. It has not been run in LS-DYNA. Re-run it before quoting any number.",
         "$ OpenRadioss -> LS-DYNA mapping:",
         "$   /MAT/PLAS_JOHNS  -> *MAT_SIMPLIFIED_JOHNSON_COOK (no fracture) or *MAT_JOHNSON_COOK (with /FAIL/JOHNSON)",
         "$   QEPH shell Ishell=24, 5 pts -> ELFORM 16 + hourglass IHQ 8, NIP 5 (formulations differ slightly)",
         "$   /INTER/TYPE7 self contact, fric 0.15 -> *CONTACT_AUTOMATIC_SINGLE_SURFACE FS=FD=0.15",
         "$   /RWALL/PLANE moving, tied -> *RIGIDWALL_PLANAR_MOVING, FRIC=1 (no sliding), same mass and speed",
         "$   /BCS 111 111 on bottom ring -> *BOUNDARY_SPC_SET all 6 DOF",
         "$   JC strain-rate filter (Fsmooth=1, Fcut=10/ms) has no direct equivalent here and is not carried over.",
         "*CONTROL_TERMINATION", "$    endtim", card(round(tend, 4)),
         "*CONTROL_TIMESTEP", "$   dtinit    tssfac", card(0.0, 0.9),
         "*CONTROL_SHELL", "$   wrpang     esort     irnxx    istupd    theory       bwc     miter      proj",
         card(20.0, 0, -1, 1, 2, 2, 1, 0),
         "*CONTROL_CONTACT", "$   slsfac    rwpnal    islchk    shlthk    penopt    thkchg     orien    enmass",
         card(0.1, 0.0, 1, 1, 1, 0, 1, 0), "$   usrstr    usrfrc     nsbcs    interm     xpene     ssthk      ecdt   tiedprj",
         card(0, 0, 0, 0, 4.0, 0, 0, 0), "$    sfric     dfric       edc       vfc        th     th_sf    pen_sf",
         card(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), "$   ignore    frceng   skiprwg    outseg   spotstp   spotdel   spothin",
         card(1, 0, 0, 0, 0, 0, 0.0),
         "*CONTROL_ENERGY", "$     hgen      rwen    slnten     rylen", card(2, 2, 2, 2),
         "*CONTROL_HOURGLASS", "$      ihq        qh", card(8, 0.1),
         "*DATABASE_GLSTAT", card(0.005), "*DATABASE_MATSUM", card(0.005), "*DATABASE_RWFORC", card(0.005),
         "*DATABASE_NODOUT", card(0.005), "*DATABASE_BINARY_D3PLOT", "$       dt", card(round(tend / 40, 4)),
         "*DATABASE_HISTORY_NODE_SET", card(2)]
    # material
    G = mt["E"] / (2 * (1 + mt["nu"]))
    if m["fail"] is None:
        L += ["*MAT_SIMPLIFIED_JOHNSON_COOK_TITLE", MAT_NAME[info.get("mat", "aa6061")],
              "$      mid        ro         e        pr        vp", card(1, mt["rho"], mt["E"], mt["nu"], 0.0),
              "$        a         b         n         c    psfail    sigmax    sigsat      epso",
              card(mt["A"], mt["B"], mt["n"], mt["C"], 1.0e17, 1.0e28, 1.0e28, mt["eps0"])]
    else:
        D = m["fail"]["D"]
        if abs(m["fail"]["eps0"] - mt["eps0"]) > 1e-12 and abs(D[3]) > 0:
            raise SystemExit("plasticity and fracture reference rates differ while D4 != 0")
        L += ["*MAT_JOHNSON_COOK_TITLE", MAT_NAME[info.get("mat", "aa6061")] + " with JC fracture",
              "$ Thermal softening switched off: M=1 with TM=1e20 (isothermal, as in the OpenRadioss run).",
              "$ NUMINT=5: shell deleted when all 5 through-thickness points have failed (OpenRadioss Ifail_sh=2).",
              "$      mid        ro         g         e        pr       dtf        vp    rateop",
              card(1, mt["rho"], G, mt["E"], mt["nu"], 0.0, 0.0, 0.0),
              "$        a         b         n         c         m        tm        tr      epso",
              card(mt["A"], mt["B"], mt["n"], mt["C"], 1.0, 1.0e20, 0.0, mt["eps0"]),
              "$       cp        pc     spall        it        d1        d2        d3        d4",
              card(1.0e20, -1.0e20, 2.0, 0.0, D[0], D[1], D[2], D[3]),
              "$       d5      c2/p      erod     efmin    numint",
              card(D[4], 0.0, 0.0, 1.0e-6, 5.0)]
    L += ["*HOURGLASS", "$     hgid       ihq        qm", card(1, 8, 0.1)]
    for pid in sorted(m["shells"]):
        L += ["*SECTION_SHELL_TITLE", m["partname"][pid] + "_SECTION",
              "$    secid    elform      shrf       nip     propt", card(pid, 16, 0.8333, 5, 1.0),
              "$       t1        t2        t3        t4", card(*[m["thick"][pid]] * 4),
              "*PART", m["partname"][pid], "$      pid     secid       mid     eosid      hgid", card(pid, pid, 1, 0, 1)]
    L += ["*SET_PART_LIST_TITLE", "ALL_PARTS", card(1), card(*sorted(m["shells"]))]
    L += ["*NODE", "$    nid               x               y               z      tc      rc"]
    for nid in sorted(struct):
        x, y, z = m["nodes"][nid]
        L.append(f"{nid:8d}{x:16.8f}{y:16.8f}{z:16.8f}{0:8d}{0:8d}")
    L += ["*ELEMENT_SHELL", "$    eid     pid      n1      n2      n3      n4"]
    for pid in sorted(m["shells"]):
        for e in m["shells"][pid]:
            L.append("".join(f"{v:8d}" for v in (e[0], pid) + e[1:]))
    def nset(sid, title, ids):
        out = ["*SET_NODE_LIST_TITLE", title, card(sid)]
        out += [card(*ids[k:k + 8]) for k in range(0, len(ids), 8)]
        return out
    L += nset(1, "CLAMPED_BOTTOM_RING", sorted(m["bottom"]))
    top = sorted(n for n in struct if abs(m["nodes"][n][2] - max(m["nodes"][k][2] for k in struct)) < 1e-6)
    L += nset(2, "IMPACT_END_NODES", top[:: max(1, len(top) // 8)][:8])
    L += nset(3, "ALL_STRUCTURE_NODES", sorted(struct))
    L += ["*BOUNDARY_SPC_SET", "$     nsid       cid      dofx      dofy      dofz     dofrx     dofry     dofrz",
          card(1, 0, 1, 1, 1, 1, 1, 1)]
    L += ["*CONTACT_AUTOMATIC_SINGLE_SURFACE_ID", card(1) + "SELF_CONTACT",
          "$     ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr", card(1, 0, 2, 0, 0, 0, 1, 0),
          "$       fs        fd        dc        vc       vdc    penchk        bt        dt", card(0.15, 0.15, 0.0, 0.0, 10.0, 0, 0.0, 1.0e20),
          "$      sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf", card(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)]
    zt, zh = m["wall_zt"], m["wall_zh"]
    L += ["*RIGIDWALL_PLANAR_MOVING_FORCES_ID", card(1) + "IMPACTOR",
          f"$ {m['wall_mass'] / 1e3:.0f} kg wall moving in -z at {m['wall_v']} mm/ms (= m/s); normal tail->head points at the box.",
          "$     nsid    nsidex     boxid    offset     birth     death     rwksf", card(3, 0, 0, 0.0, 0.0, 1.0e20, 1.0),
          "$       xt        yt        zt        xh        yh        zh      fric      wvel", card(0.0, 0.0, zt, 0.0, 0.0, zh, 1.0, 0.0),
          "$     mass        v0", card(m["wall_mass"], m["wall_v"]),
          "$     soft      ssid      node1     node2     node3     node4", card(0, 0, 0, 0, 0, 0)]
    L.append("*END")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_bdf(m, info, path):
    mt = m["mat"]; struct = sorted(n for n in m["nodes"] if n != m["wall_node"])
    f8 = lambda *v: card(*v, w=8)
    L = [f"$ {info['name']} crash box mesh, {MAT_NAME[info.get('mat', 'aa6061')]}. Units g mm ms (stress MPa, density g/mm3).",
         "$ Mesh, thickness and elastic constants only. Plasticity, fracture, contact, impactor and BCs are in the .k deck.",
         "$ Converted from the OpenRadioss deck by scripts/export_cad_fea.py.", "BEGIN BULK"]
    L.append(f8("MAT1", 1, mt["E"], None, mt["nu"], mt["rho"]))
    for pid in sorted(m["shells"]):
        L.append(f"$ PSHELL {pid} = {m['partname'][pid]}")
        L.append(f8("PSHELL", pid, 1, m["thick"][pid], 1, None, 1))
    for nid in struct:
        x, y, z = m["nodes"][nid]
        # large-field GRID* for full coordinate precision
        L.append(f"{'GRID*':<8}{nid:>16d}{0:>16d}{x:>16.9g}{y:>16.9g}*")
        L.append(f"{'*':<8}{z:>16.9g}")
    for pid in sorted(m["shells"]):
        for e in m["shells"][pid]:
            L.append(f8("CQUAD4", e[0], pid, *e[1:]))
    L.append("ENDDATA")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_inp(m, info, path):
    mt = m["mat"]; struct = sorted(n for n in m["nodes"] if n != m["wall_node"])
    L = ["*HEADING", f"{info['name']} crash box mesh, {MAT_NAME[info.get('mat', 'aa6061')]}, units g mm ms (MPa, g/mm3)",
         "** Mesh, thickness and elastic constants only; the full crash set-up is in the .k deck.", "*NODE"]
    L += [f"{n}, {m['nodes'][n][0]:.9g}, {m['nodes'][n][1]:.9g}, {m['nodes'][n][2]:.9g}" for n in struct]
    for pid in sorted(m["shells"]):
        L.append(f"*ELEMENT, TYPE=S4R, ELSET={m['partname'][pid]}")
        L += [", ".join(str(v) for v in e) for e in m["shells"][pid]]
    L.append("*NSET, NSET=CLAMPED_BOTTOM_RING")
    b = sorted(m["bottom"]); L += [", ".join(str(v) for v in b[k:k + 16]) for k in range(0, len(b), 16)]
    for pid in sorted(m["shells"]):
        L += [f"*SHELL SECTION, ELSET={m['partname'][pid]}, MATERIAL=MAT1", f"{m['thick'][pid]}, 5"]
    L += ["*MATERIAL, NAME=MAT1", "*DENSITY", f"{mt['rho']:.6g}", "*ELASTIC", f"{mt['E']:.6g}, {mt['nu']:.6g}"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- CAD
def write_cad(info, sec, out_base):
    import gmsh
    H, L = sec["half_width"], sec["length"]
    T, tc = info["t"], info["tc"]
    segs = sec.get("core_segments", []); tops = sec.get("core_top", [])
    res = {}
    # --- mid-surface model: tube walls + core ribs as surfaces, fragmented so shared edges are conformal
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mid"); occ = gmsh.model.occ
    corners = [(-H, -H), (H, -H), (H, H), (-H, H)]
    surfs = []
    def vrect(p0, p1, z1):
        pts = [occ.addPoint(p0[0], p0[1], 0), occ.addPoint(p1[0], p1[1], 0), occ.addPoint(p1[0], p1[1], z1), occ.addPoint(p0[0], p0[1], z1)]
        ls = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        return occ.addPlaneSurface([occ.addCurveLoop(ls)])
    tube = [vrect(corners[i], corners[(i + 1) % 4], L) for i in range(4)]
    core = [vrect(p0, p1, zt) for (p0, p1), zt in zip(segs, tops)]
    occ.synchronize()
    if core:
        out, omap = occ.fragment([(2, s) for s in tube], [(2, s) for s in core])
        occ.synchronize()
        tube_new = [t for k in range(len(tube)) for _, t in omap[k]]
        core_new = sorted({t for k in range(len(tube), len(tube) + len(core)) for _, t in omap[k]} - set(tube_new))
    else:
        tube_new, core_new = tube, []
    gmsh.model.addPhysicalGroup(2, tube_new, name=f"TUBE_t{T:g}mm")
    if core_new: gmsh.model.addPhysicalGroup(2, core_new, name=f"CORE_t{tc:g}mm")
    a_tube = sum(occ.getMass(2, s) for s in tube_new); a_core = sum(occ.getMass(2, s) for s in core_new)
    res.update(mid_area_tube_mm2=a_tube, mid_area_core_mm2=a_core, mid_surfaces=len(tube_new) + len(core_new))
    for ext in ("step", "igs"):
        gmsh.write(f"{out_base}_midsurface.{ext}")
    gmsh.finalize()
    # --- solid model: tube wall thickness T about the mid-surface, ribs thickness tc, fused
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("solid"); occ = gmsh.model.occ
    Ho, Hn = H + T / 2, H - T / 2
    outer = occ.addBox(-Ho, -Ho, 0, 2 * Ho, 2 * Ho, L); inner = occ.addBox(-Hn, -Hn, 0, 2 * Hn, 2 * Hn, L)
    tube_v, _ = occ.cut([(3, outer)], [(3, inner)])
    ribs = []
    for (p0, p1), zt in zip(segs, tops):
        d = np.array(p1) - np.array(p0); Ls = np.linalg.norm(d); u = d / Ls; nrm = np.array([-u[1], u[0]])
        a0 = np.array(p0) - u * tc / 2; a1 = np.array(p1) + u * tc / 2   # extend by tc/2 to close rib junctions
        P = [a0 + nrm * tc / 2, a1 + nrm * tc / 2, a1 - nrm * tc / 2, a0 - nrm * tc / 2]
        pts = [occ.addPoint(x, y, 0) for x, y in P]
        ls = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        s = occ.addPlaneSurface([occ.addCurveLoop(ls)])
        ribs += [e for e in occ.extrude([(2, s)], 0, 0, zt) if e[0] == 3]
    if ribs:
        core_v, _ = occ.fuse(ribs[:1], ribs[1:]) if len(ribs) > 1 else (ribs, None)
        clip = occ.addBox(-Ho, -Ho, 0, 2 * Ho, 2 * Ho, L)          # trim rib ends that poke out of the tube
        core_v, _ = occ.intersect(core_v, [(3, clip)])
        body, _ = occ.fuse(tube_v, core_v)
    else:
        body = tube_v
    occ.removeAllDuplicates(); occ.synchronize()
    vols = gmsh.model.getEntities(3)
    res.update(solid_volumes=len(vols), solid_volume_mm3=sum(occ.getMass(3, v) for _, v in vols))
    gmsh.model.addPhysicalGroup(3, [v for _, v in vols], name="CRASH_BOX")
    for ext in ("step", "igs"):
        gmsh.write(f"{out_base}_solid.{ext}")
    gmsh.finalize()
    return res


def check_cad(path):
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(path); gmsh.model.occ.synchronize()
    d = {dim: len(gmsh.model.getEntities(dim)) for dim in (2, 3)}
    m2 = sum(gmsh.model.occ.getMass(2, s) for _, s in gmsh.model.getEntities(2))
    m3 = sum(gmsh.model.occ.getMass(3, s) for _, s in gmsh.model.getEntities(3))
    gmsh.finalize()
    return dict(surfaces=d[2], volumes=d[3], area_mm2=m2, volume_mm3=m3)


def main():
    info_dir, out_dir = sys.argv[1], sys.argv[2]
    cases = sys.argv[3:] or sorted(f[:-5] for f in os.listdir(info_dir) if f.endswith(".json"))
    summary = []
    for c in cases:
        info = json.load(open(os.path.join(info_dir, c + ".json")))
        od = os.path.join(out_dir, c); os.makedirs(od, exist_ok=True)
        rad = regenerate(info, od)
        new = json.load(open(os.path.join(od, f"{c}_info.json")))
        for k in ("n_nodes", "n_shell_tube", "n_shell_core"):
            if new[k] != info[k]: raise SystemExit(f"{c}: {k} {new[k]} != run {info[k]}")
        if abs(new["mass_total_g"] - info["mass_total_g"]) > 1e-6 * info["mass_total_g"]:
            raise SystemExit(f"{c}: mass differs from run")
        run1 = os.path.join(info_dir, f"{c}_0001.rad")
        if os.path.exists(run1) and open(run1).read().split() != open(os.path.join(od, f"{c}_0001.rad")).read().split():
            raise SystemExit(f"{c}: engine file differs from run")
        m = parse_rad(rad)
        mass_fe = fe_mass(m)
        write_k(m, info, new["tend_ms"], os.path.join(od, f"{c}.k"))
        write_bdf(m, info, os.path.join(od, f"{c}.bdf"))
        write_inp(m, info, os.path.join(od, f"{c}.inp"))
        sec = json.load(open(os.path.join(od, f"{c}_section.json")))
        cad = write_cad(info, sec, os.path.join(od, c))
        os.remove(os.path.join(od, f"{c}_section.json"))
        rho = m["mat"]["rho"]
        cad_mass_mid = rho * (info["t"] * cad["mid_area_tube_mm2"] + info["tc"] * cad["mid_area_core_mm2"])
        row = dict(case=c, material=MAT_NAME[info.get("mat", "aa6061")], fracture=bool(info.get("fail", 0)),
                   t_mm=info["t"], tc_mm=info["tc"] if info["core"] != "none" else None,
                   nodes=len(m["nodes"]) - 1, shells=sum(len(v) for v in m["shells"].values()),
                   run_mass_g=info["mass_total_g"], fe_mass_g=mass_fe, cad_mid_mass_g=cad_mass_mid,
                   cad_solid_mass_g=rho * cad["solid_volume_mm3"], tend_ms=new["tend_ms"], **cad)
        summary.append(row)
        print(json.dumps(row))
    json.dump(summary, open(os.path.join(out_dir, "export_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

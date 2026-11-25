#!/usr/bin/env python3
"""Re-post-process every campaign case on the results branch and build one table."""
import csv, json, os, subprocess, sys, tempfile
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ref = "origin/results"
out_csv = sys.argv[1] if len(sys.argv) > 1 else "campaign_results.csv"
files = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", ref, "campaign/"],
                       capture_output=True, text=True).stdout.split()
cases = sorted({f.split("/")[1] for f in files})
rows = []
for c in cases:
    need = [f"campaign/{c}/{c}T01", f"campaign/{c}/{c}_info.json", f"campaign/{c}/{c}_0001.out"]
    if not all(n in files for n in need):
        rows.append(dict(case=c, valid=False, note="missing output")); continue
    d = tempfile.mkdtemp()
    for n in need + [f"campaign/{c}/runtime.txt", f"campaign/{c}/RUN_INFO.txt"]:
        if n in files:
            with open(os.path.join(d, os.path.basename(n)), "wb") as fh:
                fh.write(subprocess.run(["git", "-C", repo, "show", f"{ref}:{n}"], capture_output=True).stdout)
    r = subprocess.run([sys.executable, os.path.join(repo, "scripts/post_crashbox.py"), d, "--case", c, "--plot", "0"],
                       capture_output=True, text=True)
    try:
        m = json.load(open(os.path.join(d, f"{c}_metrics.json")))
    except Exception:
        rows.append(dict(case=c, valid=False, note="postprocess failed")); continue
    rt = open(os.path.join(d, "runtime.txt")).read().strip().split("=")[-1] if os.path.exists(os.path.join(d, "runtime.txt")) else ""
    m["engine_wall_s"] = rt
    ri = os.path.join(d, "RUN_INFO.txt")
    m["gh_run_id"] = open(ri).read().split()[1] if os.path.exists(ri) else ""
    rows.append(m)
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
ok = sum(1 for r in rows if r.get("valid"))
print(f"{len(rows)} cases, {ok} valid -> {out_csv}")
for r in rows:
    print(r["case"], r.get("valid"), r.get("energy_error_max_abs_pct"), round(r.get("SEA_kJ_per_kg") or 0, 2), round(r.get("PCF_kN") or 0, 1), r.get("engine_wall_s"))

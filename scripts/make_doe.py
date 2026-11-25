#!/usr/bin/env python3
"""Latin hypercube design for the re-entrant filled crash box (fixed seed)."""
import csv, sys
import numpy as np
from scipy.stats import qmc

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
seed = 2026
# design variables and ranges
#  theta : re-entrant angle (deg)            10 - 30   (Ren et al. 2018; Aktas et al. 2023 used 14-22)
#  tc    : core rib (cell wall) thickness mm  0.4 - 1.2
#  ncell : cells across the 80 mm width       2 - 4    (cell quantity)
#  beta  : cell height ratio h/l              1.5 - 3.0 (cell size in the transverse direction)
lo = np.array([10.0, 0.4, 2.0, 1.5]); hi = np.array([30.0, 1.2, 4.999, 3.0])
X = qmc.scale(qmc.LatinHypercube(d=4, seed=seed).random(N), lo, hi)
with open(f"doe/doe_lhs{N}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "theta", "tc", "ncell", "beta"])
    for k, (th, tc, nc, be) in enumerate(X, 1):
        w.writerow([f"D{k:03d}", round(th, 2), round(tc, 3), int(np.clip(np.floor(nc), 2, 4)), round(be, 3)])
print(f"wrote doe/doe_lhs{N}.csv")

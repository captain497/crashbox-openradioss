"""SAE J211 channel frequency class filter (shared by the post-processors)."""
import numpy as np


def cfc(x, dt_s, cls):  # SAE J211 CFC filter (2nd-order Butterworth, forward + backward)
    wd = 2 * np.pi * cls * 2.0775; wa = np.sin(wd * dt_s / 2) / np.cos(wd * dt_s / 2)
    den = 1 + np.sqrt(2) * wa + wa ** 2
    a0 = wa ** 2 / den; a1 = 2 * a0; a2 = a0
    b1 = -2 * (wa ** 2 - 1) / den; b2 = (-1 + np.sqrt(2) * wa - wa ** 2) / den
    def run(u):
        y = np.zeros_like(u); y[:2] = u[:2]
        for i in range(2, len(u)):
            y[i] = a0 * u[i] + a1 * u[i - 1] + a2 * u[i - 2] + b1 * y[i - 1] + b2 * y[i - 2]
        return y
    return run(run(x)[::-1])[::-1]

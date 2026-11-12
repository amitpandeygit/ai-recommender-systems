"""Fitting scaling laws (section 12.2).

A scaling law says how a model's held-out loss falls as one resource x grows (compute,
parameters, training clicks or history length):

    L(x) = L_inf + a * x^(-alpha)

L_inf is the loss no amount of x can remove, and alpha says how fast the rest shrinks:
each time x doubles, the removable part L(x) - L_inf is multiplied by 2^(-alpha).

  fit_power_law(x, y)   - Listing 12.1: least squares for L_inf, a and alpha
  frontier(points)      - the lowest loss reached at or below each compute, over many
                          runs (the "compute-efficient frontier")
  doubling(alpha)       - the factor 2^(-alpha)

Run from the repository root to check the fit on made-up points with a known answer:
    python chapter_12/scaling_laws.py
"""
import numpy as np
from scipy.optimize import least_squares


def fit_power_law(x, y):
    """Listing 12.1. Fit L(x) = L_inf + a * x^(-alpha) by least squares on the losses.
    x is rescaled by its smallest value so that a is the removable loss there. Returns
    (L_inf, a, alpha) with L_inf >= 0, a >= 0 and alpha >= 0."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x0 = x.min()

    def residuals(p):
        l_inf, a, alpha = p
        return l_inf + a * (x / x0) ** (-alpha) - y

    start = [0.9 * y.min(), y.max() - 0.9 * y.min(), 0.3]
    fit = least_squares(residuals, start, bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, 5.0]))
    l_inf, a, alpha = fit.x
    return float(l_inf), float(a * x0 ** alpha), float(alpha)


def predict(params, x):
    l_inf, a, alpha = params
    return l_inf + a * np.asarray(x, dtype=float) ** (-alpha)


def frontier(points):
    """points: (compute, loss) pairs from many runs and checkpoints. Keeps, in order of
    compute, each point whose loss is lower than every point with less compute."""
    out, best = [], np.inf
    for c, l in sorted(points):
        if l < best:
            out.append((c, l))
            best = l
    return out


def doubling(alpha):
    return 2.0 ** (-alpha)


if __name__ == "__main__":
    x = np.array([1e6, 4e6, 1.6e7, 6.4e7, 2.56e8])
    y = 5.0 + 2.0 * (x / 1e6) ** -0.3                    # a known law: L_inf 5, a 2 at x = 1e6, alpha 0.3
    l_inf, a, alpha = fit_power_law(x, y)
    print(f"made-up points from L = 5 + 2 (x / 1e6)^-0.3: fitted L_inf {l_inf:.4f}, alpha {alpha:.4f}, "
          f"removable loss at x = 1e6 {a * 1e6 ** -alpha:.4f}")

"""
Test the distance–SNR relationship by drawing (m1_det, q, Theta, rho) from
uniform distributions and computing the implied luminosity distance via

    rho = rho_0(m1_det, q) * (dL_fid / dL) * Theta
    =>  dL = rho_0(m1_det, q) * dL_fid * Theta / rho

The resulting 5D distribution (m1_det, q, Theta, rho, dL) is displayed as a
hexbin corner plot.  Diagonal panels show marginal histograms; off-diagonal
panels show hexbin 2D densities.

Usage (from repo root):
    python scripts/test_distance_jacobian.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

_repo_root = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N          = 500000
RNG_SEED   = 42
RHO_MIN    = 1
RHO_MAX    = 10.0

SNR_GRID_PATH = _repo_root / "snr_grid_aligo_design.npz"
FIGURES_DIR   = _repo_root / "figures"

# ---------------------------------------------------------------------------
# Load rho_0 interpolant
# ---------------------------------------------------------------------------
data     = np.load(SNR_GRID_PATH)
m1_grid  = data["m1_grid"]   # (N_m1,), detector-frame Msun, log-spaced
q_grid   = data["q_grid"]    # (N_q,),  linear
snr_grid = data["snr_grid"]  # (N_m1, N_q)
dL_fid   = float(data["dL_fid"])  # Gpc

rho0_interp = RegularGridInterpolator(
    (m1_grid, q_grid), snr_grid,
    method="linear", bounds_error=False, fill_value=0.0,
)

# ---------------------------------------------------------------------------
# Draw fiducial samples
# ---------------------------------------------------------------------------
rng   = np.random.default_rng(RNG_SEED)

m1    = rng.uniform(m1_grid.min(), m1_grid.max(), N)   # Msun, detector frame
q     = rng.uniform(q_grid.min(),  q_grid.max(),  N)   # mass ratio
Theta = rng.uniform(0.0, 1.0, N)                        # Finn-Chernoff parameter
rho   = rng.uniform(RHO_MIN, RHO_MAX, N)               # SNR

# Evaluate the interpolant at drawn (m1, q)
rho0 = rho0_interp(np.column_stack([m1, q]))             # (N,)

# Implied luminosity distance [Gpc]
dL = rho0 * dL_fid * Theta / rho

# Useful support variable:
#   dL in [A / RHO_MAX, A / RHO_MIN] with A = rho0 * dL_fid * Theta
A = rho0 * dL_fid * Theta

# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
w      = dL / rho                    # weight proportional to dL / rho
w     /= w.sum()
idx    = rng.choice(N, size=N, replace=True, p=w)
m1_r, q_r, Theta_r, rho_r, dL_r = m1[idx], q[idx], Theta[idx], rho[idx], dL[idx]

# ---------------------------------------------------------------------------
# Reference sample: uniform in (m1, q, Theta, dL) on the allowed support
# ---------------------------------------------------------------------------
dL_max = A.max() / RHO_MIN

def sample_uniform_reference(size):
    """Draw a reference sample uniform in (m1, q, Theta, dL) on the allowed support.

    The SNR relation implies

        dL = A(m1, q, Theta) / rho,

    where A = rho0(m1, q) * dL_fid * Theta. Since rho is restricted to
    [RHO_MIN, RHO_MAX], each triple (m1, q, Theta) admits only the interval

        dL in [A / RHO_MAX, A / RHO_MIN].

    We first draw uniformly from the bounding box

        m1 in [m1_min, m1_max],
        q in [q_min, q_max],
        Theta in [0, 1],
        dL in [0, dL_max],

    with dL_max = max(A) / RHO_MIN, and then reject samples outside the
    allowed dL interval for their (m1, q, Theta). Because the proposal density
    is constant on the bounding box, the accepted points are uniformly
    distributed on the subset satisfying the SNR bounds. This provides a
    direct visual reference for the target 4D distribution that the Jacobian
    resampling is intended to reproduce.
    """
    out_m1 = np.empty(size)
    out_q = np.empty(size)
    out_theta = np.empty(size)
    out_dL = np.empty(size)

    filled = 0
    while filled < size:
        remaining = size - filled
        batch = max(int(np.ceil(1.5 * remaining)), 1024)

        m1_try = rng.uniform(m1_grid.min(), m1_grid.max(), batch)
        q_try = rng.uniform(q_grid.min(), q_grid.max(), batch)
        theta_try = rng.uniform(0.0, 1.0, batch)
        dL_try = rng.uniform(0.0, dL_max, batch)

        rho0_try = rho0_interp(np.column_stack([m1_try, q_try]))
        A_try = rho0_try * dL_fid * theta_try
        dL_lo = A_try / RHO_MAX
        dL_hi = A_try / RHO_MIN
        keep = (dL_try >= dL_lo) & (dL_try <= dL_hi)

        n_keep = min(remaining, keep.sum())
        if n_keep == 0:
            continue

        out_m1[filled:filled + n_keep] = m1_try[keep][:n_keep]
        out_q[filled:filled + n_keep] = q_try[keep][:n_keep]
        out_theta[filled:filled + n_keep] = theta_try[keep][:n_keep]
        out_dL[filled:filled + n_keep] = dL_try[keep][:n_keep]
        filled += n_keep

    return out_m1, out_q, out_theta, out_dL

m1_u, q_u, Theta_u, dL_u = sample_uniform_reference(N)

# ---------------------------------------------------------------------------
# Corner plot helper
# ---------------------------------------------------------------------------
labels_5d = [
    r"$m_1^\mathrm{det}\ [M_\odot]$",
    r"$q$",
    r"$\Theta$",
    r"$\rho$",
    r"$d_L\ [\mathrm{Gpc}]$",
]

labels_4d = [
    r"$m_1^\mathrm{det}\ [M_\odot]$",
    r"$q$",
    r"$\Theta$",
    r"$d_L\ [\mathrm{Gpc}]$",
]

def make_corner(samples, labels, title, path):
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(11, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(samples[:, i], bins=40, color="steelblue",
                        density=True, edgecolor="none")
                ax.set_yticks([])
            else:
                ax.hexbin(samples[:, j], samples[:, i],
                          gridsize=35, cmap="viridis", mincnt=1)
            if i == ndim - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i != j:
                ax.set_ylabel(labels[i], fontsize=9)
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=7)

    fig.suptitle(title, y=1.01, fontsize=11)
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


make_corner(
    np.column_stack([m1, q, Theta, rho, dL]),
    labels_5d,
    rf"Distance–SNR test  ($N={N}$,  $d_{{L,\mathrm{{fid}}}}={dL_fid}\,\mathrm{{Gpc}}$)",
    FIGURES_DIR / "jacobian_corner.png",
)

make_corner(
    np.column_stack([m1_r, q_r, Theta_r, rho_r, dL_r]),
    labels_5d,
    rf"Distance–SNR test, resampled $\propto d_L/\rho$  ($N={N}$,  $d_{{L,\mathrm{{fid}}}}={dL_fid}\,\mathrm{{Gpc}}$)",
    FIGURES_DIR / "jacobian_corner_resampled.png",
)

make_corner(
    np.column_stack([m1_u, q_u, Theta_u, dL_u]),
    labels_4d,
    rf"Uniform reference in $(m_1^{{\mathrm{{det}}}}, q, \Theta, d_L)$ via rejection sampling  ($N={N}$)",
    FIGURES_DIR / "jacobian_corner_reference_uniform.png",
)

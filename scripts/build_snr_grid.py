"""
Build the SNR interpolant rho_0(m1_det, q) using the aLIGO design sensitivity.

The interpolant is evaluated at fiducial distance dL_fid = 1 Gpc, face-on
orientation (iota=0), and F+=1, Fc=0 (Theta=1).  At runtime the actual SNR
for an event is

    rho = rho_0(m1_det, q) * (dL_fid / dL) * Theta

This script is used to debug the distance Jacobian by providing a clean
reference grid that is independent of the production pipeline.

Usage (from repo root):
    python scripts/build_snr_grid.py
"""

import sys
from pathlib import Path

# Allow imports from src/ without installing the package
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import numpy as np
import jax
import jax.numpy as jnp
import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt
from ripplegw import ms_to_Mc_eta
from ripplegw.waveforms import IMRPhenomD
from tqdm import trange

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FMIN    = 20.0    # Hz
FMAX    = 2048.0  # Hz
DELTAF  = 0.25    # Hz
F_REF   = 20.0    # Hz  (ripple reference frequency)
DL_FID  = 1.0     # Gpc (fiducial luminosity distance)
DL_FID_MPC = DL_FID * 1e3  # Mpc

# Grid resolution (coarser than production; sufficient for debugging)
N_M1 = 200
N_Q  = 200
M1_DET_MIN = 1.0     # Msun
M1_DET_MAX = 1000.0  # Msun
Q_MIN = 0.05
Q_MAX = 1.0

# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------

def make_frequency_grid():
    """Return JAX frequency array and scalar deltaf (Hz)."""
    fs = np.arange(FMIN, FMAX + DELTAF, DELTAF)
    return jnp.array(fs, dtype=jnp.float64), float(fs[1] - fs[0])


def get_aligo_design_psd(fs_np: np.ndarray) -> np.ndarray:
    """
    Evaluate the aLIGO design sensitivity PSD (P1200087) on the given
    uniformly-spaced frequency array fs_np [Hz].

    Returns the one-sided PSD in units of strain^2 / Hz.
    """
    deltaF = float(fs_np[1] - fs_np[0])
    n = len(fs_np)
    psd_series = lal.CreateREAL8FrequencySeries(
        "", lal.LIGOTimeGPS(0), float(fs_np[0]), deltaF,
        lal.DimensionlessUnit, n
    )
    lalsim.SimNoisePSDaLIGODesignSensitivityP1200087(psd_series, FMIN)
    psd = np.array(psd_series.data.data)
    # Zero-out sub-fmin bins (LAL fills them with large values)
    psd[fs_np < FMIN] = np.inf
    return psd


# ---------------------------------------------------------------------------
# SNR computation
# ---------------------------------------------------------------------------

def compute_optimal_snr(hp, psd, df):
    """
    Compute the optimal matched-filter SNR from h+ alone.

    rho = sqrt( 4 * df * sum( |h+(f)|^2 / S(f) ) )

    Using only h+ is consistent with F+=1, Fc=0, iota=0 (Theta=1).
    """
    psd_safe = jnp.where(psd > 0.0, psd, jnp.inf)
    return jnp.sqrt(4.0 * df * jnp.sum(jnp.abs(hp) ** 2 / psd_safe))


def snr_for_params(m1_det, q, fs, psd, df):
    """
    SNR for a single (m1_det, q) grid point at dL=DL_FID_MPC, iota=0.
    """
    m2_det = m1_det * q
    Mc, eta = ms_to_Mc_eta(jnp.array([m1_det, m2_det], dtype=jnp.float64))
    eta = jnp.clip(eta, 0.0, 0.249)
    # theta = [Mc, eta, chi1z, chi2z, dL_Mpc, tc, psi, iota]
    theta = jnp.array(
        [Mc, eta, 0.0, 0.0, DL_FID_MPC, 0.0, 0.0, 0.0], dtype=jnp.float64
    )
    hp, _hc = IMRPhenomD.gen_IMRPhenomD_hphc(fs, theta, F_REF)
    return compute_optimal_snr(hp, psd, df)


def build_snr_grid(m1_grid, q_grid, fs, psd, df, batch_size=400):
    """
    Compute rho_0 on the full (m1_det, q) mesh by vmapping over batches.

    Returns snr_grid with shape (len(m1_grid), len(q_grid)).
    """
    M1, Q = jnp.meshgrid(jnp.array(m1_grid), jnp.array(q_grid), indexing="ij")
    m1_flat = M1.flatten()
    q_flat  = Q.flatten()
    n_total = len(m1_flat)

    snr_out = np.zeros(n_total)

    # vmap-compatible single-event function
    def _snr_single(m1, q):
        return snr_for_params(m1, q, fs, psd, df)

    snr_batch_fn = jax.jit(jax.vmap(_snr_single))

    n_batches = int(np.ceil(n_total / batch_size))
    for i in trange(n_batches, desc="SNR grid batches"):
        start = i * batch_size
        stop  = min(start + batch_size, n_total)
        snr_out[start:stop] = np.array(
            snr_batch_fn(m1_flat[start:stop], q_flat[start:stop])
        )

    return snr_out.reshape(len(m1_grid), len(q_grid))


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_asd_and_template(fs_np, psd, figures_dir):
    """
    Log-log plot of the aLIGO design ASD overlaid with |h+(f)| for a
    reference 30+30 Msun BBH at dL=440 Mpc.
    """
    DL_REF_MPC = 440.0  # Mpc
    fs_jnp = jnp.array(fs_np)

    Mc, eta = ms_to_Mc_eta(jnp.array([30.0, 30.0], dtype=jnp.float64))
    eta = jnp.clip(eta, 0.0, 0.249)
    theta = jnp.array(
        [Mc, eta, 0.0, 0.0, DL_FID_MPC, 0.0, 0.0, 0.0], dtype=jnp.float64
    )
    hp, _hc = IMRPhenomD.gen_IMRPhenomD_hphc(fs_jnp, theta, F_REF)
    hp_np = np.abs(np.array(hp)) * (DL_FID_MPC / DL_REF_MPC)

    psd_jnp = jnp.array(psd)
    snr_ref = float(compute_optimal_snr(hp, psd_jnp, float(fs_np[1] - fs_np[0])) * (DL_FID_MPC / DL_REF_MPC))

    asd = np.sqrt(np.where(psd > 0, psd, np.nan))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(fs_np, asd, color="steelblue", lw=1.5, label="aLIGO design ASD $\\sqrt{S(f)}$")
    ax.loglog(fs_np, hp_np, color="darkorange", lw=1.2, alpha=0.8,
              label=r"$|h_+(f)|$  ($m_1{=}30\,M_\odot,\,q{=}1,\,d_L{=}440\,\mathrm{Mpc}$)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Strain $[\mathrm{Hz}^{-1/2}]$ / $[\mathrm{Hz}^{-1}]$")
    ax.set_xlim(FMIN, FMAX)
    ax.legend(fontsize=9)
    ax.set_title(f"aLIGO design sensitivity vs signal template  ($\\rho={snr_ref:.1f}$)")
    fig.tight_layout()
    out = figures_dir / "asd_vs_template.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_snr_vs_m1(m1_grid, q_grid, snr_grid, figures_dir):
    """1-D plot of rho_0 vs m1_det at q=1."""
    q_idx = np.argmin(np.abs(q_grid - 1.0))
    snr_q1 = snr_grid[:, q_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(m1_grid, snr_q1, color="steelblue", lw=1.5)
    ax.set_xlabel(r"$m_1^\mathrm{det}$ [$M_\odot$]")
    ax.set_ylabel(r"$\rho_0(m_1^\mathrm{det},\,q{=}1)$")
    ax.set_title(f"SNR at $d_L={DL_FID}$ Gpc, $q=1$, face-on, $F_+=1$, $F_\\times=0$")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    out = figures_dir / "snr_vs_m1.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_snr_grid(m1_grid, q_grid, snr_grid, figures_dir):
    """2-D colour map of rho_0(m1_det, q)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    pcm = ax.pcolormesh(
        q_grid, m1_grid, snr_grid,
        shading="auto", cmap="viridis",
        vmin=0, vmax=np.nanpercentile(snr_grid, 99),
    )
    plt.colorbar(pcm, ax=ax, label=r"$\rho_0(m_1^\mathrm{det}, q)$")
    ax.set_xlabel("Mass ratio $q$")
    ax.set_ylabel(r"$m_1^\mathrm{det}$ [$M_\odot$]")
    ax.set_yscale("log")
    ax.set_title(f"SNR grid at $d_L={DL_FID}$ Gpc, face-on, $F_+=1$, $F_\\times=0$")
    fig.tight_layout()
    out = figures_dir / "snr_grid.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    figures_dir = _repo_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Building frequency grid and aLIGO design PSD ...")
    fs, df = make_frequency_grid()
    fs_np  = np.array(fs)
    psd_np = get_aligo_design_psd(fs_np)
    psd    = jnp.array(psd_np)

    print("Plotting ASD + template ...")
    plot_asd_and_template(fs_np, psd_np, figures_dir)

    print(f"Building SNR grid ({N_M1} × {N_Q}) ...")
    m1_grid = np.logspace(np.log10(M1_DET_MIN), np.log10(M1_DET_MAX), N_M1)
    q_grid  = np.linspace(Q_MIN, Q_MAX, N_Q)
    snr_grid = build_snr_grid(m1_grid, q_grid, fs, psd, df)

    out_npz = _repo_root / "snr_grid_aligo_design.npz"
    np.savez(out_npz, m1_grid=m1_grid, q_grid=q_grid,
             snr_grid=snr_grid, dL_fid=DL_FID)
    print(f"Saved SNR grid → {out_npz}")
    print(f"  shape: {snr_grid.shape},  "
          f"min={snr_grid.min():.1f},  max={snr_grid.max():.1f}")

    print("Plotting SNR grid ...")
    plot_snr_grid(m1_grid, q_grid, snr_grid, figures_dir)

    print("Plotting SNR vs m1 (q=1) ...")
    plot_snr_vs_m1(m1_grid, q_grid, snr_grid, figures_dir)

    print("Done.")

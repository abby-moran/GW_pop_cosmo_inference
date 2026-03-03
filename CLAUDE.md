# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian hierarchical inference of gravitational wave (GW) population parameters and cosmological parameters (Hubble constant `h`, matter density `Om`, dark energy EoS `w`) from compact binary merger events. The pipeline generates synthetic GW observations, computes SNRs via Fisher matrices/LAL, and runs MCMC to recover population hyperparameters including PISN mass function features.

## Installation

Use `uv` with the `pyproject.toml` at the repo root:

```bash
uv sync                  # creates .venv with all dependencies
source .venv/bin/activate
```

`jimFisher` (Fisher matrix sampling, used in `fisher_snrs.py`) is not on PyPI; install separately:
```bash
uv pip install git+https://github.com/jacobgolomb/jimFisher
```

All other dependencies including `lalsuite` are on PyPI and installed by `uv sync`.

## Running the Code

Scripts must be run from the `src/` directory (or add `src/` to `sys.path`) because imports are relative (e.g., `from utils import ...`).

**Data generation:**
```bash
cd src/
python gen_inj.py        # generate synthetic injections
python gen_SNRgrid.py    # precompute SNR grids (optional speedup)
```

**Inference (full pipeline):**
```bash
cd reproduce/
python run_inference_evolution.py   # runs MCMC, outputs test_cluster.nc (arviz netCDF)
```
This reads `pe_samples.h5` and `selection_samples.h5` from the working directory. Mock data lives in `src/data/`.

**Notebooks:**
- `reproduce/Run_Inference.ipynb` — end-to-end inference workflow
- `reproduce/Post_MCMC.ipynb` — posterior analysis and plots
- `src/gen_mock_files.ipynb` — data generation walkthrough

## Architecture

### Core inference model (`src/intensity_models.py`)

The main numpyro model is `pop_cosmo_model()`. It:
1. Samples hyperparameters via `sample_parameters_from_dict(priors)` (bridges prior dict → numpyro)
2. Builds `FlatwCDMCosmology` (precomputed interpolation tables for dC, dL, dVc/dz)
3. Builds `LogDNDMDQDV` population model (PISN mass function + Madau-Dickinson rate evolution)
4. Computes per-event likelihoods by importance-sampling over PE posterior samples
5. Applies selection correction using injection samples, computes effective sample sizes `neff_sel`
6. Samples the overall merger rate `R` via a Gaussian approximation

All model classes use `@dataclass` with precomputed grids in `__post_init__`. The PISN mass function is computed on a 2D (mass, redshift) grid because `mpisn` can evolve with redshift via `mpisndot`.

### Population model class hierarchy

```
LogDNDMDQDV          ← top-level: dN/dm1 dq dV, called in pop_cosmo_model
  ├── LogDNDM        ← 1D mass function with PISN + power-law tail, redshift-dependent
  │     └── LogDNDMPISN  ← PISN component: integrates CO mass function over BH masses
  └── LogDNDV        ← Madau-Dickinson redshift evolution: (1+z)^λ / [1 + ((1+z)/(1+zp))^κ]
```

### Prior files (`reproduce/priors/`)

Priors are parsed by `utils.get_priors_from_file()`. Each line is either:
- `param = float` → fixed value (becomes `numpyro.deterministic`)
- `param = DistName(args...)` → numpyro distribution (becomes `numpyro.sample`)

Distribution names map directly to `numpyro.distributions` (e.g., `TruncatedNormal`, `Uniform`, `Normal`). The active prior for the evolution model is `reproduce/priors/gwtc3_evolution.prior`.

Derived parameters computed inside the model (not sampled directly):
- `kappa = lam + dkappa`
- `fpl = exp(log_fpl)`
- `mbhmax = mpisn + dmbhmax`

### Data format

HDF5 files loaded via `pandas.read_hdf(..., 'samples')`. Key columns:
- PE samples: `mass_1`, `mass_ratio`, `luminosity_distance_Gpc`, `prior_m1d_q_dL` (= pdraw)
- Selection samples: `m1`, `q`, `z`, `pdraw_m1sqz`, `dm1sz_dm1ddl`, `ndraw`

The `pdraw` (draw probability) corrects for non-uniform priors when importance sampling. Selection samples also need `m1d = m1*(1+z)` (detector-frame mass) computed on-the-fly.

### Path management (`src/paths.py`)

`paths.py` exposes `root`, `src`, `data`, `figures`, `output` as `pathlib.Path` objects relative to the repo root. Import it from any module to get correct absolute paths.

## Population Model Reference

The population model is described in Golomb, Isi & Farr (2024), *"Physical Models for the Astrophysical Population of Black Holes: Application to the Bump in the Mass Distribution of Gravitational Wave Sources"*, ApJ 976 (arXiv:2312.03973). The paper fixes Planck 2018 ΛCDM cosmology; this codebase extends it to also infer cosmological parameters (flat w-CDM).

### Mass function: CO core IMF and PISN remnant mapping

The primary BH (first-generation, 1G) mass function is derived by convolving a broken power-law CO core IMF with a stochastic BH remnant mass relation:

```
dN/dm_CO ∝  (m_CO / 20 M☉)^(-a)   if m_CO < 20 M☉
            (m_CO / 20 M☉)^(-b)   if m_CO > 20 M☉
```

Parameters `a` (low-mass slope) and `b` (high-mass slope). The break at 20 M☉ is fixed.

The mean BH mass from a CO core is:

```
μ(m_CO | m_PISN, m_BHmax) =
    m_CO                                          if m_CO < m_PISN
    m_BHmax + (m_CO - (2*m_BHmax - m_PISN))^2
              / [4*(m_PISN - m_BHmax)]            if m_PISN ≤ m_CO < 2*m_BHmax - m_PISN
    0 (not physical / no BH formed)               otherwise
```

Parameters: `mpisn` (M_PISN, onset of PISN suppression) and `mbhmax` (maximum BH mass, sampled via `dmbhmax = mbhmax - mpisn`). The actual BH mass is log-normally scattered around μ with standard deviation `sigma` (in log space).

The 1G BH mass function is the integral over CO masses:

```
dN/dm_BH = ∫ (dN/dm_CO) × p(m_BH | μ(m_CO), σ) dm_CO
```

This integral is precomputed on a grid (1800 CO mass points × BH mass points) in `LogDNDMPISN.__post_init__`.

### High-mass power-law tail (2G / hierarchical mergers)

Above `mbhmax`, the mass function gains a power-law tail:

```
dN/dm += f_pl × (dN/dm)|_{m=mbhmax} × (m / mbhmax)^(-c) × S(m | mbhmax)
```

where `S` is a smooth turn-on at `mbhmax`, `fpl` is the relative amplitude (sampled as `log_fpl`), and `c` is the tail slope. This component represents second-generation or hierarchically-assembled BHs. At low masses, a Planck-taper smooth window turns on at `mbh_min` over a width `delta_m` (fixed at 5.0 and 2.5 M☉ respectively).

### Redshift evolution of m_PISN (`mpisndot`)

The PISN threshold mass evolves with redshift (proxy for metallicity evolution):

```
m_PISN(z) = m_PISN(z=0) + mpisndot × (1 - 1/(1+z))
m_BHmax(z) = m_PISN(z) + dmbhmax         [dmbhmax held fixed]
```

`mpisndot` (sampled from `Uniform(-2, 8)`) has units of M☉. The full 2D PISN grid (mass × redshift) is precomputed over 30 redshift points in `LogDNDM.setup_interp()`.

### Pairing function and mass-ratio distribution

The joint distribution over primary mass m₁, mass ratio q = m₂/m₁, and redshift z is:

```
dN / (dm₁ dq dV dt) ∝ (m₁ + m₂)^β × (dN/dm₁) × (dN/dm₂) × R(z)
```

where `beta` controls pairing preference (β > 0 favors equal-mass; β < 0 favors unequal-mass). The full expression is normalized so that `m₁ × dN/(dm₁ dq dV dt) = 1` at reference values (`mref=30 M☉`, `qref=1`, `zref=0.001`); the overall rate `R` then has units of the differential merger rate at the reference point.

### Merger rate evolution (Madau-Dickinson)

```
R(z) ∝ (1+z)^lam / [1 + ((1+z)/(1+zp))^kappa]
```

Normalized to unity at `zref=0.001`. Parameters: `lam` (low-z slope), `kappa = lam + dkappa` (sets high-z decay; `kappa > lam` ensures a peak), `zp` (peak redshift).

### Hierarchical Bayesian likelihood

For N_det detected events with posterior samples {θ_j^(i)} for event i:

```
log L(Λ) = Σᵢ log[ (1/n_samp) Σⱼ π(θⱼ⁽ⁱ⁾|Λ) / p_draw(θⱼ⁽ⁱ⁾) ]
           - N_det × log[ (1/N_draw) Σₖ π(θₖ^sel|Λ) / p_draw^sel(θₖ^sel) ]
```

- `π(θ|Λ)`: population model evaluated at parameters θ = (m₁_det, q, d_L), including the Jacobian `dV_c/dz × 1/(d(d_L)/dz) × 1/(1+z)²` for the detector-frame → source-frame + volume element transformation
- `p_draw`: prior used when generating PE samples or injections (corrects for non-uniform sampling)
- The second term is the log of the expected detection fraction μ_sel (selection correction)
- The overall rate `R` is sampled via a Gaussian approximation: `R = N_det/μ_sel + sqrt(N_det)/μ_sel × R_unit` where `R_unit ~ N(0,1)`
- `neff_sel = exp(2*log_μ_sel - log_s²)` monitors Monte Carlo error in the selection integral (should be >> N_det)

### JAX usage patterns

- All hot-path functions are decorated with `@jax.jit` or are called within a jitted numpyro model
- Use `jnp` (JAX numpy) for array operations inside models; use `np` only for precomputation in `__post_init__`
- `jnp.where` is used extensively to handle boundary conditions without Python branching (required for JIT)
- 2D bilinear interpolation is done manually (not `jnp.interp`) for the PISN grid because it requires indexing in two dimensions

# GW Population And Cosmology Inference

Bayesian hierarchical inference of gravitational-wave population parameters and cosmological parameters from compact binary merger events. The pipeline generates synthetic GW observations, computes SNRs via Fisher matrices and LAL, and runs MCMC to recover population hyperparameters including PISN mass-function features.

## Installation

Use `uv` with the `pyproject.toml` at the repo root:

```bash
uv sync
source .venv/bin/activate
```

`jimFisher` is not on PyPI and must be installed separately:

```bash
uv pip install git+https://github.com/jacobgolomb/jimFisher
```

All other dependencies, including `lalsuite`, are installed by `uv sync`.

## Running The Code

Scripts in `src/` rely on relative imports, so run them from `src/` or add `src/` to `sys.path`.

### Data Generation

```bash
cd src/
python gen_inj.py
python gen_SNRgrid.py
```

`gen_inj.py` generates synthetic injections. `gen_SNRgrid.py` precomputes SNR grids for faster downstream work.

### Inference

```bash
cd reproduce/
python run_inference_evolution.py
```

This reads `pe_samples.h5` and `selection_samples.h5` from the working directory and writes an ArviZ netCDF output such as `test_cluster.nc`. Mock data lives in `src/data/`.

### Notebooks

- `reproduce/Run_Inference.ipynb` contains the end-to-end inference workflow.
- `reproduce/Post_MCMC.ipynb` contains posterior analysis and plots.
- `src/gen_mock_files.ipynb` walks through mock-data generation.

## Architecture

### Core Inference Model

The main NumPyro model is `pop_cosmo_model()` in `src/intensity_models.py`. It:

1. Samples hyperparameters via `sample_parameters_from_dict(priors)`.
2. Builds `FlatwCDMCosmology` with precomputed interpolation tables for `dC`, `dL`, and `dVc/dz`.
3. Builds the `LogDNDMDQDV` population model combining the PISN mass function and Madau-Dickinson rate evolution.
4. Computes per-event likelihoods by importance-sampling over PE posterior samples.
5. Applies selection correction using injection samples and tracks effective sample sizes with `neff_sel`.
6. Samples the overall merger rate `R` via a Gaussian approximation.

All model classes use `@dataclass` with heavy precomputation in `__post_init__`. The PISN mass function is evaluated on a 2D mass-redshift grid because `mpisn` can evolve with redshift via `mpisndot`.

### Population Model Class Hierarchy

```text
LogDNDMDQDV          <- top-level: dN/dm1 dq dV, called in pop_cosmo_model
  |- LogDNDM         <- 1D mass function with PISN + power-law tail, redshift-dependent
  |    \- LogDNDMPISN <- PISN component: integrates CO mass function over BH masses
  \- LogDNDV         <- Madau-Dickinson redshift evolution: (1+z)^lam / [1 + ((1+z)/(1+zp))^kappa]
```

### Prior Files

Prior files live in `reproduce/priors/` and are parsed by `utils.get_priors_from_file()`. Each line is either:

- `param = float` for a fixed value, which becomes `numpyro.deterministic`.
- `param = DistName(args...)` for a sampled parameter, which becomes `numpyro.sample`.

Distribution names map directly to `numpyro.distributions`, such as `TruncatedNormal`, `Uniform`, and `Normal`. The active prior for the evolution model is `reproduce/priors/gwtc3_evolution.prior`.

Derived parameters computed inside the model include:

- `kappa = lam + dkappa`
- `fpl = exp(log_fpl)`
- `mbhmax = mpisn + dmbhmax`

### Data Format

HDF5 files are loaded via `pandas.read_hdf(..., "samples")`.

- PE samples use `mass_1`, `mass_ratio`, `luminosity_distance_Gpc`, and `prior_m1d_q_dL`.
- Selection samples use `m1`, `q`, `z`, `pdraw_m1sqz`, `dm1sz_dm1ddl`, and `ndraw`.

The `pdraw` columns store draw probabilities used to correct for non-uniform sampling. Selection samples also need `m1d = m1 * (1 + z)` computed on the fly.

### Path Management

`src/paths.py` exposes `root`, `src`, `data`, `figures`, and `output` as `pathlib.Path` objects relative to the repository root.

## Population Model Reference

The population model is described in Golomb, Isi, and Farr (2024), "Physical Models for the Astrophysical Population of Black Holes: Application to the Bump in the Mass Distribution of Gravitational Wave Sources", ApJ 976 ([arXiv:2312.03973](https://arxiv.org/abs/2312.03973)). That paper fixes Planck 2018 $\Lambda$CDM cosmology; this codebase extends the framework to infer flat $w$-CDM cosmological parameters as well.

### Mass Function: CO Core IMF And PISN Remnant Mapping

The primary black-hole mass function for first-generation systems is derived by convolving a broken power-law CO-core IMF with a stochastic remnant-mass relation:

$$
\frac{dN}{dm_{\mathrm{CO}}} \propto
\begin{cases}
\left(\frac{m_{\mathrm{CO}}}{20\,M_\odot}\right)^{-a}, & m_{\mathrm{CO}} < 20\,M_\odot \\
\left(\frac{m_{\mathrm{CO}}}{20\,M_\odot}\right)^{-b}, & m_{\mathrm{CO}} > 20\,M_\odot
\end{cases}
$$

Here $a$ and $b$ are the low-mass and high-mass slopes, and the break at $20\,M_\odot$ is fixed.

The mean BH mass from a CO core is:

$$
\mu(m_{\mathrm{CO}} \mid m_{\mathrm{PISN}}, m_{\mathrm{BHmax}}) =
\begin{cases}
m_{\mathrm{CO}}, & m_{\mathrm{CO}} < m_{\mathrm{PISN}} \\
m_{\mathrm{BHmax}} + \dfrac{\left(m_{\mathrm{CO}} - (2m_{\mathrm{BHmax}} - m_{\mathrm{PISN}})\right)^2}{4(m_{\mathrm{PISN}} - m_{\mathrm{BHmax}})}, & m_{\mathrm{PISN}} \le m_{\mathrm{CO}} < 2m_{\mathrm{BHmax}} - m_{\mathrm{PISN}} \\
0, & \text{otherwise}
\end{cases}
$$

$m_{\mathrm{PISN}}$ sets the onset of PISN suppression and $m_{\mathrm{BHmax}}$ is the maximum BH mass, sampled through $dm_{\mathrm{BHmax}} = m_{\mathrm{BHmax}} - m_{\mathrm{PISN}}$. The realized BH mass is log-normally scattered about $\mu$ with log-space width $\sigma$.

The first-generation BH mass function is then:

$$
\frac{dN}{dm_{\mathrm{BH}}} = \int \frac{dN}{dm_{\mathrm{CO}}}\, p\!\left(m_{\mathrm{BH}} \mid \mu(m_{\mathrm{CO}}), \sigma\right)\, dm_{\mathrm{CO}}
$$

This integral is precomputed on a large CO-mass by BH-mass grid in `LogDNDMPISN.__post_init__`.

### High-Mass Power-Law Tail

Above $m_{\mathrm{BHmax}}$, the mass function gains a power-law tail:

$$
\frac{dN}{dm} \mathrel{+}= f_{\mathrm{pl}} \left.\frac{dN}{dm}\right|_{m = m_{\mathrm{bhmax}}}
\left(\frac{m}{m_{\mathrm{bhmax}}}\right)^{-c} S(m \mid m_{\mathrm{bhmax}})
$$

Here $S$ is a smooth turn-on near $m_{\mathrm{BHmax}}$, $f_{\mathrm{pl}}$ is the relative amplitude, and $c$ is the tail slope. This component represents hierarchical or second-generation black holes. At low masses, a Planck-taper window turns on at $m_{\mathrm{BH,min}}$ over a width $\delta_m$, fixed in the current implementation.

### Redshift Evolution Of `m_PISN`

The PISN threshold mass evolves with redshift as a proxy for metallicity evolution:

$$
m_{\mathrm{PISN}}(z) = m_{\mathrm{PISN}}(z=0) + mpisndot \left(1 - \frac{1}{1+z}\right)
$$

$$
m_{\mathrm{BHmax}}(z) = m_{\mathrm{PISN}}(z) + dmbhmax
$$

$\dot{m}_{\mathrm{PISN}}$ is sampled from `Uniform(-2, 8)` in units of $M_\odot$. The full 2D PISN grid over mass and redshift is precomputed in `LogDNDM.setup_interp()`.

### Pairing Function And Mass-Ratio Distribution

The joint distribution over primary mass $m_1$, mass ratio $q = m_2/m_1$, and redshift $z$ is:

$$
\frac{dN}{dm_1\,dq\,dV\,dt} \propto (m_1 + m_2)^\beta \frac{dN}{dm_1} \frac{dN}{dm_2} R(z)
$$

$\beta$ controls pairing preference: positive values favor equal-mass binaries and negative values favor unequal-mass binaries. The expression is normalized so that

$$
m_1 \frac{dN}{dm_1\,dq\,dV\,dt} = 1
$$

at reference values $m_{\mathrm{ref}} = 30$, $q_{\mathrm{ref}} = 1$, and $z_{\mathrm{ref}} = 0.001$. The overall rate $R$ then has units of the differential merger rate at that reference point.

### Merger-Rate Evolution

The redshift evolution follows a Madau-Dickinson form:

$$
R(z) \propto \frac{(1+z)^\lambda}{1 + \left(\frac{1+z}{1+z_p}\right)^\kappa}
$$

It is normalized to unity at $z_{\mathrm{ref}} = 0.001$. The parameters are $\lambda$, $z_p$, and $\kappa = \lambda + \Delta\kappa$, where $\kappa > \lambda$ produces a turnover at high redshift.

### Hierarchical Bayesian Likelihood

For $N_{\mathrm{det}}$ detected events with posterior samples $\theta_j^{(i)}$ for event $i$,

$$
\log \mathcal{L}(\Lambda) =
\sum_i \log \left[ \frac{1}{n_{\mathrm{samp}}} \sum_j \frac{\pi(\theta_j^{(i)} \mid \Lambda)}{p_{\mathrm{draw}}(\theta_j^{(i)})} \right]
- N_{\mathrm{det}} \log \left[ \frac{1}{N_{\mathrm{draw}}} \sum_k \frac{\pi(\theta_k^{\mathrm{sel}} \mid \Lambda)}{p_{\mathrm{draw}}^{\mathrm{sel}}(\theta_k^{\mathrm{sel}})} \right]
$$

- $\pi(\theta \mid \Lambda)$ is the population model evaluated at detector-frame parameters, including the Jacobian for transforming into source-frame quantities and comoving volume.
- $p_{\mathrm{draw}}$ is the draw distribution used to generate PE samples or injections.
- The second term is the expected detection fraction $\mu_{\mathrm{sel}}$.
- The merger rate is sampled via a Gaussian approximation, $R = N_{\mathrm{det}} / \mu_{\mathrm{sel}} + \sqrt{N_{\mathrm{det}}} / \mu_{\mathrm{sel}} \cdot R_{\mathrm{unit}}$ with $R_{\mathrm{unit}} \sim \mathcal{N}(0, 1)$.
- $neff_{\mathrm{sel}} = \exp(2 \log \mu_{\mathrm{sel}} - \log s^2)$ monitors Monte Carlo noise in the selection integral and should be comfortably larger than $N_{\mathrm{det}}$.

### JAX Usage Patterns

- Hot-path functions are either `@jax.jit`-decorated or called inside a jitted NumPyro model.
- Use `jnp` for array operations inside model code and `np` only for setup and precomputation.
- Prefer `jnp.where` to Python branching in jitted logic.
- The PISN grid uses manual 2D bilinear interpolation instead of `jnp.interp`.

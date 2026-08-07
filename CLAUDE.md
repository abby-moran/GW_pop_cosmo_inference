# CLAUDE.md

This file keeps agent-facing guidance brief. Project overview, science background, installation, and workflow notes live in `README.md`.

## Agent Quick Notes

- Scripts in `src/` use relative imports, so run them from `src/` or add `src/` to `sys.path`.
- The main NumPyro model is `pop_cosmo_model()` in `src/intensity_models.py`.
- Population-model classes use `@dataclass` and do most expensive setup in `__post_init__`.

## Priors

- Prior files live in `reproduce/priors/`.
- Priors are parsed by `utils.get_priors_from_file()`.
- `param = float` becomes a fixed `numpyro.deterministic`.
- `param = DistName(args...)` becomes a `numpyro.sample`.
- The active evolution prior is `reproduce/priors/gwtc3_evolution.prior`.
- Derived parameters inside the model include:
  - `kappa = lam + dkappa`
  - `fpl = exp(log_fpl)`
  - `mbhmax = mpisn + dmbhmax`

## Data Format

- HDF5 samples are loaded with `pandas.read_hdf(..., "samples")`.
- PE samples use columns `mass_1`, `mass_ratio`, `luminosity_distance_Gpc`, `prior_m1d_q_dL`.
- Selection samples use columns `m1`, `q`, `z`, `pdraw_m1sqz`, `dm1sz_dm1ddl`, `ndraw`.
- `pdraw` stores the draw probability used for importance-sampling corrections.
- Selection samples also require `m1d = m1 * (1 + z)` on the fly.

## Paths And JAX

- `src/paths.py` exposes `root`, `src`, `data`, `figures`, and `output` as `pathlib.Path` objects.
- Use `jnp` inside model code and reserve `np` for precomputation.
- Favor `jnp.where` over Python branching in jitted paths.
- The PISN mass-grid interpolation is manual 2D bilinear interpolation rather than `jnp.interp`.
- `intensity_models_fast.pop_cosmo_model` accepts float32 recentering baselines (`loglike_ref`, `log_mu_sel_ref`, from `recentering_baselines()`); they shift the potential by a constant only. See `notes/2026-08-07-float32-recentering.md`.

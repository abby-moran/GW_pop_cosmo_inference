# Complete Workflow Trace of `run_inf.py`

## Overview

`run_inf.py` lives at `reproduce/run_inf.py`. It runs NumPyro NUTS inference on gravitational-wave population + cosmology parameters using detected events and selection/injection samples.

---

## Script Initialization & Configuration

### Environment Setup
```python
ndevice = 1
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
numpyro.set_host_device_count(ndevice)
```

### MCMC Hyperparameters
```python
nmcmc = 1200        # warmup AND sampling steps
nchain = 1
random_seed = 1652819403
```

### Imports
- JAX/NumPyro: `jax`, `jnp`, `numpyro`, `MCMC`, `NUTS`, `init_to_value`
- Astropy: `Planck18`
- Local (via `sys.path.append('../src/')`): `intensity_models`, `utils`, `paths`

---

## Core Helper: `get_pop_params(config_file)`

Parses a `key=value` config file → `(dict of population params, FlatwCDMCosmology)`.

Key params: `h`, `Om`, `w`, `zmax`, `a`, `b`, `c`, `mpisn`, `dmbhmax`, `sigma`, `fpl`, `beta`, `lam`, `dkappa`, `zp`, `mpisndot`, `mbh_min`, `delta_m`.

---

## Main Execution Block

### 1. Load Priors
```python
prior = get_priors_from_file("priors/high_zmax.prior")
```
→ `utils.get_priors_from_file()`: parses each line; float → fixed deterministic, `DistName(args)` → numpyro distribution.

### 2. Load Detected Events
```python
pe_samples = pd.read_hdf('../src/pe_c2_zm55_err.h5', key='samples').iloc[0:2000]
```
Columns used: `mass_1`, `mass_ratio`, `luminosity_distance_Gpc`, `prior_m1d_q_dL`
Shapes after reshape: `m1s, qs, dls, pdraws` → `(nobs, nsamp)`

### 3. Load Selection/Injection Samples
```python
sel = pd.read_hdf('../src/sel_c2_zm55_err.h5', key='true_parameters')
ndraw = sel['ndraw'].iloc[0]   # total injections drawn before SNR cut
```
Columns: `m1d`, `q`, `dl`, `pdraw_sel` → 1D arrays of shape `(nsel,)`

### 4. Initialize from Config
```python
population_parameters, cosmo = get_pop_params('../reproduce/configs/c2_zp5.txt')
init_vals = {k: jnp.array(float(v)) for k, v in population_parameters.items() if k in prior}
```

### 5. Set Up NUTS
```python
kernel = NUTS(intensity_models.pop_cosmo_model, init_strategy=init_to_value(values=init_vals))
mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain, progress_bar=True)
```

### 6. Run Inference
```python
mcmc.run(jax.random.PRNGKey(random_seed),
         m1s, qs, dls, pdraws,                              # detected events (nobs, nsamp)
         m1s_sel, qs_sel, dls_sel, pdraw_sel,               # injections (nsel,)
         ndraw, prior)
```

### 7. Save Results
```python
samples = mcmc.get_samples(group_by_chain=True)
np.savez("o3_c2_zm55_err.npz", **samples)
```

---

## The NumPyro Model: `pop_cosmo_model()`

**Location**: `src/intensity_models.py`

### Signature
```python
def pop_cosmo_model(m1s_det, qs, dls, log_pdraw,
                    m1s_det_sel, qs_sel, dls_sel, pdraw_sel,
                    Ndraw, priors=None)
```

### Step 1 — Sample Hyperparameters
`sample_parameters_from_dict(priors)` loops over prior dict:
- float → `numpyro.deterministic(name, value)`
- distribution → `numpyro.sample(name, dist)`

Derived parameters computed immediately after:
- `kappa = lam + dkappa`
- `fpl = exp(log_fpl)`
- `mbhmax = mpisn + dmbhmax`

### Step 2 — Build Models
```python
cosmo = FlatwCDMCosmology(h, Om, w, zmax=zmax)
log_dN = build_population_model(sample)   # returns LogDNDMDQDV
```

### Step 3 — Detected-Event Likelihood
```python
zs   = cosmo.z_of_dL(dls)          # (nobs, nsamp)
m1s  = m1s_det / (1 + zs)

log_wts = (log_dN(m1s, qs, zs)     # population density
           - log_pdraw              # importance correction
           - 2*log1p(zs)           # mass+time Jacobian
           - log(cosmo.ddL_dz(zs)) # dL/dz Jacobian
           + log(cosmo.dVCdz(zs))) # comoving volume element

log_like = logsumexp(log_wts, axis=1) - log(nsamp)  # (nobs,)
numpyro.factor('loglike', sum(log_like))
```

### Step 4 — Selection Effect Correction
```python
zs_sel = cosmo.z_of_dL(dls_sel)
m1s_sel_src = m1s_det_sel / (1 + zs_sel)

log_sel_wts = log_dN(...) - log_pdraw_sel - 2*log1p(zs_sel) + log(dVCdz) - log(ddL_dz)
log_mu_sel  = logsumexp(log_sel_wts) - log(Ndraw)   # expected detection rate

numpyro.factor('selfactor', -nobs * log_mu_sel)
```

### Step 5 — Variance / Effective Sample Size Penalties
- `neff` per event: must be ≥ `nobs`
- `neff_sel` selection: must be ≥ `4*nobs`
- Smooth penalties via `log_smooth_neff_boundary()` to keep gradients finite

### Step 6 — Rate & Derived Quantities
```python
R = numpyro.deterministic('R', nobs/mu_sel + sqrt(nobs)/mu_sel * R_unit)
```
Plus grids `mdNdmdVdt_fixed_qz`, `dNdqdVdt_fixed_mz`, `dNdVdt_fixed_mq`, `hz` saved as deterministics.

---

## Key Supporting Classes

### `FlatwCDMCosmology` (intensity_models.py, dataclass)
- Precomputes log-spaced `z` grid, comoving distance via cumtrapz of `1/E(z)`, and derived quantities
- Methods: `E(z)`, `dC(z)`, `dL(z)`, `dVCdz(z)`, `ddL_dz(z)`, `z_of_dL(dL)`

### `LogDNDMDQDV`
- Joint `dN/d(m1)d(q)d(V)d(t)` model
- Composed of `LogDNDM` (mass function) × `LogDNDV` (redshift evolution)

### `LogDNDM`
- PISN contribution: numerical integral via `LogDNDMPISN`, stored as 2D grid interpolated over (m_BH, z)
- Power-law tail above `mbhmax(z)`: `fpl * (m/mbhmax)^{-c}`
- `mpisn(z) = mpisn + mpisndot*(1 - 1/(1+z))`

### `LogDNDV`
- Madau-Dickinson: `(1+z)^λ / [1 + ((1+z)/(1+z_p))^κ]`

---

## File I/O Summary

| Direction | File | Format | Purpose |
|-----------|------|--------|---------|
| Input | `reproduce/configs/c2_zp5.txt` | Text | True population params / init values |
| Input | `priors/high_zmax.prior` | Text | MCMC priors |
| Input | `src/pe_c2_zm55_err.h5` | HDF5 | PE samples on detected events |
| Input | `src/sel_c2_zm55_err.h5` | HDF5 | Injection samples for selection |
| Output | `o3_c2_zm55_err.npz` | NumPy archive | MCMC posterior samples |

---

## Data Flow

```
config.txt  →  get_pop_params()  →  init_vals + cosmo
prior file  →  get_priors_from_file()  →  prior dict
pe_*.h5     →  m1s, qs, dls, pdraws  (nobs, nsamp)
sel_*.h5    →  m1s_sel, qs_sel, dls_sel, pdraw_sel, Ndraw  (nsel,)
                         ↓
             NUTS on pop_cosmo_model()
               ├─ sample priors
               ├─ build cosmology + population model
               ├─ detected-event likelihood
               ├─ selection correction
               └─ rate + derived quantities
                         ↓
              o3_c2_zm55_err.npz  (posterior samples)
```

---

## No CLI Arguments

All paths and MCMC settings are hardcoded in the `__main__` block. Edit them directly to change data paths, number of steps, or output filename.

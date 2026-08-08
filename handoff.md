# Handoff — 2-D mass table + free-cosmo / free-mpisndot inference tests

*2026-08-07 → 2026-08-08. Branch `cursor/optimize-jax-population-model` (pushed through commit `7fc7a6e`).*

## What this conversation did

1. **Implemented a 2-D mass-function table** so sampling `mpisndot` no longer falls back to the slow per-sample direct evaluation (`notes/2026-08-07-mass-table-2d.md`, commit `7fc7a6e`). Work was done in an isolated git worktree on `local/mass-table-2d`, then fast-forward merged and pushed.
2. **Launched three production-scale mock-O5 inference tests** on Slurm (same PE/selection data as `endO5_val2`, 9000 events × 4000 PE samples, 1800 warmup + 1800 sample × 2 chains, 2×H100) to isolate freeing cosmology vs freeing `mpisndot`.
3. **Made corner plots** for those runs in the same `corner.corner` style as `endO5_val2`.

## 2-D mass-function table (code)

**Problem.** With `mpisndot` pinned to 0 the mass function is z-independent and is tabulated on an 8192-node log-m grid (~4× faster than direct). Sampling `mpisndot` used to disable that path → ~68 ms/grad and ~20 GiB at production scale.

**Fix** (`src/intensity_models_fast.py`). When `_z_dependent`, build `f_tab[i,j] = log_dndm(m_j, z_i)` on the PISN grid’s own 30 log1p-uniform z nodes and look up bilinearly (`_gather_lerp2d`) with the same linear-in-z cell weights as the direct PISN interp. Default `tabulate_mass_function=None` now means *enabled always* (1-D or 2-D as appropriate).

**Selection stays on the direct path** in the 2-D case: the table’s O(3e-3) z-lerp bias in `log_mu_sel` is multiplied by `nobs` in the selection factor. Per-event likelihoods have no such amplification; selection is ~5% of the points.

**Bench (same GPU A/B, `mpisndot` free):** gradient 68.5 → 38.0 ms/leapfrog; peak GPU memory 20.4 → 9.4 GiB.

**Tests.** `scripts/test_fast_equivalence.py` gained `test_tabulated_path_zdep`. Also fixed a pre-existing test-6 failure on the parent branch: after the `Omh2` default, `Omh2` is an edge parameter like `h` and belongs in the AD-vs-FD edge set.

**Benchmark flag.** `scripts/bench_model.py --no_tab` forces the direct path for A/B timing.

## Three inference tests

All reuse `endO5_val2` data via symlinks. Truth is `scripts/pop_configs/mock_O5_noevo.txt` (`w = -1`, `mpisndot = 0`, `h = 0.674`, `Om = 0.315`).

| run | prior | free among {h, Omh2/Om, w, mpisndot} | Slurm job | wall | samples |
|---|---|---|---|---|---|
| `endO5_fullcosmo_evo` | `runs/priors/gwtc5_fullcosmo_evo.prior` | h, Omh2, w, **mpisndot** | 6783672 | 2h04m | `O5_fullcosmo_evo.nc` |
| `endO5_fullcosmo` | `runs/priors/gwtc5_fullcosmo.prior` | h, Omh2, w (`mpisndot=0`) | 6784160 | 1h22m | `O5_fullcosmo.nc` |
| `endO5_evo` | `runs/priors/gwtc5_evo.prior` | **mpisndot** (h/Om/w pinned to truth) | 6784161 | 59m | `O5_evo.nc` |

Configs: `scripts/run_configs/mock_O5_{fullcosmo_evo,fullcosmo,evo}.ini`.  
Launchers: `scripts/call_inf_{fullcosmo_evo,fullcosmo,evo}.sh`.  
Logs: `scripts/{fullcosmo_evo_6783672,fullcosmo_6784160,evo_inf_6784161}.log`.

Reference for comparison: `endO5_val2` (h free, Om fixed in that run’s prior era / effectively cosmo mostly pinned, `w`/`mpisndot` fixed) — healthy recovery, 0 divergences.

### Recovery summary

| run | truth in 95% (free params) | divergences | verdict |
|---|---|---|---|
| **fullcosmo** (cosmo free, `mpisndot=0`) | **18/18** (14/18 in 68%) | 13 | **recovers truth** |
| **evo** (cosmo fixed, `mpisndot` free) | **4/16** | 14 | **fails** — `mpisndot` → prior floor −2 |
| **fullcosmo_evo** (both free) | **4/19** | 43 | **fails** — same `mpisndot` floor mode |
| val2 (ref) | 14/15 | 0 | recovers (aside from mild `mp_low` tension) |

**Bottom line:** pathology tracks freeing `mpisndot`, not freeing cosmology. Full cosmology with `mpisndot` pinned is healthy and comparable to val2.

### Key posterior numbers

**fullcosmo** (healthy):

| param | truth | median [16%, 84%] |
|---|---|---|
| h | 0.674 | 0.771 [0.622, 0.921] |
| Omh2 | 0.143 | 0.120 [0.093, 0.148] |
| w | −1.0 | −1.17 [−1.41, −0.93] |
| mpisn | 33.29 | 31.55 [29.89, 33.49] |
| dmbhmax | 3.44 | 4.06 [2.91, 5.18] |

**evo** (failed — boundary mode):

| param | truth | median [16%, 84%] |
|---|---|---|
| mpisndot | 0 | **−1.99 [−2.00, −1.98]** (floor) |
| mpisn | 33.29 | 38.66 [38.48, 39.26] |
| dmbhmax | 3.44 | **0.51 [0.50, 0.52]** (floor) |
| a, b, sigma | … | pinned to prior edges |

**fullcosmo_evo** (failed — same mode): `h` → 0.41 (floor 0.4), `mpisndot` → −1.99, `dmbhmax` → 0.52, mass params collapsed; `w` and `Omh2` alone looked ok but derived `Om = Omh2/h²` was nonsense (~0.79).

n_eff diagnostics were **fine in all three** (not an MC-accuracy issue): e.g. fullcosmo_evo had `min_neff ≈ 7.4`, `mc_var_loglike ≈ 3.68` (budget 5, 0% of draws over), `neff_sel ≈ 1.29×10⁵` (hinge at 3.6×10⁴, 0% under). Convergence looked fine (r̂ ≈ 1, ESS hundreds) — a real posterior mode against the prior walls, not a failed warmup.

## Corner plots

Same recipe as `endO5_val` / `endO5_val2` (`corner.corner`, red truth crosshairs, 16/50/84% quantiles, titles):

- `runs/endO5_fullcosmo/O5_fullcosmo_corner.png`
- `runs/endO5_evo/O5_evo_corner.png`
- `runs/endO5_fullcosmo_evo/O5_fullcosmo_evo_corner.png`
- (reference) `runs/endO5_val2/O5_val2_corner.png`

## RESOLVED 2026-08-08: the `mpisndot`-free pathology was a real bug — now fixed

Commit `555e833` identified and fixed it: the 2-D table evaluated the event
samples on the table but the selection integral on the direct path.  The
R-marginalized likelihood is the ratio `prod_i lambda(x_i) / (int lambda
p_det)^nobs`, which is only a probability model when both sides share the
same density; the split left the numerator's parameter-dependent z-lerp bias
uncancelled (+125 nats at the wall corner) and the sampler climbed it.
`tabulate_selection` now defaults to consistent.  Rerun `endO5_evo2`: truth
in 95% 4/15 → 15/15, divergences 14 → 0, max r̂ 1.83 → 1.02.  Details:
`notes/2026-08-08-tabulated-selection-consistency.md`; independent review in
`critique.md` concurred.

Follow-up validation (this session): tests 7+8 re-verified; `endO5_evo2`
importance-reweighted from n_z=30 to n_z=60/120 with ESS 99.8% and median
shifts ≤ 0.004σ (`scripts/diag_nz_reweight.py`) — n_z=30 is fine
posterior-wide.  A both-free rerun (`fullcosmo_evo2`, Slurm 6786045) was in
flight at the time of writing; reweight/validate it the same way when done.

Remaining notes:

1. The old evo chains' MCMC convergence was **not** clean (max r̂ 1.83), even
   though the MC/IS n_eff diagnostics were healthy — the earlier claim of
   r̂ ≈ 1 in this file's tables applied to the both-free run whose two chains
   found the same wall mode.
2. Run priors/configs/launchers for these tests are now committed (see
   `scripts/run_configs/mock_O5_*.ini`, `scripts/call_inf_*.sh`).

## Key file map

| path | role |
|---|---|
| `src/intensity_models_fast.py` | 1-D / 2-D tabulated mass function + scatter-free cosmo VJP |
| `notes/2026-08-07-mass-table-2d.md` | design + measured speedups + FD caveats |
| `notes/2026-08-07-optimization-changelog.md` | series changelog (updated for 2-D table) |
| `scripts/test_fast_equivalence.py` | equivalence + AD-vs-FD (incl. `test_tabulated_path_zdep`) |
| `scripts/bench_model.py` | `--mpisndot_free`, `--no_tab`, `--cosmo_free`, `--omh2` |
| `runs/priors/gwtc5_{fullcosmo,evo,fullcosmo_evo}.prior` | priors for the three tests |
| `runs/endO5_{fullcosmo,evo,fullcosmo_evo}/` | outputs (`.nc` + `_corner.png`) |

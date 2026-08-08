# Change log: model optimization pass (August 2026)

*Everything that changed since we started optimizing the population model,
in one place.  Companion notes:*

- `2026-08-07-neff-penalty-redesign.md` — the new Monte-Carlo accuracy guard
- `2026-08-07-jax-performance-improvements-explained.md` — why each speedup works
- `2026-08-07-profiling-jax-numpyro-guide.md` — how to profile / avoid pitfalls
- `2026-08-07-mass-table-2d.md` — 2-D mass table when mpisndot is sampled
- `2026-08-06-join-point-machinery-removed.md`, `2026-08-06-mco-floor-configurable.md`
  — related archaeology in the same mass function

## Headline numbers

Measured at production scale (9000 events x 4000 PE samples + 1.7M selection
samples), float32.  "Gradient" is one `value_and_grad` of the potential — what
NUTS pays per leapfrog step:

| | original (`intensity_models`) | final (`intensity_models_fast`) | |
|---|---|---|---|
| potential (forward) | 123 ms | ~3 ms | ~40x |
| gradient (per leapfrog) | 477 ms | 18 ms (A6000) / 9.5 ms (H100) | ~26-50x |
| peak GPU memory | 30 GiB | ~6 GiB | 5x |

End-to-end (validated 2026-08-07 on mock O5, run `endO5_val`): 1800+1800
NUTS samples at max_tree_depth=7 take **75 min/chain on one H100** (the
original would have needed ~60 h/chain).  Zero divergences, r-hat <= 1.006,
bulk ESS 190-570 across all 15 sampled parameters.

## The module

All work lives in `src/intensity_models_fast.py`, a drop-in replacement for
`src/intensity_models.py` with the same public API.  The original file is
**unchanged** and kept as the reference implementation for the equivalence
tests.  `scripts/run_inf.py` and `scripts/reweight_res.py` now import the
fast module; revert by switching the import back.

## Performance changes (numerics preserved)

Verified equivalent to the original to float32 roundoff by
`scripts/test_fast_equivalence.py` (potential to ~1e-6 relative, all gradient
components to ~3e-4; the fast module is *more* accurate than the original
against float64 references).

1. Replicated lookup tables (32 copies) to kill scatter contention in the
   backward pass of table gathers.
2. Closed-form grid indices on log-/log1p-uniform axes instead of
   `jnp.interp`'s binary search.
3. One fused cosmology table lookup (z, log-Jacobian) instead of three
   separate interpolations; tables are compile-time constants when Om, w are
   fixed.  Side effect: fixes the dVC/dz error (up to 2x at z < 0.005) that
   linear interpolation of the original's coarse first grid cell produced.
4. Max-subtracted linear trapezoid for the PISN mco integral (one `exp`
   instead of logaddexp/logsumexp chains), mco axis moved last for contiguous
   reduction.
5. `mpisndot == 0` detected statically -> PISN grid built with 1 z-slice
   instead of 30, interpolation collapses to 1-D.
6. Single-pass fused logsumexp + n_eff over the (nobs, nsamp) weight array.
7. **Tabulated mass function** (`tabulate_mass_function`, default on): the
   log dN/dm is evaluated once per likelihood call on an 8192-node log-m
   grid, and every per-sample mass evaluation becomes a table lerp.  ~2x on
   top of everything else when mpisndot is pinned to 0 (1-D table).  When
   mpisndot is *sampled* the table gains a z axis on the PISN grid's own 30
   z nodes and the lookup becomes bilinear.  The selection set uses the same
   table as the event samples -- it briefly did not, which was a bug; see the
   correction at the end of this file.  Measured for the mpisndot-free case at
   production scale: gradient 68.5 -> 38.0 ms, peak memory 20.4 -> 9.4 GiB.
   Details: `2026-08-07-mass-table-2d.md`.
8. Constant-in-the-sampler data quantities (logs of masses, ratios, pdraw)
   hoisted out of the per-step computation.
9. **Default cosmology prior samples `Omh2 = Om*h^2`** (not `Om`); the model
   derives `Om = Omh2/h^2` as a deterministic.  Better conditioned against `h`
   when matter density is free.  Active prior: `runs/priors/gwtc5_cosmo.prior`.
   Priors that still set `Om` directly keep the old behaviour.

## Bug fixes

1. **`@jax.jit` on `get_deterministic_parameters` silently dropped the
   derived sites** (`kappa`, `mbhmax`, `fpl`, `flow`) from the MCMC output:
   `numpyro.deterministic` inside a jitted function records its site only on
   the trace that compiles; on cache hits the sites vanish.  Decorator
   removed.
2. **`nan_to_num` fixed values but not gradients.**  An event whose weights
   all underflow produced `neff = exp(2*(-inf) - (-inf))` = NaN;
   `nan_to_num` made the value finite but the *gradient* stayed NaN, which
   NUTS treats as a divergence.  Replaced with a `where`-guarded logsumexp
   (`_logsumexp_and_neff`) and a finite floor (`_LOG_ZERO_FLOOR = -1e6`).
   The original could not even initialize when one event was out of support;
   the fast module returns a finite potential and clean gradients.
3. `nan_to_num(log_like_per_event, nan=0)` silently treated a NaN event as
   likelihood 1.  Now impossible by construction (see 2).
4. `safe_log(x, eps=1e-300)` was a no-op in float32 (1e-300 underflows to
   exactly 0).  Epsilon now defaults to the smallest normal of the dtype.
5. `mco_min` in a prior/pop-config file was silently ignored —
   `build_population_model` never passed it through.  Fixed (and `mco_floor`
   got the same treatment later, see its note).
6. `selfactor` floor asymmetry: dead selection integrals now floor to
   `_LOG_ZERO_FLOOR` symmetrically with the event side.
7. Two portability fixes in the fast module itself: int32/int64 dtype clash
   in the table-replica offset under `jax_enable_x64`, and mass-table bounds
   computed from numpy inputs *before* they become tracers (with a fixed
   fallback), avoiding a `ConcretizationTypeError`.

## Model changes (deliberate, defaults changed)

These change the sampled posterior (slightly).  Each has a switch to
reproduce the original exactly.

1. **`smooth_tail_edge=True`** (default): the power-law tail is no longer
   hard-zeroed below m = mbhmax, making the density continuous there.  With
   the hard edge, the AD gradients of the potential w.r.t. `h`, `mpisn`,
   `dmbhmax` disagree with finite differences by 10-30% at typical points
   (AD differentiates the expression, which cannot see samples crossing a
   step).  Set `smooth_tail_edge=False` for the old behavior.  Applied
   consistently in `reweight_res.py` (population generation) and
   `run_inf.py` (inference).
2. **`neff_penalty="mc_variance"`** (default): the Monte-Carlo accuracy
   guard now penalizes the total MC variance of the log likelihood,
   sum_i 1/n_eff_i, above `mc_variance_budget=5.0`, with a smooth
   (-softplus) boundary; the selection guard (n_eff_sel >= 4*nobs) keeps its
   threshold but also gets the smooth boundary.  The old guard
   (min-over-events n_eff >= nobs, kinked boundaries everywhere) is
   `neff_penalty="min_neff"`.  See the dedicated note for the full story.
3. `store_per_event=False` (default): the per-event log-likelihood and n_eff
   arrays (2 x 9000 doubles per posterior sample, hundreds of MB per run)
   are only recorded when asked for.  `min_neff` and `mc_var_loglike`
   scalars are always recorded.

To reproduce the original model exactly: `smooth_tail_edge=False,
neff_penalty="min_neff"` (or import `intensity_models`).

## Removed / cleaned up

- `join_point_terms` / `log_mix_at_join`: dead code from the pre-July-2026
  height-matched tail parameterization; deleted (see its note — no number
  changes, XLA had already eliminated it).
- Nested `@jax.jit` decorators on small helper functions: useless under the
  outer jit and the source of bug 1.

## New scripts

| script | purpose |
|---|---|
| `scripts/test_fast_equivalence.py` | numerical equivalence vs original, AD-vs-FD gradient checks, dead-event safety, tabulated-selection consistency |
| `scripts/test_mco_floor.py` | mco_floor plumbed end to end |
| `scripts/bench_model.py` | gradient-of-potential benchmark + `--diagnose` site tracer |
| `scripts/bench_breakdown.py` | per-term cost bisection of the potential |
| `scripts/bench_neff_penalty.py` | A/B/C of the three neff_penalty modes |
| `scripts/call_inf_val.sh` | slurm script for the end-to-end validation run |

`scripts/gen_inj.py` also gained a config-file `num_loops` (short validation
runs) and a `$HOME`-anchored ceph path.

## End-to-end validation (2026-08-07)

Full pipeline from scratch on mock O5 (`run_configs/mock_O5_noevo_val.ini`,
truth `pop_configs/mock_O5_noevo.txt`): gen_inj (12.6M injections) ->
reweight_res (1.75M detected selection samples, 10000 events x 4000 PE) ->
run_inf (2 chains x 1800+1800).  Zero divergences; the mc_variance guard
stayed inactive over the whole posterior (sum 1/n_eff in 3.45-3.83 vs budget
5) while min_neff ~ 10 shows the *old* default (>= nobs = 9000) would have
been permanently active.  Cosmology + PISN parameters (h, mpisn, sigma,
dmbhmax) recover their truths within ~1 sigma; some mass/redshift shape
parameters land 2-6 sigma off, consistent with a single mock realization
from a reduced injection pool — flagged for follow-up, not attributed to the
optimization (posterior: `runs/endO5_val/O5_val.nc`).

## Correction (2026-08-08): tabulated selection consistency

The 2-D mass table shipped with the selection set on the *direct* evaluation
while the event samples used the table.  That is not a valid hierarchical
likelihood -- the R-marginalized form is a ratio, so both sides must use the
same density -- and it made every mock run with `mpisndot` free walk onto the
prior walls (`runs/endO5_evo`, `runs/endO5_fullcosmo_evo`).  Fixed by
`tabulate_selection` (default: follow `tabulate_mass_function`).  Full
account, measurements and regression guard:
`2026-08-08-tabulated-selection-consistency.md`.

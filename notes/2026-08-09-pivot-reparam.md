# Pivoted mass scale and log-space sampling: implementation and benchmarks

*2026-08-09.  Branch `local/reparam-pass-1` (off `local/opt-pass-3`).
Implements suggestions 1 and 2 of `notes/model-suggestions.md` and
benchmarks them.  Verdict: the mpisn pivot is a modest, real improvement
(~2x on worst-case grads per effective sample, biggest on mpisndot);
log-space sampling is a clear regression and removing sigma's hard floor
causes mass divergences.  Only the pivot is recommended for production.*

## What was implemented

All opt-in via the prior file; priors sampling the canonical names are
byte-for-byte unchanged.  Resolution happens in
`get_deterministic_parameters` (`src/intensity_models_fast.py`), before the
existing `mbhmax`/`Om` derivations that consume the canonical values.

1. **Pivoted mass scale.**  The model's evolution is
   `mpisn(z) = mpisn + mpisndot * z/(1+z)` with `mpisn` defined at z=0,
   while the data pin the mass scale near the bulk of detections.  A prior
   that samples `mpisn_ref` (with a fixed `zpivot`) makes the model derive

       mpisn = mpisn_ref - mpisndot * zpivot/(1+zpivot)

   For the endO5 mock the detection-weighted mean of x = z/(1+z) over PE
   samples is 0.429, i.e. `zpivot = 0.75`.
   IMPORTANT: tighten the `mpisn_ref` bounds so the derived `mpisn` stays
   inside the old support for every `mpisndot` (for [20, 50] and
   mpisndot in [-2, 8] at zpivot=0.75: `low=23.5, high=49.1`); see
   `runs/priors/gwtc5_fullcosmo_evo_pivot.prior`.

2. **Log-space alternates** `log_h`, `log_mpisn`, `log_mpisn_ref`,
   `log_sigma`, `log_mp_low` (exp'd into the canonical names).
   Implemented and tested, but benchmarks say don't use them (below).

3. **Truth mapping.**  `map_truths_to_prior_coords` (same file) converts
   canonical pop-config truths into the prior's coordinates;
   `scripts/run_inf.py` applies it so `init_to_value` and
   `recentering_baselines` still start at the truth under a
   reparameterized prior.

Correctness: `test_fast_equivalence.py` test 10 traces the model under the
canonical and reparameterized priors at matched points and requires every
likelihood factor and derived deterministic to agree; they are bit-identical.
Full suite (tests 1-10) passes.

## Benchmarks

`scripts/bench_reparam.py`: single-chain NUTS, 150 warmup + 150 samples,
`max_tree_depth=10`, `dense_mass=True`, mpisndot free; figure of merit is
total leapfrog gradients per effective sample of the *canonical*
parameters (recorded as deterministics when reparameterized).

### Synthetic data: unusable for this comparison (a finding in itself)

`make_synthetic_data`'s junk pdraw weights put the whole posterior at
`mc_var_loglike ~ 7`, past the MC-variance budget of 5, so chains sample
*inside* the noise-penalty region where the potential surface is rough.
Results there were seed-dominated: the pivot won at seed 1 (0 divergences,
worst grads/ESS 981 -> 629) and blew up at seed 2 (103/150 divergences with
`mpisn` samples nowhere near any bound).  Diagnosis: divergences correlate
with `mc_var_loglike` in [6.7, 7.5], not with any parameter value.
Geometry benchmarks need data whose MC guard is inert.

### Real endO5_val2 mock (first 2000 events x 4000 samples, half the
selection set, `mc_var ~ 0.75`, zpivot=0.75, seed 2)

| | baseline (A6000) | pivot_tight (H100) | log (H100) |
|---|---|---|---|
| wall, 300 iters | 2767 s | 1638 s | 1606 s |
| mean leapfrogs/iter | 807 | 982 | 1023 (saturated) |
| divergences | 6/150 | 3/150 | 0/150 |
| final step size | 1.4e-2 | 5.6e-3 | 1.3e-3 |
| worst grads/ESS | 11993 (mpisndot) | 5136 (sigma) | 59622 (mpisn) |
| mpisndot ESS | 10.1 | 46.4 | 5.6 |

(Wall times are not comparable across GPUs; grads/ESS is.)

- **Pivot**: 2.3x better worst-case grads/ESS, 4.6x on `mpisndot` itself.
  The bottleneck moves to `sigma` (ESS 29), which sits against its hard
  floor at 0.05 -- a different problem, not created by the pivot.
- **Log-space**: every iteration saturates depth 10, step size collapses
  3x below baseline, ESS craters across the board.  The spectral-siren
  "multiplicative degeneracy" argument for log coordinates is empirically
  wrong here: in log space the dense mass matrix has to relearn scale
  factors it already had, and the transform curves the previously-linear
  directions.
- **Removing sigma's floor** (`log_sigma = Normal(...)`, synthetic seed 1):
  94/150 divergences.  The floor guards a real cliff (narrow PISN peak
  under-resolved by the mass table / MC noise), it is not just a prior
  wall.  Keep `low=0.05`.

## Production recommendation

Use `runs/priors/gwtc5_fullcosmo_evo_pivot.prior` (pivot only, linear
everything else) for the next fullcosmo+evo validation run and compare
mixing against evo3/evo4.  Expected: same per-gradient cost, ~2x fewer
gradients per effective sample, biggest gain on `mpisndot`.  The `sigma`
floor geometry is the next candidate once this lands.

# Is float32 safe? A production-scale audit

*2026-08-07.  Applies to `src/intensity_models_fast.py`.  Harness:
`scripts/test_float32_accuracy.py`.  Companion to
`2026-08-07-jax-performance-improvements-explained.md` ("What we deliberately
did not do: no float64") and `2026-08-07-profiling-jax-numpyro-guide.md`
(pitfall 3, "Respect float32").*

## The gap this closes

The optimization notes assert float32 is safe, resting on
`scripts/test_fast_equivalence.py`.  That suite is sound but its only
*full-potential* checks run at nobs=400 / nsamp=300 / nsel=40k, and it compares
float32 against float32 (fast vs slow), scoring only sub-components against
float64 references.  Float32 error in the potential grows with nobs, so those
checks were ~40x more forgiving than a production run, and no one had measured
the float32-vs-float64 potential or gradient at 9000 x 4000 x 1.7M.

## Method

`scripts/test_float32_accuracy.py` runs two precision legs as subprocesses
(x64 must be set before jax is imported).  Two details make the comparison
meaningful:

- **Same point.**  The float64 leg is forced onto the float32 leg's
  unconstrained `z0`.  Letting each leg compute its own `z0` from `init_to_value`
  puts them ~1e-7 apart, and with |dPE/dz| ~ 1e5 that alone fabricates ~1e-2
  nats of *apparent* difference -- the same size as the real effect.
- **Same inputs.**  All input arrays are rounded to float32 and stored back as
  float64, so the measurement isolates *arithmetic* precision from *input*
  precision.

Three independent measurements: (A) float32 vs float64 potential and gradient;
(B) reduction-order jitter -- permuting the event and selection order leaves the
potential mathematically invariant, so the spread over permutations is the
float32 noise floor with no reference needed (the probe asserts the data really
was permuted, so a silent no-op cannot masquerade as zero jitter); (C) scans over
nobs and over parameter points.

Reference scale: NUTS targets 80% acceptance, so the leapfrog integrator's own
energy error is a few tenths of a nat; numpyro flags a divergence only above
`max_delta_energy` = 1000.

## Result 1: the error is ~1 ulp of the log-likelihood sum

| nobs | `loglike` | 1 ulp | measured error | err/ulp | jitter | jit/ulp |
|---|---|---|---|---|---|---|
| 400 | 6.49e3 | 4.9e-4 | 5.9e-4 | 1.20 | <1 ulp | - |
| 1000 | 1.60e4 | 9.8e-4 | 4.9e-4 | 0.50 | 5.3e-4 | 0.55 |
| 3000 | 4.86e4 | 3.9e-3 | 3.0e-3 | 0.77 | 3.9e-3 | 1.00 |
| 9000 | 1.46e5 | 1.6e-2 | **1.87e-2** | 1.20 | 9.6e-3 | 0.61 |

Both probes agree, and the error sits at 0.5-1.2 ulp of `loglike` at every
scale.  **This is not accumulated roundoff over 36M elements -- it is the
representation limit of a single float32 number of magnitude `loglike`.**
Consequence: no better summation (Kahan, pairwise, wider tree) can improve it.
Only holding that accumulator in float64 can.

Mechanism: the potential is a near-cancellation of two nobs-proportional terms,
`loglike` = +1.46e5 and `selfactor` = -1.27e5 (the latter is literally
`-nobs * log_mu_sel`), summing to -1.87e4.  Each is accurate to ~5e-8
*relative* -- excellent -- but at magnitude 1.4e5 that is ~1e-2 nats *absolute*,
and the cancellation preserves the absolute error while shrinking the value.
So the potential's relative error (1e-6) flatters it; **absolute nats is the
figure of merit.**

Caveat on attribution: the per-term decomposition comes from a separate trace
pass, so it is itself only resolvable to ~1 ulp of 1.46e5 = 1.6e-2 nats.  The
mechanism is solid; the split between `loglike` and `selfactor` is not.

## Result 2: gradients are far better than the potential

At nobs=9000 the gradient *direction* is exact to `1 - cos = 4e-11`, and the
norm ratio is 1.0000063.  Worst per-component relative errors (~1e-4) all land
on components whose gradient is near zero -- e.g. `d/dsigma` = -2.2e-2 at
nobs=3000, where a 1e-3 "relative" error is 2.2e-5 absolute.  The relative
metric is unfair to small components; the direction cosine is the honest
summary.  `d/dmpisn` and `d/ddmbhmax` are systematically the noisiest of the
large components (~2e-5) because they move the mass-function edge through the
8192-node table -- consistent with the `smooth_tail_edge` story, not a new
problem.  Gradient error does **not** grow with nobs (it is table-interpolation
limited, not accumulation limited); only the potential does.

## Result 3: parameter-point dependence is mild, and the truth point is worst

Production scale, three points from the equivalence suite's sweep:

| point | sigma | error | vs energy scale | worst grad rel | 1-cos |
|---|---|---|---|---|---|
| truth | 0.054 | **1.87e-2** | 0.093x | 1.0e-4 | 4.4e-11 |
| sharp | 0.051 | 1.38e-2 | 0.069x | 2.1e-4 | 1.4e-10 |
| broad | 0.35 | 5.8e-3 | 0.029x | 5.6e-6 | 7.4e-14 |

All three sit inside the 0.5-1.2 ulp band.  `broad` is best because a broader
mass function spreads the importance weights, lowering |`loglike`| and hence its
ulp.  Useful corollary: **float32 error tracks |`loglike`|, so the points that
stress precision are the high-likelihood ones -- which is where NUTS spends its
time.**  The truth point is therefore near the worst case, not a lucky one.

## Verdict

**Safe at current scale.**  In order of evidential strength:

1. The `endO5_val` production run (nobs=9000, float32) had zero divergences and
   r-hat <= 1.006 -- direct empirical evidence, not extrapolation.
2. The arithmetic error is 9% of the leapfrog integrator's own error budget and
   1.9e-5 of numpyro's divergence threshold.
3. Gradient direction is exact to 4e-11, so trajectories are not misdirected.
4. The estimator's own Monte-Carlo error, `sqrt(mc_var_loglike)` ~ 2-3 nats, is
   ~150x larger and is accepted by design via `mc_variance_budget`.

Point 4 needs one qualification: MC error with fixed samples is a *smooth*
function of the hyperparameters, whereas roundoff is genuinely discontinuous, so
the 150x ratio is not a strict apples-to-apples bound.  That is why the jitter
probe (result 1, column 6) is reported separately -- it measures the non-smooth
part directly and agrees.

**But this is a scale-dependent verdict with ~10x headroom, not a
precision-independent one.**  Since the error is 1 ulp of a sum proportional to
nobs:

| nobs | 1 ulp of `loglike` | expected error | vs HMC energy scale |
|---|---|---|---|
| 9,000 | 0.016 | 0.004-0.019 nats | 0.09x |
| 100,000 | 0.125 | 0.03-0.15 nats | **0.75x** |
| 1,000,000 | 1.0 | 0.25-1.2 nats | **6x** |

## If a future catalogue needs it

*Superseded 2026-08-07 -- see `2026-08-07-float32-recentering.md`.*  Two
corrections to the paragraph that originally stood here:

1. It proposed "casting only the two scalar accumulators to float64" with x64
   left disabled.  That is not possible in JAX: with `JAX_ENABLE_X64=0` a
   requested float64 dtype is silently downcast to float32, so the targeted
   cast requires enabling x64 globally and pinning every large array to
   float32 explicitly.
2. It would not have worked anyway.  Recentering (implemented in
   `pop_cosmo_model(loglike_ref=..., log_mu_sel_ref=...)`) removes the
   final-sum representation error exactly as a float64 accumulator would, and
   the measured float32-vs-float64 offset barely moves: the offset is
   dominated by *coherent element-wise* error (shared float32 scalars such as
   the population-model normalization entering every event identically),
   which no accumulator precision can touch.  What recentering does fix is
   the non-smooth reduction-order jitter (result 1, column 6) and the
   representation of the potential itself.  The recentering note has the
   production-scale measurements and the revised scaling discussion.

Global x64 remains the wrong tool: measured peak memory went 6.01 -> 10.52
GiB, and it roughly halves GPU throughput.

## Operational gotchas found

- **A float64 leg at production scale needs
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.**  JAX's default 75% preallocation
  fragmented the device and produced a spurious OOM on a 96 MB allocation, while
  the same run needs only 3.7-10.5 GiB.
- **`run_inf.py` records no `extra_fields`**, so `az.from_numpyro` captures only
  `diverging`; `energy` and `accept_prob` are discarded and sampler health
  cannot be checked after the fact.  Adding
  `extra_fields=('energy','accept_prob','num_steps','diverging')` to the
  `mcmc.run` call would make exactly this class of question answerable from a
  finished run.
- **A prior value exactly on a `TruncatedNormal` bound kills initialization.**
  `sigma=0.05` with `TruncatedNormal(0.1, 0.1, low=0.05)` has log_prob -inf, and
  `initialize_model` fails with the unhelpful "Cannot find valid initial
  parameters."  Relevant because `gwtc3-cosmo.prior`'s `sigma` prior is already
  tight relative to some config true values.

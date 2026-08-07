# Float32 recentering of the potential

*2026-08-07.  Implements the "recentering" fix motivated by
`2026-08-07-float32-accuracy-audit.md`.  Applies to
`src/intensity_models_fast.py`; measured with
`scripts/test_float32_accuracy.py --recenter`.*

## What it is

MCMC is invariant under adding a constant to the potential, and the thing
that varies over the posterior is small: the total log likelihood moves by
O(#parameters) ~ 15 nats, regardless of nobs.  Yet the model was summing
`log_like_per_event` to a float32 of magnitude ~16*nobs (1.46e5 at
nobs=9000) whose 1 ulp = 1.6e-2 nats was the audit's headline error.  The
fix subtracts fixed baselines *inside* the sums:

- `pop_cosmo_model(loglike_ref=..., log_mu_sel_ref=...)`: a constant (nobs,)
  per-event baseline subtracted before the event sum, and a constant scalar
  subtracted from `log_mu_sel` inside the selection factor.
- `recentering_baselines(model_args, ref_params, **model_kwargs)` evaluates
  both baselines at a reference point (the init point is fine; a plain
  float32 evaluation is sufficient because whatever numbers come out are
  exact constants once fixed) and returns them plus the dropped constant
  `offset = sum(loglike_ref) - nobs*log_mu_sel_ref`.
- `run_inf.py` computes the baselines at the init point automatically, prints
  the offset, and stores it in the output netcdf as
  `posterior.attrs["recentering_offset"]`.  The recorded `lp`/`energy` are
  shifted by the offset; add it back if absolute values are ever needed.
  Defaults (`None`) reproduce the previous behaviour bit-for-bit.

**Verified invariance** (float64, production scale, identical z0): the
potential shifts by exactly the dropped constant (difference 3.6e-12 of
1.87e4) and every gradient component matches to 4.9e-15 relative.  The
posterior is untouched by construction and in practice.

## What it does — and the audit claim it refutes

Production scale (nobs=9000, nsamp=4000, nsel=1.7M), recentered vs the
audit's unrecentered numbers:

| point | offset before | offset after | jitter std before | jitter std after |
|---|---|---|---|---|
| truth | +1.87e-2 | -1.58e-2 | 9.6e-3 | 7.7e-3 (one 2-ulp sel jump) |
| sharp | +1.38e-2 | -9.5e-3 | - | 4.4e-4 |
| broad | +5.8e-3 | +1.76e-2 | - | 8.7e-4 |

Two distinct findings:

1. **The reduction-order jitter of the event sum is eliminated.**  With
   recentering, permuting the 9000 events leaves the potential
   bit-identical.  The remaining jitter is entirely the `log_mu_sel` scalar
   (below).  At evaluation points far from the baseline (sharp/broad probe
   points ~1e4 nats away, like an early-warmup trajectory) the centered sum
   is large again and 1 ulp of *it* reappears (1.95e-3 at broad) — still
   ~8x below the old floor, and near the posterior bulk, where NUTS spends
   its time, the centered sum is O(10) nats and contributes nothing.

2. **The float32-vs-float64 offset is unchanged (~1-2e-2 at all three
   points).  This corrects the audit's mechanism claim.**  The audit
   attributed the offset to the representation limit of the final sum and
   proposed float64 accumulators as the fix.  Recentering removes the
   final-sum representation error *exactly as a float64 accumulator would*
   (the float32 leg's centered event sum is literally 0.0 at the reference
   point), yet the offset persists: it is dominated by *coherent
   element-wise* error — theta-dependent scalars (population normalization,
   `log_mu_sel`, ...) that are rounded once in float32 and enter every
   event's log weight identically, so their ~1e-6 errors are amplified by
   nobs instead of averaging out.  (Cleanest at nobs=400: the recentered
   float64 leg reads the entire float32 error as +6.3e-4 in the centered
   `loglike` term — the coherent element-wise sum — while the audit's
   unrecentered total at the same scale was 5.9e-4.)  No accumulator
   precision, Kahan summation, or recentering can touch this term.  The
   audit's numbers were right; the attribution ("not accumulated roundoff
   over elements") was wrong — the two components are the same order at
   every nobs, which is also why the audit could not resolve the
   loglike/selfactor split.

Why the offset finding is not alarming: a bias that varies smoothly with the
hyperparameters is absorbed into the target density at the ~1e-6 relative
level and is invisible to HMC; what degrades sampling is the *non-smooth*
component, which is exactly what the jitter probe measures — and which
recentering does fix on the event side.  The recentered term decomposition
now resolves the split directly: at truth, -4.9e-3 (loglike, element-wise
coherent) + 8.5e-3 (selfactor) with everything else negligible.

## The remaining floor: the `log_mu_sel` scalar

The selection factor is `-nobs * (log_mu_sel - ref)`, and the absolute error
of the `log_mu_sel` scalar itself (~1 ulp of 14.1 = 9.5e-7, no matter how it
is summed) is coherently amplified by nobs.  Observed directly: the truth
point's one non-zero selection permutation moved the potential by exactly
2 ulp x 9000 = 1.717e-2 nats.  This term is quantized — most orderings give
the identical float32 scalar (3 of 5 orderings bit-equal), occasionally it
steps by nobs*ulp.

Scaling of the post-recentering non-smooth error (post-warmup, so the
centered event sum is O(10) nats and free):

| nobs | sel-scalar step nobs*ulp(14) | vs 0.2-nat HMC energy scale |
|---|---|---|
| 9,000 | 8.6e-3 (2-ulp: 1.7e-2) | 0.04-0.09x |
| 100,000 | 0.095 | ~0.5x |
| 1,000,000 | 0.95 | ~5x |

So recentering buys the next ~10x of catalogue growth in pure float32, with
the selection scalar becoming the binding (and directly monitorable) term.
Beyond that, two options:

- **Pure float32:** rescale `pdraw_sel` by a constant so that `log_mu_sel`
  sits near 0 instead of 14 (correcting `R` by the same constant).  The ulp
  of the final scalar shrinks ~16x and the error floor moves to the ~1e-7
  relative error of the internal weight sum — roughly another 10x.  Not
  implemented.
- **Targeted float64:** enable x64 and pin every large array explicitly to
  float32, keeping only the scalar tail (logsumexp outputs, factors) in
  float64.  Note the audit's original suggestion — casting two accumulators
  to float64 with x64 *disabled* — is not possible in JAX (requested float64
  is silently downcast to float32), and per finding 2 above an accumulator
  cast alone would not have reduced the offset anyway.

One caveat at very large nobs: the coherent element-wise *bias* (finding 2)
also grows ~1.6e-6*nobs and is not removed by any of the above; at 1M events
it reaches ~1 nat of smooth-ish bias.  Its non-smooth fraction is unknown —
a float64 verification leg at that scale would be needed before trusting
float32 there, independent of the selection-scalar fix.

## Cost and operational notes

- No measurable cost: peak GPU memory 6.42 GiB recentered float32 (vs 6.01
  unrecentered in the audit, difference from the added `log_mu_sel`
  deterministic and harness bookkeeping); the extra work per likelihood call
  is one (nobs,) subtraction and one scalar subtraction.
- The baselines must be treated like the data: the float64 harness leg loads
  the float32 leg's baselines (`--ref_in/--ref_out`, analogous to z0),
  because legs with different constants differ by a real constant, not by
  roundoff.  Likewise the per-event baseline must be permuted together with
  the events in the jitter probe.
- `recentering_baselines` warns if any event has (near-)zero likelihood at
  the reference point: a dead reference event puts ~1e6 back into the sum
  and defeats the recentering.  Initialization would fail at such a point
  anyway.
- Reproduce: `uv run python test_float32_accuracy.py --recenter --stress`
  (both legs need `XLA_PYTHON_CLIENT_PREALLOCATE=false` at production scale,
  see the audit's gotchas).

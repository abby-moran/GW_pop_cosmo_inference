# Removal of the `join_point_terms` / `log_mix_at_join` machinery

*2026-08-06.  Applies to `src/intensity_models.py`, `src/intensity_models_fast.py`,
and `src/intensity_models_coup.py`.*

## What was removed

Every `LogDNDM` class used to carry a method `join_point_terms(z)` that
returned two things:

1. `mbhmax_at_samples` — the location of the join point,
   `mbhmax(z) = mpisn + mpisndot * (1 - 1/(1+z)) + dmbhmax`;
2. `log_mix_at_join` — the log-height of the rest of the mass function
   (the normalized PISN component plus `flow` times the low-mass Gaussian
   peak) evaluated *at* `m = mbhmax`.

`LogDNDM.__call__` also accepted a `join_terms=` keyword so a caller could
compute these once per redshift and share them between the `m1` and `m2`
evaluations.

The second return value was computed and then thrown away — the call sites
unpacked it as `mbhmax_at_samples, _ = ...`.  We have now deleted the whole
method and the `join_terms=` kwarg; `__call__` computes `mbhmax(z)` inline.

## Why `log_mix_at_join` existed

It is a fossil of an earlier parameterization of the high-mass power-law
tail.  Originally the tail was **height-matched** ("glued") to the bulk of
the mass function:

```
p_tail(m) = fpl * mix(mbhmax) * (m / mbhmax)^(-c)
```

where `mix(mbhmax)` is exactly `exp(log_mix_at_join)`.  In that design,
`fpl` (and analogously `flow` for the peak) were *continuity-anchored
amplitudes*: `fpl = 1` meant "the tail starts as tall as the continuum is
at the join," not "the tail contains some fraction of events."

## Why that design was abandoned

Commit `e7ad354` (2026-07-21) switched the tail to an independently
normalized shape.  The reason, recorded in the commit itself: with
height-anchored amplitudes the *total* integral of the mass function is not
fixed — it drifts strongly with `flow` and mildly with `z` (because the
join height depends on the PISN grid, which depends on `a`, `b`, `sigma`,
`mpisn`, `dmbhmax`, `mpisndot`, and `z`).  That drifting normalization was
silently absorbed into the other inferred hyperparameters and broke MCMC
recovery of injected populations.

The current model instead builds a proper mixture over three **unit-area**
shapes (PISN, low-mass peak, tail), with simplex weights derived from
`flow` and `fpl`:

```
w_pisn : w_low : w_pl  =  1 : flow : fpl,   normalized by 1/(1 + flow + fpl)
```

so the mixture integrates to ~1 in `m` at every `z` regardless of the
hyperparameters, and `flow`/`fpl` are directly interpretable as relative
population fractions.  This is the parameterization the current analyses
(and the hierarchical-likelihood treatment as in Golomb et al.) assume when
placing priors on `log_fpl` / `log_flow`.

Note this is a question of **amplitude bookkeeping, not continuity**: the
total density is continuous at the join either way, because the tail rises
smoothly from zero there via `log_smooth_turnon` (and, since the
`smooth_tail_edge=True` default in the fast module, the hard zeroing of the
tail below `mbhmax` is gone too, which is what makes the NUTS gradients for
`h`, `mpisn`, `dmbhmax` correct).

## Why delete rather than keep

After `e7ad354`, `log_mix_at_join` was computed and discarded.  It cost
nothing at runtime (XLA dead-code-eliminates it), but it was misleading —
it looked like a load-bearing piece of the model, and the `join_terms=`
plumbing suggested a sharing optimization that no caller actually used
(the fast module shares `mbhmax(z)` between `m1` and `m2` through a
different path, `call_from_logs`).

## Practical consequences

- **No change to any number.**  The removed code never affected the
  potential, its gradients, or any output.  The full equivalence suite
  (`scripts/test_fast_equivalence.py`) passes unchanged.
- **API change:** `LogDNDM.__call__(m, z)` no longer accepts `join_terms=`,
  and `join_point_terms` no longer exists.  Old notebooks that called
  either will fail loudly with an `AttributeError` / `TypeError`.
- If you ever want the glued-tail model back (e.g. to reproduce pre-July
  2026 runs), the construction lives in the git history: `e7ad354^` has the
  last height-matched version, and the diff of `e7ad354` documents the
  simplex conversion.

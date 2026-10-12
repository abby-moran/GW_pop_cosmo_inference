# Tail anchoring at `zref`: the amplitude is fixed, the tail still moves

*2026-10-12. Applies to `src/intensity_models_fast.py` on `main`. Records
behaviour that was previously undocumented; no code change.*

## Summary

The high-mass power-law tail is **height-anchored at a single reference
redshift**, `zref = 0.001`, and that is the only mode `main` supports (the
old `tail_anchor = "ref_z"`; see "History" below). Two facts about it that
are easy to conflate:

1. **The tail's amplitude does not evolve with redshift.** The sampled
   parameter is `r` (via `log_r`), the ratio of tail height to the
   PISN+bump mixture height at the join, *at `zref`*. It is converted once,
   at construction, into a scalar simplex weight `fpl`, which is then used
   unchanged at every `z`.
2. **The tail's location does evolve when `mpisndot != 0`.** The tail's
   onset and smooth turn-on are tied to `mbhmax(z)`, not to `mbhmax(zref)`.
   When mass evolution is on, the tail slides up or down in mass together
   with the PISN edge.

So "the tail doesn't evolve" is true of its weight and false of where it
sits. This is the intended design: the PISN feature floats around a tail
of fixed amplitude. It is written down here because nothing in the code's
naming makes (2) obvious, and the question has come up.

## What is held fixed

`LogDNDM.__post_init__` (`:633-642`) converts `r` to `fpl` exactly once:

```
f_eff(z) = r * mu_j(z) * m_j(z) / (c - 1)          # log_f_eff, :720
fpl      = f_eff(zref)                               # :642, zref = 0.001
```

where `m_j(z) = mbhmax_at_z(z)` (`:698`) is the join and `mu_j(z)` is the
normalized PISN+bump mixture height there (`log_mix_at_join`, `:709`).
After that line `fpl` is a plain scalar; `call_from_logs` says so itself
(`:764-766`: "already converted once ... constant here, not recomputed per
sample"). The mixture is then the usual simplex

```
w_pisn : w_low : w_pl = 1 : flow : fpl,   normalized by 1/(1 + flow + fpl)
```

so the tail's *fractional* weight is the same at every redshift.

## What moves

In `call_from_logs` (`:752-762`) every piece of the tail is evaluated
against `mbhmax_at_samples = mbhmax_at_z(z)`:

```
:758   log_p_pl_raw = log_normalized_power_law_tail_from_log(log_m, log(mbhmax_at_z(z)), c)
:761   (hard-edge path)  where(log_m < log(mbhmax_at_z(z)), -inf, ...)
:762   log_p_pl = log_p_pl_raw + log_smooth_turnon(m, mbhmax_at_z(z))
```

With `mbhmax(z) = mpisn + mpisndot * (1 - 1/(1+z)) + dmbhmax`, a nonzero
`mpisndot` moves the tail's onset by the same amount it moves the PISN
edge. Two consequences follow directly:

- **The tail density at a fixed mass changes with `z`.** The unit-area
  tail is `(c-1)/m_j * (m/m_j)^-c`, i.e. proportional to `m_j(z)^(c-1)` at
  fixed `m`. For `c > 1` and `mpisndot > 0` the tail gets denser at a given
  mass above the edge as `z` increases, even though its total weight is
  constant.
- **The height ratio at the join drifts away from `r`.** With `fpl` fixed
  at `f_eff(zref)`, the actual tail/mixture ratio at the join at redshift
  `z` is

  ```
  r(z) = r * [mu_j(zref) * m_j(zref)] / [mu_j(z) * m_j(z)]
  ```

  equal to `r` only at `zref`. This is the PISN feature floating relative
  to the fixed-amplitude tail -- the behaviour wanted -- not an error.

When `mpisndot == 0` *and* `mlow_dot == 0`, `mbhmax_at_z` and `mu_j` are
constant (`_pisn_z_dependent`, `:646`; `_z_dependent`, `:648`), so
`f_eff(z) == f_eff(zref)` identically and anchoring at one redshift is the
same as anchoring at every redshift. With `mlow_dot != 0` only, the
residual `z`-dependence enters through the bump term of `mu_j`
(`mp_low_at_z(z)`, `:717`) and is negligible, since a <= 2.5 Msun-wide
bump at ~10 Msun contributes ~nothing at a >= 40 Msun join.

## Two further things that are *not* fixed

- **`zref` itself is hardcoded.** `zref: object = 0.001` on `LogDNDM`
  (`:624`), `LogDNDV` (`:797`) and `LogDNDMDQDV` (`:864`). It is not a
  model kwarg, not an ini key, not a prior site, and the same field sets
  the rate-normalization pivot (`LogDNDV.log_norm = -self(zref)`, `:802`),
  so the tail anchor and the rate pivot cannot be moved independently
  without a source edit.
- **The tail's absolute rate still follows the global rate evolution.**
  `fpl` is a fraction of the mixture, and the whole mixture rides
  `LogDNDV(z)` (`lam`, `kappa`, `zp`). A tail that is non-evolving in the
  sense of a separate channel with its own rate history would be a
  two-population model, which this is not.

## Considered and not done: pinning the onset at `mbhmax(zref)`

Making the tail truly static would mean evaluating `:758-762` against
`mbhmax_at_z(zref)` instead of `mbhmax_at_z(z)` while leaving the PISN
continuum on `mbhmax_at_z(z)`. Mechanically this is a one-site change
that stays consistent between the event and selection terms, because
every evaluation path (`__call__` `:739`, `table_over_grid` `:750`, the
event weights `:1432`, the selection weights `:1420`) goes through the
single `call_from_logs`. It was not done, for two reasons:

- **`mpisndot < 0` opens a hole.** Priors allow it
  (`scripts/priors/CE_cosmo_2evo.prior`: `mpisndot = Uniform(-25, 25)`).
  If the PISN edge drops below a fixed tail onset, the region between
  them holds only the log-normal scatter shoulder of the continuum
  (width `sigma`) plus the tail's 5% turn-on. A few `sigma` of gap and
  the density there is exponentially suppressed; any event landing in it
  at that `z` gets a floored likelihood with vanishing gradient. The
  posterior would be pushed away from negative `mpisndot` by model
  structure rather than by data.
- **`mpisndot > 0` creates two edges.** The tail's turn-on at
  `mbhmax(zref)` and the pile-up at `mbhmax(z)` become separately
  visible, so the tail can absorb pile-up density at high `z` and the
  edge/peak assignment can become multimodal.

If this is revisited, an onset of `min(mbhmax(zref), mbhmax(z))` avoids
the hole while keeping the tail static for `mpisndot > 0`.

## History

- `e7ad354` (2026-07-21) replaced the original height-matched ("glued")
  tail with independently normalized unit-area shapes and simplex weights,
  because the glued amplitudes let the total mass-function integral drift
  with the hyperparameters (`notes/2026-08-06-join-point-machinery-removed.md`).
- The branch `lvk-powerlaw-z` (commit `95d0f2f`, 2026-09-01) reintroduced
  height anchoring **as a parametrization of `fpl`**, not of the density:
  `r` is converted to a simplex weight, so the unit-area mixture and its
  fixed normalization are untouched and the 2026-07 objection does not
  recur. That branch offered three modes (`simplex`, `ref_z`, `per_z`);
  `main` adopted `ref_z` alone, via an independent reimplementation
  (`308ee29`, `3f01003`), and the other two were deliberately not revived
  in the selective port `db9be4a` (2026-10-12).
- The `kappa(c)` turn-on area correction is omitted from `log_f_eff`
  (`:723-725`). Under the single-`zref` conversion this is a
  `z`-independent constant factor on the mixture normalization, which
  rescales `R` and cancels in the `R`-marginalized likelihood. (It would
  *not* cancel under a per-redshift conversion, which is one more reason
  `per_z` should not be brought back without revisiting it.)

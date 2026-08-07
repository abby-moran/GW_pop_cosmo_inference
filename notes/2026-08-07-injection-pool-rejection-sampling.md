# Injection pool: rejection sampling, mass-proposal cap, pdraw normalization

*2026-08-07.  Applies to `scripts/gen_inj.py` and `scripts/reweight_res.py`
(mock-population generation).  The inference code is untouched; existing
datasets can be reproduced with `sampling_method = multinomial`.*

## The problem these changes fix

An end-to-end validation run (endO5-style mock, 9000 events, 4000 PE samples)
recovered several hyperparameters far from truth -- beta off by 6.3 posterior
sigma, log_fpl by 4.9, c by 4.0, a by 3.4 -- while every internal check was
clean: generation and inference population densities agreed bit-exactly, the
mock-PE machinery passed a population-weighted PIT test at the 1-2% level,
zero divergences, r_hat <= 1.006, and the per-event and selection n_eff
diagnostics all looked healthy.  Refitting the *same* events at their exact
true parameters (no PE at all, `scripts/run_inf_true.py`) landed on the same
offset values, exonerating the likelihood and the PE machinery.

The cause was the injection pool.  `reweight_res.py` used to build the mock
population by resampling `num_tot = 2e7` draws **with replacement** from the
proposal pool, with probability proportional to the population weight
`w = pop/pdraw`.  But a pool of proposal draws only carries
`n_eff = sum(w)/max(w) ~ Z` effective population draws -- about 2e5 for the
12.6M-row validation pool, because the broad proposal (m1 ~ powerlaw(0.5) on
[2, 2502] Msun, z ~ U(0, 6.5)) puts ~99.4% of its draws where the population
is negligible.  Resampling 2e7 from an effective 2e5 replicates each distinct
truth ~100 times: the 9000 "events" contained only 2060 distinct injections
(Kish-effective ~700 independent events; single truths repeated up to 55x).

The fit then faithfully recovers the *empirical* population of the pool,
which deviates from the smooth truth by the realization noise of a ~700-event
sample -- while the posterior contracts as if there were 9000 independent
events.  Truth-recovery z-scores are inflated by ~sqrt(9000/700) ~ 3.6x, so
all the "biases" above were ordinary <2 sigma realization scatter reported
with 3.6x-overconfident error bars.  No per-event diagnostic can see this:
each duplicated event is individually healthy; the deficit is cross-event
independence.

**Production configs are affected too.**  `mock_O5_noevo.ini` sets
`n_total = 3e7`, so reweighting reads only the first 30M rows of the
injection file even when the file holds the full `num_loops = 150` (1.5e9)
draws.  A 3e7-row pool supports ~5e5 effective draws -> roughly 1700
effective events per 9000 -> ~2.3x overconfident recovery tests, which will
masquerade as persistent 2-4 sigma hyperparameter "biases".

## Change 1: rejection sampling in `reweight_res.py` (default)

Each pool row is now tested once and accepted with probability
`exp(log_w - log_w_max)`.  Accepted rows are exact i.i.d. draws from the
population, and duplicates are structurally impossible.  Consequences:

- The yield is whatever the pool actually supports (~Z draws).  `num_tot`
  is now a **cap**, not a promise: excess acceptance is thinned uniformly;
  a shortfall proceeds with a loud warning that includes the pool size
  needed to reach `num_tot`.  Pool exhaustion is visible instead of being
  silently papered over with duplicates.
- The `ndraw` column (the count the selection-function estimate divides by)
  records the actual number of accepted population draws.
- `sampling_method = multinomial` in the `.ini` restores the old
  resampling-with-replacement exactly, for reproducing legacy datasets.

Rule of thumb for pool sizing: the acceptance efficiency is printed
(~0.9% per surviving pool row with the capped proposal); to get N
independent population draws you need ~N/efficiency pool rows, and
`n_total` must be at least that large or the file is not even read.

## Change 2: mass-proposal cap in `gen_inj.py`

The source-frame m1 proposal upper bound now equals the SNR grid's
detector-frame maximum (`m1_grid.max()`, 1575 Msun for the A+ grid) instead
of a hardcoded 2502.  Since `m1_det = m1_src (1+z) >= m1_src`, any draw above
the grid max can never land inside the grid at any redshift and was already
being discarded with SNR = 0 -- the cap is exactly behavior-preserving while
raising the in-grid fraction from ~42% to ~60% (~1.4x more usable injections
per CPU-hour).  A detectability audit (population- and volume-weighted,
including the h = 1.2 prior edge) found the expected number of detections
with m1_src > 1000 among 2e7 population draws is ~4e-7, so even much lower
caps would be physically safe; the grid-max cap needs no physics argument at
all.  `pdraw` renormalizes automatically because it is evaluated from the
same frozen scipy distribution.  (`gen_inj.py` also gained a `num_loops`
config key, fallback 150, and resolves its ceph output directory via $HOME
rather than a relative path.)

## Change 3: pdraw_sel normalization

`pdraw_sel` used to be recorded as `w * pdraw_cosmo = pop * exp(-log_w_max)`,
which is proportional to, but not equal to, the normalized density the draws
actually follow, `p = pop/C` with `C = int pop ~ Z exp(log_w_max)/n_total`.
The missing `n_total/Z` factor deflated `mu_sel`, and hence the recovered
rate `R = nobs/mu_sel`, by the constant `Z/n_total` (~1/74 for the validation
pool).  Population-*shape* posteriors were unaffected -- the factor is
constant in the hyperparameters -- but the mock rate was meaningless.
`pdraw_sel` now includes the factor, and the reweighting prints the mock's
implied true rate `R = n_pop/C`, so rate recovery is a testable part of
validation.

## Diagnostics (all in `scripts/`, reusable against any dataset)

- `test_gen_vs_inf.py` -- point-by-point check that the inference-side
  population density at truth minus `log(pdraw_sel)` is constant across the
  selection set (`log_C` for normalized files, `log_w_max` for legacy ones);
  any trend with m1/q/z localizes a generation-vs-inference inconsistency.
- `test_pe_pit.py` -- population-weighted PIT of the true parameters within
  each event's PE samples; must be U(0,1) per coordinate if the PE cloud is
  the flat-prior posterior it claims to be (no selection correction needed:
  detection and the mass cut condition on data only).
- `run_inf_true.py` + `call_inf_true.sh` -- refit events at their exact true
  parameters (one delta-function sample each, `neff_penalty="none"`);
  separates data-realization effects from PE-machinery effects.
- Duplication census: hash the sel file's true `(m1, q, z)` and compare
  distinct vs total counts, and the Kish effective size of the event set.

## Checklist for existing / future runs

- Regenerate mock datasets with rejection sampling before quoting
  truth-recovery accuracy; with the old datasets, deviations up to
  ~sqrt(nobs / n_eff_events) posterior sigma are expected and are not model
  biases.
- Raise `n_total` to the full injection-file size (and `chunk_size` to ~2e6;
  40k-row chunks make a 1e9-row pass take a day).
- Watch the printed acceptance efficiency and the exhaustion warning; grow
  `num_loops` until the accepted count reaches the target.
- The recovered `R` from pre-fix datasets is deflated by the constant
  `Z/n_total` of that run; shapes are fine.

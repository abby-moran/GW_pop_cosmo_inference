# The N_effective penalty: what changed and why

*2026-08-07.  Applies to `pop_cosmo_model` in `src/intensity_models_fast.py`.
Default behavior changed; `neff_penalty="min_neff"` reproduces the original.*

## Background: why there is a penalty at all

The hierarchical likelihood replaces two integrals with Monte-Carlo sums:

- **per event**: the marginal likelihood of event *i* is estimated by
  importance-sampling over its `nsamp` PE samples with weights
  `w_ij = dN(theta_ij) / pdraw_ij`;
- **selection**: the expected number of detections is estimated from the
  injection set.

Both estimators have Monte-Carlo *variance*, and NUTS is very good at finding
the corners of parameter space where that variance blows up: regions where a
handful of samples carry all the weight look (noisily) high-likelihood, and
the chain can wander into them and get stuck.  The standard diagnostic is the
effective sample size of the weights,

    n_eff_i = (sum_j w_ij)^2 / (sum_j w_ij^2),

which is `nsamp` for uniform weights and ~1 when a single sample dominates.
To leading order, **1/n_eff_i is the MC variance of that event's log
likelihood** (in nats^2).  The penalty is a soft barrier added to the log
posterior that pushes the sampler away from high-variance regions instead of
letting it exploit them.

## What the original did, and its problems

```python
numpyro.factor("neff_criteria",
               min(0, (min_i n_eff_i - nobs) / (0.05 * nobs)))
```

i.e. penalize when the *worst single event's* n_eff drops below the *number
of events*.  Three problems:

1. **Unsatisfiable when nsamp < nobs.**  A per-event n_eff can never exceed
   `nsamp`.  Our production runs have nsamp = 4000 PE samples and
   nobs = 9000 events, so `min_i n_eff_i >= 9000` cannot hold: the penalty
   was *always active*, contributing a constant ~-11 to the log posterior
   with a permanently nonzero gradient driven by whichever event happened to
   be worst.  (Confirmed in the end-to-end validation: the posterior sits at
   min_neff ~ 10.)  The threshold `nobs` is borrowed from the *selection*
   criterion (see below), where it belongs; there is no argument for it as a
   per-event target.
2. **The `min` is doubly non-smooth.**  Its gradient flows only into the
   argmin event and jumps whenever the identity of the worst event switches;
   and the `min(0, x)` hinge has a kink at the threshold.  Gradient jumps
   produce HMC energy errors — i.e. divergences — exactly when trajectories
   probe the guard.
3. **Worst-event is the wrong summary anyway.**  What controls the MC error
   of the *total* log likelihood is the sum of the per-event variances, not
   the single worst term.

## What the new default does

```python
mc_var = sum_i 1 / max(n_eff_i, 1)        # MC variance of total log L
numpyro.factor("neff_criteria",
               -softplus((mc_var - budget) / (0.05 * budget)))
```

with `neff_penalty="mc_variance"` and `mc_variance_budget=5.0` as defaults.

- **The penalized quantity is the actual error budget.**  `sum_i 1/n_eff_i`
  is (to leading order) the MC variance of the total log likelihood, the
  quantity whose control the importance-sampling literature actually asks
  for (cf. Talbot & Golomb 2023, arXiv:2304.06138).  The MC standard
  deviation is its square root: budget 5 means sigma_logL <~ 2.2 nats,
  budget 1 means <~ 1 nat.  It cannot be made unsatisfiable by nsamp/nobs
  bookkeeping, and every event contributes smoothly — no argmin switching.
- **The boundary is smooth.**  `-softplus(x)` is the C-infinity version of
  the hinge `min(0, -x)`: ~0 well inside the budget (down by e^-1 one
  smoothing width inside, ~1e-5 at half the budget), exactly -log(2) at the
  threshold, and asymptotically linear with the same slope beyond it.  No
  gradient jump anywhere.
- **Dead events stay finite.**  An alive event always has n_eff >= 1
  (Cauchy-Schwarz), so 1/n_eff <= 1; an event whose weights all underflow
  has n_eff = 0 and is capped at contribution 1.  Its likelihood is already
  floored at -1e6, so the penalty never needs to produce an inf/NaN.
- **The selection guard keeps its threshold, gains the smooth boundary.**
  n_eff_sel >= 4*nobs (the Farr 2019 criterion — this is where a
  "number-of-events" threshold genuinely belongs, because the selection
  integral enters the likelihood ~nobs times) is unchanged in threshold but
  uses the same -softplus form in the new modes.

### The three modes

| `neff_penalty=` | event guard | selection guard |
|---|---|---|
| `"mc_variance"` (default) | sum 1/n_eff vs budget, softplus | 4*nobs, softplus |
| `"min_neff"` (legacy-exact) | min n_eff vs `neff_criterion` (= nobs), kinked | 4*nobs, kinked |
| `"none"` | none | 4*nobs, softplus |

In **every** mode the scalars `min_neff` and `mc_var_loglike`
(= sum_i 1/n_eff_i) are recorded as deterministic sites, so the a-posteriori
check is always available.  `"min_neff"` reproduces the original model's
potential exactly (both guards, kinks and all) — that is also what the
equivalence test suite pins.

## Cost: zero

Measured at 9000 x 4000 (A6000): 18.0 / 18.3 / 17.9 ms per value+gradient
for min_neff / mc_variance / none — differences are run-to-run noise.  The
n_eff's come from the same max-subtracted pass the likelihood's logsumexp
already needs, so either penalty adds only one square-and-sum over the
weight array; and with `"none"` XLA dead-code-eliminates even that from the
potential (deterministic sites are only evaluated in postprocessing).
Choose on statistical grounds.

## Practical guidance

- After a run, look at `mc_var_loglike`.  If the posterior sits comfortably
  below the budget (validation run: 3.45-3.83 vs 5), the guard was inactive
  and the posterior is exactly what you would have gotten with no penalty —
  it only ever acted as a safety rail.  If the posterior *piles up against
  the budget*, the guard is shaping your posterior: either raise
  `mc_variance_budget` (and accept more MC noise) or, better, increase
  `n_pe` / the number of selection injections.
- Narrow population features are the classic trigger: e.g. a low-mass
  Gaussian peak with true width < 1 Msun concentrates the per-event weights
  on few PE samples, dragging n_eff down and sum 1/n_eff up.  If you plan
  such runs, check `mc_var_loglike` early on a short chain.
- `min_neff` is still recorded; a value near 1 for some posterior draws
  means single events are being estimated by essentially one PE sample —
  worth investigating even when the total variance looks acceptable.

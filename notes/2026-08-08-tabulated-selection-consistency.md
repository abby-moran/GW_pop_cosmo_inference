# The mpisndot-free pathology: numerator and denominator must use the same density

*2026-08-08.  Fixes a bug introduced by the 2-D mass-function table
(`2026-08-07-mass-table-2d.md`, commit `7fc7a6e`), which made every mock run
with `mpisndot` free walk onto the prior walls.  See `handoff.md` for the runs
that exposed it.*

## Symptom

Three production-scale mock-O5 tests (9000 events, truth `mpisndot = 0`):

| run | free | verdict |
|---|---|---|
| `endO5_fullcosmo` | h, Omh2, w | 18/18 truths in 95% — healthy |
| `endO5_evo` | **mpisndot** | 4/16 — `mpisndot` pinned to the −2 prior floor |
| `endO5_fullcosmo_evo` | h, Omh2, w, **mpisndot** | 4/19 — same floor mode |

(With `mpisndot` pinned, `_z_dependent` is False and the selection set already
went through the table, so those runs were consistent all along — which is
exactly why only the evo runs failed.)

The failure tracked freeing `mpisndot`, not freeing cosmology.  Convergence
diagnostics were clean (r̂ ≈ 1, ESS in the hundreds, `min_neff` ≈ 7.5,
`mc_var_loglike` ≈ 3.7 against a budget of 5): a real posterior mode against
the walls, not a broken warmup or an MC-noise artifact.

The mode's tell is that *almost every* parameter sat on a prior boundary:
`sigma` → 0.0501 (floor 0.05), `dmbhmax` → 0.508 (floor 0.5), `a` → −1.636
(floor −1.65), `b` → −2.086 (floor −2.1), `mpisndot` → −1.990 (floor −2),
`fpl` → 1.78 (ceiling 2).  Nothing about the physics wants that corner; it is
the corner where the mass function is as narrow and as fast-moving in z as the
priors allow.

## Cause

The 2-D table note made a deliberate choice: with `mpisndot` sampled, the
`(nobs, nsamp)` event samples were evaluated on the 2-D table while the
selection set kept the direct evaluation.  The reasoning was per-piece
accuracy — the z-lerp bias in `log_mu_sel` is O(3e-3) per point and the
selection factor multiplies `log_mu_sel` by `nobs`, so tabulating the
selection set looked like the dangerous half.

That reasoning had the wrong criterion.  With `R` marginalized under a
scale-invariant prior, the hierarchical likelihood is a **ratio**

    prod_i lambda(x_i) / (int lambda p_det)^nobs

so it is a probability model only when numerator and denominator evaluate the
*same* `lambda`.  Splitting them leaves the numerator's interpolation error
completely uncancelled.  That error is parameter dependent, and since it is
one-sided (log-linear interpolation of a peaked log density overestimates
where the log density is convex), the sampler can climb it without bound.  It
did: it drove every shape parameter to whichever wall made the mass function
sharpest and its z motion fastest, i.e. to wherever a 30-node z-lerp is
worst.

## Measurement

Potential decomposed at two points on the real `endO5_evo` data (9000 events x
4000 PE samples, float32, GPU) — `truth` from `pop_configs/mock_O5_noevo.txt`,
`mode` the maximum-`lp` sample of `runs/endO5_evo/O5_evo.nc`:

| config | logpost(mode) − logpost(truth) |
|---|---|
| **split** (table on PE, direct on selection — the shipped path) | **+99.2  → mode wins** |
| **consistent, both tabulated** | −26.2  → truth wins |
| **consistent, both direct** | −25.7  → truth wins |

Where the +99 comes from, relative to the all-direct evaluation:

| at the mode | `loglike` | `selfactor` | total |
|---|---|---|---|
| split − direct | **+124.77** | 0.00 | **+124.77** |
| consistent − direct | +124.77 | **−125.30** | **−0.53** |

The z-lerp bias is +124.8 nats in the event sum at that point.  With both
sides on the table it is matched by −125.3 in the selection factor and the
ratio survives to 0.5 nats.  With the split, the −125 is simply absent.

At the truth point the same bias is 0.07 nats — which is why
`test_tabulated_path_zdep`, which only probed near truth, passed.

Scan along `mpisndot` with all other parameters at truth, consistent path
vs. direct path: the two agree to ≤ 0.16 nats over the whole prior range
[−2, 8], and the potential's shape is essentially identical.

## Fix

`pop_cosmo_model` gains `tabulate_selection`, defaulting to `None` = *follow
`tabulate_mass_function`*, i.e. always consistent.  In the z-dependent case
the selection set now goes through the same `_log_weights` closure as the
event samples (which the z-independent case already did).  Setting it to
`False` restores the split and is for diagnostics only.

Note this also makes the 2-D path *cheaper*: the selection set is back on a
bilinear lerp instead of the full direct evaluation.

An interpolated-but-consistent model is a slightly different population model,
not a broken one.  Whatever bias the z-lerp carries is common to both sides of
the ratio and largely cancels; the residual is the difference between the PE
and selection populations' averages of it, measured at 0.5 nats even at the
sharpest point the priors allow.  Accuracy per point was never the
requirement — consistency was.

## Result: the rerun recovers the truth

`runs/endO5_evo2/O5_evo2.nc` (Slurm 6785493) — same data, prior and seed as
`runs/endO5_evo`, only the fix differs:

| | truth in 68% | truth in 95% | divergences | max r-hat |
|---|---|---|---|---|
| old (split) | 3/15 | 4/15 | 14 | 1.83 (`mpisn`) |
| **new (consistent)** | **11/15** | **15/15** | **0** | 1.02 |

Every parameter is off its prior wall.  Key ones (truth -> median [16%, 84%]):

| param | truth | old | new |
|---|---|---|---|
| `mpisndot` | 0 | -1.99 [-2.00, -1.98] | -1.20 [-1.76, -0.25] |
| `mpisn` | 33.29 | 38.66 [38.48, 39.26] | 32.23 [30.88, 33.47] |
| `dmbhmax` | 3.44 | 0.51 [0.50, 0.52] | 4.12 [3.10, 5.31] |
| `sigma` | 0.0539 | 0.0502 [0.0500, 0.0504] | 0.0557 [0.0514, 0.0637] |
| `a` | -0.943 | -1.635 [-1.646, -1.610] | -0.011 [-0.710, 0.449] |

`mpisndot` still sits low — truth is just outside 68% but comfortably inside
95% — which matches the 1-D potential slice (interior maximum near -0.6, see
below).  That is this mock realization, not a pathology.

Going from 14 divergences to 0 is itself evidence about the mechanism: the
old chains were probing a surface where the numerator and denominator
disagreed, and the disagreement was steepest exactly where they ended up.

## Cost

Same GPU (A6000), production scale, `mpisndot` free, via
`bench_model.py --module intensity_models_fast --mpisndot_free [--split|--no_tab]`:

| | gradient / leapfrog | peak GPU |
|---|---|---|
| consistent (default) | **37.93 ms** | 9.37 GiB |
| split (`--split`) | 38.43 ms | 9.37 GiB |
| direct (`--no_tab`) | 66.04 ms | 20.42 GiB |

The fix is free -- marginally faster, since the selection set drops from the
full direct evaluation to a bilinear lerp.  `--split` was added to
`bench_model.py` for this A/B only; it does not produce a valid likelihood.

## Follow-up: is n_z = 30 enough once mpisndot is free?

Making the paths consistent exposed a separate question.  `tab2d` and
`direct` no longer agree tightly at the production `n_z = 30` -- 1.2 nats at
nobs=400 on the synthetic bench point (`mpisndot = 3`) -- and some gradients
differ by far more.  Both turn out to be **z-grid discretization, converging
correctly**, not a second bug.

Potential difference `|tab2d - direct|` at that point, same data, varying n_z:

| n_z | 30 | 60 | 120 | 240 |
|---|---|---|---|---|
| nats | 1.216 | 0.383 | 0.097 | 0.024 |

That is ~4x per doubling: the O(dz^2) rate linear interpolation must have.
The two are the same model, sampled on different z grids.

The gradients are the striking part.  At the same point:

| | n_z=30 | n_z=60 | n_z=120 | n_z=240 |
|---|---|---|---|---|
| d/d`mpisn` | 92.9 (direct 82.6) | 32.0 | 28.6 | 29.1 |
| d/d`dmbhmax` | 60.4 (direct 51.4) | 4.59 | 1.44 | 1.86 |
| d/d`mpisndot` | 15.6 (direct 12.1) | 4.64 | 4.42 | 4.49 |

tab and direct agree to 1.4% by n_z=240, but *both* are off by 3-30x at
n_z=30.  This corrects the record in `2026-08-07-mass-table-2d.md`, which saw
FD swing "81 -> 69 -> 30 -> 41" against an AD value of 82.6 and concluded AD
was the trustworthy side.  The converged answer is ~29: the large-step FD was
closer, and the n_z=30 AD value is the slope of a grid ripple, not of the
model.

**But this does not affect production.**  What sets the posterior is the
potential, not the gradient (AD returns the exact gradient of the potential
actually evaluated, so HMC's stationary distribution is the one that
potential defines; bad gradients cost efficiency and divergences, not
correctness).  On the real 9000-event data the potential profiles are
converged at n_z = 30:

    logpost vs mpisndot (all else at truth), relative to mpisndot = 0

    mpisndot   n_z=30   n_z=60  n_z=120
      -2.000    1.843    1.906    1.914
      -1.250    3.002    2.972    2.968
      -0.625    3.375    3.402    3.414
       0.000    0        0        0
      +1.000   -6.691   -6.672   -6.664

Agreement is <= 0.12 nats across profiles spanning ~10 nats, and a fine scan
in `mpisn` at `mpisndot = -1` behaves the same.  Since the number of grid
kinks scales with n_z, profiles that do not separate are direct evidence the
kinks are not contributing.  The n_z=30 gradient artifact appears only at
`mpisndot = 3` on the adversarial synthetic bench data -- a region the real
data disfavours by 6.7 nats already at `mpisndot = +1`.  **n_z = 30 stays the
production default.**

Worth noting for the science: on the fixed model the `mpisndot` slice has an
*interior* maximum near -0.6 (+3.4 nats over the truth) and falls back to
+1.8 at the -2 floor.  There is no runaway to the wall any more.  The mild
preference for negative `mpisndot` is a property of this mock realization
along a 1-D slice at fixed other parameters; the marginal posterior will be
flatter.

## Regression guard

`scripts/test_fast_equivalence.py` gained `test_tabulated_selection_consistency`
(test 8).  It tests an **exact identity**, not a tolerance on a physical
difference.  Feed the model a selection set that *is* the event samples,
flattened, with the matching `pdraw`, and set `nsamp = Ndraw` bookkeeping
accordingly; then

    log_mu_sel + log(Ndraw) == logsumexp_i(loglike_i) + log(nsamp)

holds to roundoff if and only if both code paths evaluate the same density on
the same points.  Measured at a point mirroring the failed mode
(`sigma = 0.051, dmbhmax = 0.52, mpisndot = -1.99, a = -1.6, b = -2.0,
c = 1.5, mpisn = 38.5`):

| config | identity residual | x nobs=400 |
|---|---|---|
| default (consistent) | +0.00000 | +0.00 |
| `tabulate_selection=False` | -0.00993 | -3.97 nats |

`selfactor` multiplies that residual by `nobs`, which is why a 1e-2 drift is
~90 nats of potential at production scale.  The test asserts only on the
default -- and separately fails if the sharp point stops separating the split
from the default, since that would mean the guard has gone blind rather than
that the model got better.

A first attempt compared consistent-vs-direct and split-vs-direct
*potentials* under a tolerance.  That was the wrong instrument: on
`make_synthetic_data` the selection samples come from a much broader
population than the events, so their interpolation biases do not cancel the
way they do on real data, and the split landed *closer* to the direct path
than the consistent one did.  Luck is not a guard.

## Reproducing the measurements

Ad-hoc diagnostics used above, all in `scripts/` and all run from there:

| script | what it shows |
|---|---|
| `diag_evo.py` | potential + per-factor decomposition at truth vs. the failed mode, for split / consistent / direct, on the real run data; `--profile` adds the `mpisndot` scan |
| `diag_nz.py` | tab-vs-direct potential and per-parameter gradients as n_z is refined (the O(dz^2) convergence table) |
| `diag_nz_profile.py` | logpost profiles along `mpisndot` and `mpisn` at several n_z on the real data (the "n_z=30 is enough" evidence) |

## Lesson

For any importance-sampled hierarchical rate model, an approximation applied
to the event likelihoods must be applied identically to the selection
integral.  A per-point error budget is not the right test; the right test is
whether the same function is on both sides of the ratio.  If it is not, the
error does not merely add noise — it defines a direction the sampler will
follow.

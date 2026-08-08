# Review of `555e833`: tabulated-selection consistency with free `mpisndot`

## Verdict

I agree with the core fix.  It corrects a real likelihood inconsistency, and
the reported behaviour is exactly what that inconsistency can produce.

With the rate marginalized under the stated prior, the population term is

```text
prod_i lambda(x_i) / (int lambda(x) p_det(x) dx)^nobs.
```

The old 2-D path used `lambda_table` for the event samples but
`lambda_direct` in the selection integral.  It was therefore not the
selection-corrected likelihood of the specified population model.  The new
default makes the selection and event paths call the same `_log_weights`
closure when tabulation is enabled; setting
`tabulate_selection=None` to follow `tabulate_mass_function` is the correct
default.

The all-table and all-direct calculations need not agree exactly: the table
defines a slightly different, but internally consistent, approximated
intensity.  Internal consistency is essential; per-factor agreement with the
direct implementation is not a valid reason to split the numerator and
denominator.

## Evidence checked

- The dedicated consistency test has the right structural form: using the PE
  samples themselves as the selection set makes
  `log_mu_sel + log(Ndraw)` equal `logsumexp(loglike_i) + log(nsamp)` when
  both paths use the same density.  A reduced local run gave zero residual for
  the default and a `-0.01196`-nat residual at the sharp split point.
- The saved chains substantiate the reported rerun.  `runs/endO5_evo/O5_evo.nc`
  has 14 divergences and maximum finite scalar R-hat 1.829; the fixed
  `runs/endO5_evo2/O5_evo2.nc` has zero divergences and maximum finite scalar
  R-hat 1.022.  The reported posterior quantiles are consistent with those
  files.

## Concerns and follow-up work

1. The note is internally inconsistent about the failed run's convergence.
   It says the diagnostics were clean with R-hat approximately one, but later
   gives a maximum R-hat of 1.83 for the old chain.  The latter is correct.
   The former should be changed to say that the importance-sampling/
   Monte-Carlo diagnostics were healthy, while MCMC convergence was not.

2. The test still says that AD is the trustworthy side at `n_z=30`.  This
   contradicts the new note's more careful convergence study, which finds
   that the `n_z=30` derivative can be a grid-ripple slope rather than the
   refined-grid derivative.  The inline test commentary should be updated,
   and AD-vs-FD agreement at fixed `n_z` should not be presented as a test of
   physical-gradient accuracy.

3. The selection-consistency fix does not itself establish that `n_z=30` is
   accurate enough for science results.  The documented `n_z=30` profiles
   vary by at most 0.12 nats only on selected fixed-parameter slices, whereas
   the adversarial synthetic example has gradients wrong by factors of
   3--30.  Before relying on the fast posterior, compare the `n_z=30`
   potential to `n_z=60` or 120 over posterior draws.  Importance-reweighting
   `endO5_evo2` and reporting the reweighting ESS and posterior shifts would
   be a low-cost check; a short `n_z=60` chain would be stronger validation.

4. The new structural regression test describes itself as exact but accepts
   an absolute residual below `1e-3` nats.  Multiplied by 9,000 events, that
   would permit roughly 9 nats of production-scale distortion.  The threshold
   should be justified in terms of a production potential-error budget and,
   if float32 roundoff permits, tightened.

5. `tabulate_selection=False` deliberately restores an invalid target when
   mass-function tabulation is enabled.  It is well documented, but leaving
   it as a normal public model option creates an avoidable footgun.  Prefer a
   clearly unsafe diagnostics-only switch, or isolate the split computation
   in the diagnostic script rather than the production model API.

## Recommendation

Keep the commit: returning to the split path would reintroduce the pathology.
The appropriate next validation is posterior-wide sensitivity to the
remaining z-grid discretization, not an attempt to make the selection factor
match the direct path while the event factor remains tabulated.

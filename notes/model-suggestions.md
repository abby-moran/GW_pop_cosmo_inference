# Model reparameterizations and remaining performance levers

*2026-08-09.  Written after the scatter-free VJP pass
(`notes/2026-08-09-scatter-free-vjp.md`).  Companion to the optimization
changelog.*

*UPDATE, same day: suggestions 1 and 2 were implemented and benchmarked on
branch `local/reparam-pass-1`; see `notes/2026-08-09-pivot-reparam.md`.
Outcome: the pivot (suggestion 1) is a real ~2x improvement in worst-case
gradients per effective sample and is recommended for production.
Log-space sampling (suggestion 2) is a clear empirical regression --
the multiplicative-degeneracy argument below did not survive contact with
the data -- and removing sigma's hard floor causes mass divergences (the
floor guards a genuine MC-noise cliff, not just a prior wall).*

## Status of code-level speedups

We are not entirely out of ideas, but they are in diminishing-returns
territory.  A production gradient with free `h` / `Omh2` / `mpisndot` is
~30 ms after the scatter-free VJP.  The `endO5_fullcosmo_evo3` run spends
up to 1023 gradients per iteration (`max_tree_depth=10`) for ESS in the
tens.  The cost that matters is therefore *gradients per effective
sample*, which is a geometry problem rather than a kernel problem.
Reparameterization is the right place to look next.

## Suggested reparameterizations

Ordered by expected payoff.

### 1. Pivot the redshift-dependent quantities  [IMPLEMENTED -- WORKS]

`mpisn` is defined at `z = 0`, but the data constrain the mass scale best
around the bulk of detections (`z ≈ 0.3–0.8`).  That makes `mpisn` and
`mpisndot` strongly correlated by construction — the same reason
power-law fits always use a pivot.  Reparameterize to

```text
mpisn(z) = mpisn_ref + mpisndot · (x − x*),   x = z/(1+z)
```

with the pivot at the detection-weighted mean of `x` (`zpivot = 0.75` for
the endO5 mock).  Benchmarked ~2.3x better worst-case grads/ESS, 4.6x on
`mpisndot`.  Caveat: tighten the `mpisn_ref` prior bounds so the derived
`mpisn` stays inside the old support for every `mpisndot`.

The same logic applies to the rate evolution (`lam` is effectively
measured at `z ≈ 0.3`, not 0), but the rate amplitude `R` is already
analytically marginalized, so the payoff there is smaller.

### 2. Log-space for the multiplicative degeneracies  [IMPLEMENTED -- REJECTED]

The hypothesis: the spectral-siren degeneracy is multiplicative (source
mass = detector mass / `(1 + z(dL; h))`), so `log h` / `log mpisn`
coordinates should straighten the banana for the dense mass matrix.
Empirically false on the endO5 mock: every iteration saturates
`max_tree_depth=10`, the step size collapses 3x, and worst-case grads/ESS
gets 5x *worse*.  Support remains in the model (`log_h`, `log_mpisn`,
`log_sigma`, `log_mp_low` prior names) for future experiments, but do not
use it for production.

Separately, `sigma`'s hard floor (`low=0.05`) must stay: removing it
(sampling `log_sigma` unbounded) produced 94/150 divergences.  The floor
fences off a real cliff -- at small sigma the narrow PISN peak is
under-resolved and the MC-noise guard region is rough -- not just a prior
wall.

### 3. `w` is the suspect for the remaining stiffness

`w` only enters through `dL(z)` at moderate `z`, nearly degenerate with
`h` / `Omh2` / evolution combinations.  The principled fix is to sample
what the data measure — e.g. `E(z)` or `dL` at a pivot redshift — and
derive `w`, but that is invasive.  First look at evo3/evo4's pair plots in
unconstrained space to see if `w`'s ridges are actually curved.

### 4. If hand-crafted transforms stall: NeuTra

Fit a normalizing-flow guide (`AutoBNAFNormal`) with SVI — at 30 ms/grad,
a 10k-step SVI fit is minutes — then run NUTS through `NeuTraReparam`.
This composes with the pivot.  With ~18 free parameters this is squarely
in the regime where it works well.

## Other non-reparameterization levers

- The likelihood cost is linear in sample counts.  Measured on evo2:
  `mc_var_loglike` ≈ 3.5 of budget 5, so only ~30% headroom in PE samples
  (do not cut 4000/event); the selection set has ~3.6x margin
  (`neff_sel` ≈ 131k vs required 36k).  The real lever is per-event
  *adaptive* sample allocation: `min_neff` ≈ 6-7 means a few events
  consume most of the variance budget.
- More, shorter chains (`chain_method="vectorized"`) buy wall-clock time
  if trees stay shallow.

## Practical next step

Run the pivot prior (`runs/priors/gwtc5_fullcosmo_evo_pivot.prior`) on the
full endO5 mock and compare mixing against evo3/evo4.  Then revisit
`sigma`'s geometry (floor-adjacent posterior) and item 3.

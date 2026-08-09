# Post-run diagnostics: `scripts/diagnose_run.py`

*2026-08-09.  One command that reads a finished run's `.nc` and says what is
wrong with it and which config key to change.  Collects the thresholds that
were, until now, scattered across this `notes/` directory and re-derived by
hand for every run.*

## What it is for

The recurring failure of this campaign has not been that runs crash; it is
that a run *finishes* and then takes an afternoon of ad-hoc arviz to decide
whether its numbers can be quoted.  Worse, the two failure modes look alike
from a corner plot: `endO5_broadbump` has a clean sampler and an
unidentifiable parameter, `endO5_fullcosmo_evo2` has a well-posed model and a
sampler that never mixed.  Both produce a plausible-looking figure.  This
script separates them.

```bash
cd scripts
uv run python diagnose_run.py --run endO5_broadbump
uv run python diagnose_run.py --nc ../runs/endO5_evo/O5_evo.nc \
                              --prior ../runs/priors/gwtc5_evo.prior
uv run python diagnose_run.py --run endO5_evo2 --json
```

`--run` resolves the single `.nc` under `../runs/<name>/`, the same way
`plot_corner.py` does.  `--prior` and `--pop_config` are optional: they are
read from an `.ini` copied into the run directory, or failing that from the
`scripts/run_configs/*.ini` whose `[run] run_dir` matches.  Anything that
cannot be found degrades to an explicit *"insufficient information"* finding
rather than a guess -- `runs/endO5_val2`, which predates `extra_fields`, is
the standing regression case for that path.

Exit code **0** for OK/NOTE/WARN, **1** if any finding is FAIL, **2** on a
usage error.  `JAX_PLATFORMS=cpu` is set inside the script, so it never
competes with a running job for a GPU.

## The checks, and where each threshold comes from

Nothing below is invented; each line traces to a note or to the model source.

### 1. Sampler convergence

| quantity | threshold | provenance |
|---|---|---|
| r-hat | <=1.01 OK, <=1.05 NOTE, <=1.10 WARN, >1.10 FAIL | standard |
| bulk / tail ESS | >=200 OK, >=100 NOTE, >=50 WARN, <50 FAIL | calibrated so `endO5_evo2` (min bulk ESS 118), our reference healthy run, lands at NOTE |
| divergences | 0 / <0.2% / <1% / >=1% | standard; the text names the 14-divergence `endO5_evo` precedent |
| BFMI | <0.3 WARN | Betancourt |
| `lp` r-hat + per-chain `lp` means | >1.05 WARN | catches `fullcosmo_evo2` (1.229) and `endO5_evo` (1.068), both of which have chains at different log-posterior levels |
| tree-depth cap | taken from the config's `max_tree_depth`, cross-checked against `max(n_steps) = 2^d - 1` | `run_inf.py` default 7 |

When `recentering_offset` is present the report says so and warns that only
`lp` *differences* are meaningful (`notes/2026-08-07-float32-recentering.md`).

### 2. Monte-Carlo adequacy -- deliberately two-sided

The point is to catch settings that were **needlessly strong** as well as too
weak; the campaign has paid for both.

- **`mc_var_loglike`** (= `sum_i 1/n_eff_i`, the MC variance of the *total*
  log likelihood) against `mc_variance_budget = 5.0`.  `>=1.0x` FAIL,
  `>=0.8x` WARN.  Below `1.0` absolute it reports that `n_pe` could be cut,
  and computes how far, using cost ~ linear in `n_pe` and variance ~ `1/n_pe`.
  Source: `notes/2026-08-07-neff-penalty-redesign.md`.  All production runs so
  far sit at 3.5 of 5, i.e. the guard has never shaped a posterior -- and
  halving `n_pe` would put them *over*, which the report states rather than
  recommending a cut it cannot support.
- **`min_neff`** <2 WARN, <5 NOTE, with the standing reminder that the legacy
  `neff_penalty="min_neff"` guard (>= `nobs`) would have been permanently
  active at the values we actually observe (~7).
- **`neff_sel`** against the model's `4*nobs` hinge: `<1x` FAIL, `<1.5x` WARN,
  `>3x` NOTE (oversized).  The FAIL text states that `neff_sel <= nsel`, so a
  selection set smaller than `4*nobs` can never satisfy the guard, and quotes
  the required `nsel` -- the 17,624-rows-against-a-36,000-hinge case from
  `notes/2026-08-09-low-mass-bump-width-identifiability.md` that got a run
  cancelled before it started.

`nobs` comes from `evt_end - evt_start`; without a config the `4*nobs` check
is skipped explicitly.

### 3. Sampler geometry

Recommendations are conditioned on joint evidence, never on saturation alone
-- every run we have measured saturates depth 7, so saturation by itself
carries no information.

| evidence | verdict |
|---|---|
| saturated + low ESS + depth/`dense_mass` not yet raised | raise `max_tree_depth`, and `dense_mass = true` when max \|r\| > 0.7 (with the caveat that a dense metric estimates `d(d+1)/2` entries from the final warmup window, so `nmcmc` may need raising too) |
| saturated + low ESS + depth already > 7 **and** `dense_mass` already on | conditioning problem, not trajectory length: each level doubles cost for a factor 2 in length.  Reparameterize instead |
| saturated + healthy ESS | depth is the cost driver, not the failure mode; raising it is an efficiency choice |
| **not** saturated + low ESS | `max_tree_depth` is not the binding constraint -- look at the posterior surface (walls, modes, divergences) |
| not saturated + healthy ESS | settings were stronger than needed; `max_tree_depth` / `nmcmc` could come down |

The correlation-matrix condition number is reported but deliberately **not**
used to predict a required depth: it sits at 266-836 across healthy and broken
runs alike and `log2(sqrt(cond)) ~ 4.3` badly under-predicts the observed
depth 7.  It is supporting evidence for `dense_mass` only.

### 4. Model choice / identifiability

This is the part a generic arviz summary does not give.

- **Prior-dominated marginals.**  Posterior sd / prior sd (prior sd from 60k
  draws of the parsed numpyro distribution).  `>0.9` WARN, `>0.6` NOTE, with
  the explicit statement that such a median is not a measurement.  On
  `fullcosmo_evo2` this correctly flags `h` (1.02) and `w` (0.93).
- **Prior walls.**  Fraction of draws within 1% of the prior's effective range
  of a truncation bound.  `>0.5` WARN, and **three or more parameters above
  0.3 simultaneously is FAIL**, naming
  `notes/2026-08-08-tabulated-selection-consistency.md` as a possible cause.
  On `endO5_evo` this fires with six parameters pinned; on
  `endO5_fullcosmo_evo` -- whose r-hat is 1.01 and min ESS 632 -- it fires with
  seven, which is the whole point: that run's sampler is fine and its
  likelihood is not.  The finding separates the measured pattern from the
  inferred cause in as many words.
- **Degeneracy directions.**  Pearson *and* Spearman for the strongest pairs;
  `|rho| > |r|` by more than 0.05 marks a curved (banana) degeneracy, which
  `dense_mass` cannot straighten.  Plus an eigen-decomposition of the
  free-parameter correlation matrix, reporting every direction with eigenvalue
  > 2 and its loadings -- the same instrument used in `mass-model-audit.md`
  section 7.2.
- **The low-mass bump.**  `msigma_low > utils.BUMP_MSIGMA_LOW_MAX` (2.5) or
  `msigma_low / mp_low > 0.3` means the Gaussian has stopped being a bump and
  become a second continuum across 6-20 Msun, the only window where the CO-IMF
  index `a` has leverage.  The report adds the Gaussian's mass fraction in
  that window, the Fisher `sigma(a)` log-interpolated from the audit's
  2.55 / 0.73 / 0.29 / 0.15 table, the `a`-bump correlations, and that block's
  share of the leading eigenvector.  Source: `mass-model-audit.md`,
  `notes/2026-08-09-low-mass-bump-width-identifiability.md`.

### 5. Truth recovery

Quantile of each truth within its marginal -- reported always, but **demoted
to NOTE behind an explicit "not interpretable" banner whenever min bulk ESS <
100**.  This is the one we were burned by: `endO5_fullcosmo_evo2` at bulk ESS
~3 shows 17/18 truths inside 95% while being unusable, and
`endO5_fullcosmo_evo3` at ESS 22 appears to badly miss `a` for a reason that
turned out to be a model property.  Both readings were noise.

## Validation

Run against every finished run with a known verdict.  The script's conclusion
matched in all seven cases.

| run | known verdict | script |
|---|---|---|
| `endO5_evo2` | healthy: ESS 118, r-hat 1.02, 0 div | sampler OK/NOTE throughout; WARN overall from the `a`/broad-bump finding only |
| `endO5_broadbump` | clean sampler, `a` degenerate + prior-pulled | NOTE sampler, geometry *"depth is the cost driver, not the failure mode"*, WARN bump (64% of the leading eigenvector on the `a`-`log_flow`-`msigma_low`-`mp_low` block) |
| `endO5_narrowbump` | `a` fine, `mpisn`/`dmbhmax`/`b` unmixed (ESS 4-13, r-hat 1.52) | FAIL ESS/r-hat; bump section **OK** (width 1.95, ratio 0.21); recommends `max_tree_depth` 7->10, `dense_mass`, `nmcmc` |
| `endO5_fullcosmo_evo2` | depth-7 saturated 99.9%, ESS 2.8, r-hat 1.92 | FAIL; 100.0% at cap, `lp` r-hat 1.229; recommends `max_tree_depth` then `dense_mass` |
| `endO5_fullcosmo_evo3` | depth 10 + dense, ESS 14.6, r-hat 1.12, still 100% capped | FAIL; *"conditioning problem, not a trajectory-length problem"*, top recommendation is reparameterization |
| `endO5_evo` | broken by the split-density bug: 14 div, r-hat 1.83, walls everywhere | FAIL on prior boundaries (6 params pinned) naming the tabulated-selection note; correctly declines to blame `max_tree_depth` (only 25.8% at cap) |
| `endO5_val2` | predates `extra_fields` | runs clean; depth / energy / `lp` / acceptance all reported as insufficient information |

Two runs come out WARN rather than OK because they were fit against the
`msigma_low = 4` truth, so `a` should not be reported from them; that is the
audit's conclusion, not a false positive, and their sampler subsections are
all OK/NOTE.  `endO5_fullcosmo` (not in the table) lands FAIL at min bulk ESS
48 with 13 divergences -- right on the boundary, and arguably harsher than the
"18/18 truths, healthy" verdict it carries in
`notes/2026-08-08-tabulated-selection-consistency.md`.

## Known limitations

- **`nsel` is not in the `.nc`.**  The "selection set too small" case can only
  be inferred from `neff_sel < 4*nobs` plus `neff_sel <= nsel`; the script
  states the required `nsel` rather than asserting the actual one.
- **"Step size fell while ESS stayed low" is not evaluable from one file** --
  there is no baseline within a single run.  The equivalent decision is
  encoded from the config instead (depth already raised and/or `dense_mass`
  already on), which reproduces the evo2 -> evo3 reasoning from a single `.nc`.
- **A badly-mixed chain under-estimates its own correlation matrix**, so the
  degeneracy block from a FAIL-level run is indicative only.  The report says
  this where it matters.
- The Fisher `sigma(a)` figure is interpolated from four audit points indexed
  by the *true* width and evaluated at the *posterior* width; it is context,
  not a prediction, and the audit's own caveat (the Fisher overstates the
  upward width) applies.
- Constant columns are detected with an exact range test (`np.ptp(v) == 0`),
  never a std threshold -- a parameter pinned to a float32 constant can still
  report std ~1e-7, which is what aborted a corner figure once
  (`plot_corner.py`).
- Without a prior file the sampled/derived split falls back to site names,
  including the interchangeable bump-amplitude parametrizations
  (`log_fpeak` / `logit_flow` / `log_flow`, see
  `notes/2026-08-09-log-fpeak-parametrization.md`).  Passing `--prior` is
  always more reliable.

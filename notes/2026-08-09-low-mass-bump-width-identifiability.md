# The low-mass bump must stay narrow or the CO-IMF index `a` is lost

*2026-08-09.  Follow-up to the mass-model audit (`mass-model-audit.md`,
prompted by `runs/endO5_fullcosmo_evo3` losing `a`).  Records the design
decision and the guard rails added for it.*

## Design intent

The low-mass Gaussian (`flow`, `mp_low`, `msigma_low` in `LogDNDM`) exists to
capture **narrow** features that the broken-power-law CO IMF cannot express —
e.g. a peak at ~9-10 Msun.  It is *not* meant to model broad low-mass
continuum structure; that is the power law's job (index `a` below the fixed
break at 20 Msun of CO mass).

## Why width matters

`a`'s entire identifiable content is a single smooth ramp in log density over
6-20 Msun (below `mco_floor` = 6 and above the break it is a pure amplitude,
exactly absorbed by the mixture weights under the marginalized rate `R`), with
maximum amplitude ~0.19 after mixture dilution.  A Gaussian with
`msigma_low` ≈ 4 at `mp_low` ≈ 9 puts ~78% of its mass inside that same
window: the three bump parameters then reproduce `a`'s ramp almost exactly
(cosine 0.91 with `log_flow`'s response; R² = 0.99 onto the other shape
sensitivities), and the likelihood is flat in `a` over half its prior range.

The degeneracy is controlled by the true width, not by `mco_floor`
(Fisher σ(a) at 9000 perfectly-measured O5 events, other parameters free):

| true `msigma_low` | 4.0 | 3.0 | 2.0 | 1.0 |
|---|---|---|---|---|
| σ(a) | 2.55 | 0.73 | 0.29 | 0.15 |

At width 4 the posterior on `a` is prior times a one-sided likelihood wall;
this is what happened in `endO5_fullcosmo_evo3` (`a` recovered at the prior
pull, truth −0.94 outside the chain, strong `a`–`log_flow`–`msigma_low`
correlations).

## The rule

**Keep `msigma_low` at or below ~2.5 Msun** (equivalently
`msigma_low / mp_low` ≲ 0.3), both for mock true populations and for the
inference prior's support.  The constant lives in
`utils.BUMP_MSIGMA_LOW_MAX`; `utils.warn_if_bump_too_broad()` prints a
warning from

- `scripts/reweight_res.py` when a pop config's true `msigma_low` exceeds it
  (the standard `mock_O5_noevo.txt` truth of 4.0 trips this — by design: that
  mock is a broad-bump universe in which `a` should not be reported);
- `scripts/run_inf.py` when the prior's upper support exceeds it.

## Tradeoff (deliberate)

Restricting the width is a real model restriction: if the true universe has
broad low-mass structure beyond a power law, the restricted model is
misspecified and the excess is absorbed by `a`, biasing it.  We accept this
because the bump is *defined* as a narrow-feature component; a broad "bump"
is a second continuum and makes `a` meaningless instead of merely biased.

## Status of the prior files

`runs/priors/gwtc5_*.prior` still carry
`msigma_low = TruncatedNormal(4.0, 2.0, low=0.5, high=8.0)`.  They are
deliberately **not yet capped**: the two in-flight confirmation runs
(`runs/endO5_broadbump`, Slurm 6791077; `runs/endO5_narrowbump`, Slurm
6791202) read `gwtc5_massonly.prior`, and editing it under a restartable job
would alter the very experiment that tests the σ(a)-vs-width prediction.
Once those runs land (expected: `a` prior-dominated at true width 4,
recovered with σ ≈ 0.3 at width 2), change the prior line to e.g.

```text
msigma_low = TruncatedNormal(1.5, 1.0, low=0.5, high=2.5)
```

and use true widths ≤ 2.5 in new mock pop configs.

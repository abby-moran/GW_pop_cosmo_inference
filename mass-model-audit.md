# The CO-IMF index `a` is degenerate with the low-mass Gaussian bump

*2026-08-09.  Audit of the single-mass function `LogDNDM` in
`src/intensity_models_fast.py`, prompted by `runs/endO5_fullcosmo_evo3`, which
recovered every parameter except `a` (median +0.007 vs truth −0.9426, truth
outside the chain entirely) with `a` strongly anti-correlated against
`msigma_low` and `log_flow`.  Two follow-up runs testing the conclusion are in
flight (§8).*

## Verdict

**A practical near-degeneracy, severe enough to be operationally
indistinguishable from an exact one at O5 volumes.**

With 9000 *perfectly measured* events and the real O5 selection function,
moving `a` from truth (−0.9426) to the evo3 posterior median (+0.0071) costs
**108.6** nats of log likelihood if nothing else moves, but only **0.230**
after re-fitting the other 13 shape parameters — and only **0.076** at
`a` = −1.65, the prior's lower bound.  The likelihood is flat to
< 0.25 nats over `a` ∈ [−1.65, ≈ +0.25]: the entire lower half of the prior
range.

It is not mathematically exact.  Compensation breaks down going *up*: N·KL
reaches 2.8 at `a` = +0.5 and 24.6 at `a` = +1.0.  So more data does help, but
only above the truth, never below it.  Within the prior the profile has one
wall and one cliff-free direction, and the prior on `a`
(`TruncatedNormal(2.35, 2, ...)`, `runs/priors/gwtc5_fullcosmo_evo.prior:4`)
pushes the posterior against the wall.

**The `mco_floor` mitigation described in the `log_dNdmCO` docstring does not
achieve what it claims.**  Fisher σ(`a`) is 2.45 with the floor disabled
(`mco_floor = mco_min = 4`) and 2.55 with it at 6 — a 4% effect.  The
degeneracy is controlled by the *bump width*: σ(`a`) = 2.548 / 0.731 / 0.289 /
0.154 at true `msigma_low` = 4 / 3 / 2 / 1.

Actionable summary: do not report `a` as measured at `msigma_low` ≈ 4; its
evo3 posterior is prior times a one-sided likelihood wall, and its offset from
truth is not evidence of a bug.

## Scope and method

Model code read: `src/intensity_models_fast.py` (production) with
`src/intensity_models.py` as the reference cross-check.  Truth point from
`scripts/pop_configs/mock_O5_noevo.txt`; run configuration from
`scripts/run_configs/mock_O5_fullcosmo_evo3.ini` (9000 events, 4000 PE samples
each, `runs/priors/gwtc5_fullcosmo_evo.prior`).

Two metrics are used throughout.

1. **Shape.** RMS of Δ log dN/dm on a log-mass grid, weighted by the
   source-frame fraction of events per log m at truth and with an additive
   offset projected out (overall amplitude is free, see §1.4).

2. **Statistical.**  N_obs · KL(p_det(truth) ‖ p_det(Λ)) with the rate `R`
   marginalized under a scale-invariant prior, so that

   ```text
   E[ln L(truth) - ln L(Λ)] = N_obs * KL( p_det(.|truth) || p_det(.|Λ) )
   ```

   where `p_det ∝ dN/dθ · p_det(θ)` is the normalized *detected* distribution.
   Estimated by self-normalized importance sampling on the 124,826 first-half
   injections of `runs/endO5_fullcosmo_evo3/sel_noevo.h5` — the same rows the
   likelihood uses — with n_eff = 124,431/124,826, so MC noise is negligible
   and common-mode.  N·KL is directly comparable to a Δ ln L: ≈ 0.5 is 1σ,
   ≈ 2 is 2σ.

   The Fisher matrix is the Hessian of N_obs·KL at truth, taken by JAX
   autodiff through the production `LogDNDMDQDV`.

Cosmology is pinned at truth and measurement is perfect in both metrics.  Both
assumptions *narrow* the posterior, so every degeneracy number below is a
lower bound on the real flatness.

## 1. Mechanism

### 1.1 Where each parameter acts

The single-mass density is built in `LogDNDM.call_from_logs`
(`src/intensity_models_fast.py:861-890`) as a three-component mixture:
a PISN component of weight 1, a Gaussian of weight `flow`, a power-law tail of
weight `fpl`, normalized by `log1p(flow + fpl)` (`:872-882`), and multiplied by
the low-mass window `mmin_log_smooth_turnon` (`:889`).

The remnant map is the identity below `mpisn`
(`mean_mbh_from_mco`, `:571-574`), so CO mass ≈ BH mass over the whole low-mass
region; the lognormal smear has width `sigma` = 5.4% only.  The CO-mass axis
and the BH-mass axis are therefore effectively the same axis below ~33 M☉.

Mixture composition at the truth point, evaluated on the density (before the
window):

| m [M☉] | 3.2 | 4.0 | 6.0 | 8.0 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | ≥25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PISN share | 0.013 | 0.383 | 0.195 | 0.178 | 0.233 | 0.300 | 0.436 | 0.642 | 0.839 | 0.951 | 0.989 | ≈1 |
| bump share | 0.987 | 0.617 | 0.805 | 0.822 | 0.767 | 0.700 | 0.564 | 0.358 | 0.161 | 0.049 | 0.011 | ≈0 |

Weighted by the *detected* injections (component masses, both m1 and m2):

| m [M☉] | [3,4) | [4,6) | [6,10) | [10,14) | [14,20) | [20,33) | [33,∞) |
|---|---|---|---|---|---|---|---|
| bump share | 0.600 | 0.739 | **0.790** | **0.696** | 0.303 | 0.006 | 0.000 |
| weight of components here | 0.0002 | 0.018 | 0.194 | 0.208 | 0.162 | 0.223 | 0.196 |

76% of detected events have at least one component in [6, 20) M☉; only 3.5%
have any component in [3, 6).  So the band that matters is 6–20 M☉, and the
bump owns 70–79% of the density over most of it.

### 1.2 The exact `a`-response

The CO IMF is floored at `mco_floor` (`:608`, and `:622` in the `_from_log`
fast path), so its exact log-derivative is

```text
d ln(dN/dm_CO) / da  =  ln( 20 / max(m_CO, mco_floor) ) * 1{m_CO < 20}
                     =  ln(20/6) = 1.2040                for m_CO < 6
                        log-linear ramp 1.204 -> 0        for 6 < m_CO < 20
                        0                                 for m_CO > 20
```

The plateau below the floor is *not* zero — flooring `mco_eff` makes the
density `(mco_floor/20)^-a` there, which still carries `a`.  Above 20 the index
is −`b`, so `a` drops out entirely.

The PISN component is separately normalized (`log_Z_grid`, `:746`, subtracted
at `:864`), which removes the constant.  Autodiff through the production
grid gives, for the normalized PISN component alone:

| m [M☉] | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | ≥22 |
|---|---|---|---|---|---|---|---|---|---|
| ∂ ln P_pisn/∂a | +0.70 | +0.57 | +0.34 | +0.25 | +0.15 | +0.06 | −0.02 | −0.08 | **−0.0947** |

Total contrast 1.19, matching ln(20/6) − 0.095.  **`a` has no shape freedom
anywhere**: pure amplitude below 6, pure amplitude above 20, and a single
smooth monotone ramp between.

### 1.3 Why the bump can mimic it

For a mixture `p = (P + φG + ψT)/(1+φ+ψ)` with `φ = flow`, `ψ = fpl`,

```text
d ln p / d ln φ  =  f_bump(m)  -  φ/(1+φ+ψ)
```

which is exactly a monotone low-to-high ramp with the same sign structure, and
by §1.1 `f_bump` runs 0.99 → 0.80 → 0.70 → 0.30 → 0.005 over m = 3 → 8 → 12 →
16 → 22.  Measured directly, the cosine between ∂ ln p/∂a and
∂ ln p/∂log_flow in function space (source-frame event-weighted, overall
amplitude projected out) is

```text
cos( d ln p/da , d ln p/d log_flow )  =  0.918
```

Projecting ∂ ln p/∂a onto the span of the seven other shape sensitivities
(`b`, `c`, `mpisn`, `log_fpl`, `mp_low`, `msigma_low`, plus amplitude) gives
**R² = 0.9897** (source-frame weighted); onto the three bump parameters alone,
R² = 0.863.

The amplitude is also small.  Because the total response is
`f_pisn(m) × (PISN response)` and `f_pisn` ≈ 0.2–0.3 exactly where the ramp
lives, max |∂ ln dN/dm/∂a| over the whole band is **0.17** (at m ≈ 5) and its
rms is 0.082.  `a` is a 20%-level lever on a curve the bump controls.

### 1.4 Normalization: why the amplitude part is free

Each component is separately normalized — PISN over the [1.5, 100] M☉ grid
(`:746`), the Gaussian over the whole real line
(`log_normalized_gaussian`, `:661-663`), the tail over [mbhmax, ∞)
(`:665-671`).  The *overall* scale is then set at a single reference point
(`LogDNDMDQDV._normalize`, `:972-975`), not by an integral, and is absorbed by
`R` under the marginalized rate.

Consequence: the pure-amplitude halves of `a`'s response (the plateau below 6,
the −0.095 constant above 20) are *exactly* absorbable by `log_flow` and
`log_fpl`.  Only the 6–20 ramp is in principle distinguishable — and that is
precisely where the bump lives.

## 2. Direct compensation test

Fix `a` at a grid of values; re-fit the other shape parameters to minimize
N_obs·KL (L-BFGS-B from truth, box bounds equal to the prior supports).
N_obs = 9000.

| `a` | Δa | no refit | refit bump only (`log_flow`, `mp_low`, `msigma_low`) | refit all 13 |
|---|---|---|---|---|
| −1.650 (prior floor) | −0.707 | 23.40 | 1.88 | **0.076** |
| −1.400 | −0.457 | 11.14 | 0.91 | 0.041 |
| −0.9426 (truth) | 0 | 0 | 0 | 0 |
| −0.500 | +0.443 | 17.36 | 1.53 | 0.075 |
| **+0.0071** (evo3 median) | **+0.950** | **108.56** | 10.46 | **0.230** |
| +0.500 | +1.443 | 337.07 | 38.03 | 2.80 |
| +1.000 | +1.943 | 813.30 | 131.37 | 24.64 |
| +2.000 | +2.943 | 2979.6 | 1006.7 | 93.18 |
| +2.350 (prior mean) | +3.293 | 4209.1 | 1598.5 | 127.28 |
| +3.500 | +4.443 | 9719.4 | 3724.2 | 250.48 |
| +5.000 | +5.943 | 18257 | 5279 | 638.2 |
| +6.350 (prior ceiling) | +7.293 | 25478 | 5761 | 1254.8 |

The bump alone absorbs ~90% of the damage; the remaining 10% goes to
`log_fpl`, `c`, `b`, `mpisn`, `beta`.  Reading the profile: Δ ln L = 0.5 at
`a` ≈ +0.25, Δ ln L = 2 at `a` ≈ +0.45, and nothing at all below truth.

Pure-shape residual, rms |Δ log dN/dm| after the full refit (offset-free,
event-weighted):

| `a` | [4,60] | m > 15 | m > 20 | [4,20] | no refit, [4,60] |
|---|---|---|---|---|---|
| −1.650 | 0.0020 | 0.0017 | 0.0012 | 0.0026 | 0.047 |
| −0.500 | 0.0017 | 0.0009 | 0.0008 | 0.0023 | 0.042 |
| **+0.0071** | **0.0057** | **0.0022** | 0.0019 | 0.0081 | 0.106 |
| +0.500 | 0.0237 | 0.0104 | 0.0052 | 0.0340 | 0.189 |
| +1.000 | 0.0733 | 0.0786 | 0.0792 | 0.0660 | 0.295 |
| +2.350 | 0.1754 | 0.1015 | 0.0965 | 0.2194 | 0.685 |

At the evo3 median the mass function is reproduced to **0.6% in density**
across [4, 60] M☉, and the m > 15 residual (0.0022) is *smaller* than the
full-band one — the perturbation is not being shuffled out to high mass where
the bump is dead, it is genuinely absorbed everywhere.

## 3. Fisher / curvature at truth

Hessian of N_obs·KL over
(`a`, `b`, `c`, `mpisn`, `dmbhmax`, `sigma`, `log_fpl`, `mp_low`,
`msigma_low`, `log_flow`, `beta`, `lam`, `kappa`, `zp`), flat priors:

```text
sigma(a)  marginal     =  2.5479
sigma(a)  conditional  =  0.0855      (all other parameters held fixed)
degradation factor     =  29.8x
```

Fisher correlations with `a`:

| | `log_flow` | `msigma_low` | `log_fpl` | `mp_low` | `b` | all others |
|---|---|---|---|---|---|---|
| ρ(a, ·) | **−0.990** | **−0.978** | −0.941 | +0.811 | +0.364 | \|ρ\| < 0.10 |

σ(`a`) if individual parameters were fixed at truth:

| fixed | none | `log_flow` | `msigma_low` | (`log_fpl`, `c`) | bump triple | everything |
|---|---|---|---|---|---|---|
| σ(a) | 2.548 | 0.354 | 0.527 | 0.570 | 0.260 | 0.0855 |

Principal axis of the Fisher covariance, physical units, normalized to δa = +1:

```text
d log_flow   = -0.296
d msigma_low = -0.257
d log_fpl    = -0.097
d mp_low     = +0.093
everything else < 0.021
```

Smallest Fisher eigenvalue 0.132 against a spectrum reaching 1.4e5 — a
condition number of ~1.1e6 dominated entirely by this direction.

Adding the run's priors as diagonal Gaussian curvature gives σ(`a`) = 1.50.
That is larger than the observed evo3 posterior σ = 0.42, as expected: the
Fisher is a local quadratic and the profile in §2 shows the valley terminates
around `a` ≈ +0.4, while the low ESS of evo3 (§7) makes its interval an
under-estimate.  The two bracket the truth from opposite sides and are
consistent.

## 4. Where is `a` actually constrained?

Nowhere cleanly.

- **m < 6 M☉.**  The floor makes `a`'s effect exactly constant (§1.2), i.e. a
  pure rescaling of the PISN component — perfectly degenerate with `log_flow`
  by construction, not approximately.  Also only 3.5% of detected events have
  any component here.
- **6–14 M☉.**  `a`'s ramp lives here, but the bump carries 70–79% of the
  detected density and can reproduce the ramp to 0.918 cosine.
- **14–20 M☉.**  The bump has fallen to 30% → 0.5%; this is `a`'s only clean
  window.  It is also where `b`, `mpisn`, `dmbhmax` and `sigma` act, and it
  carries 16% of detected component mass.
- **> 20 M☉.**  `a` enters only as the constant −0.0947, absorbed exactly by
  `log_fpl` / `log_flow`.  The index there is −`b`; the PISN pile-up and the
  `fpl` tail dominate above ~33.

So `a`'s entire uncontested leverage is a ~1.4-in-log window, 14–20 M☉,
shared with four other parameters.  Hence the 30× degradation.

## 5. The tapering

Two tapers, and they overlap each other and the bump.

- **CO axis**, `log_smooth_turnon(mco, mco_min=4, width=0.05)` (`:632-636`,
  applied at `:724-726`): logistic with scale `mco_min * width` = 0.2 M☉.
  Values 0.119 at 3.6, 0.500 at 4.0, 0.953 at 4.6.
- **BH axis**, `mmin_log_smooth_turnon(m, delta_m=1.6, mbh_min=3)`
  (`:639-647`, applied at `:889`): support [3.0, 4.6].  Values 0.065 at 3.4,
  0.500 at 3.8, 0.935 at 4.2.

The bump's support covers both: Gaussian(9.121, 4.0) has 3.7% of its mass in
[3, 4] and ~5.5% in [3, 4.6].

**The floor interacts with the taper in the worst possible way.**
`mco_floor = 6` sits *above the entire taper window*, so throughout
[3.0, 4.6] M☉ `∂ ln(dN/dm_CO)/∂a` is exactly the constant 1.204 — `a` is a
pure multiplicative rescale of the PISN component there.  Its effect on the
low edge is not merely suppressed, it is *exactly* the effect of `log_flow`.

It is moot in practice, because the PISN component barely exists there: its own
low edge is set by `mco_min` = 4, not by `mbh_min` = 3, so below ~3.8 M☉ the
mass function is essentially **pure bump** (98.7% bump at m = 3.2).  Only 3.5%
of detected events have any component below 6 M☉ anyway.

`delta_m` and `mbh_min` are fixed floats in every production prior
(`runs/priors/gwtc5_fullcosmo_evo.prior:20-21`, and identically in
`gwtc5_cosmo`, `gwtc5_evo`, `gwtc5_fullcosmo`, `gwtc5_massonly`).  Fixing them
hides nothing relevant to `a` — freeing them would add another low-mass
amplitude knob and make matters marginally worse — but it does mean the low-edge
shape is not marginalized over, so quoted `flow` / `mp_low` uncertainties are
optimistic.

## 6. Does `mco_floor` succeed?

**It succeeds at the pathology it names and fails at the degeneracy it claims
to mitigate.**

Fisher σ(`a`) with the mock truth regenerated at each floor:

| `mco_floor` | 4.0 (off) | 5.0 | 6.0 | 8.0 | 10.0 |
|---|---|---|---|---|---|
| σ(a), data only | 2.450 | 2.522 | **2.548** | 2.548 | 2.538 |
| σ(a), data + prior | 1.481 | 1.496 | **1.500** | 1.501 | 1.508 |
| σ(a) conditional | 0.0855 | 0.0855 | 0.0855 | 0.0866 | 0.0914 |
| degradation | 28.7× | 29.5× | **29.8×** | 29.4× | 27.8× |

Turning the floor off entirely buys 4%.  The floor does prevent the
non-integrable pile-up at `mco_min` and the quadrature concentration — those are
real problems and it solves them — but the degeneracy lives at 6–20 M☉,
*above* the floor, where the bump has 78% of its probability.  If anything the
floor is mildly counterproductive for `a`: it truncates the response contrast
from ln(20/4) = 1.609 to ln(20/6) = 1.204 and converts the 3–6 M☉ response from
a slope into a pure amplitude.

**The bump width is the actual control parameter.**  Regenerating the mock with
narrower true bumps (floor held at 6, everything else at truth):

| true `msigma_low` | 4.0 | 3.0 | 2.0 | 1.0 |
|---|---|---|---|---|
| σ(a), data only | **2.548** | 0.731 | 0.289 | **0.154** |
| σ(a), data + prior | 1.500 | 0.682 | 0.286 | 0.153 |
| σ(a) conditional | 0.0855 | 0.0888 | 0.0897 | 0.0769 |

A factor **16.5** between width 1 and width 4, with the conditional σ flat —
i.e. the information content of the mass function barely changes; what changes
is how much of it the bump can steal.  `msigma_low` = 4 at `mp_low` = 9.1 is
the whole story.

### Two documentation errors in `log_dNdmCO`

`src/intensity_models_fast.py:596-604` makes two claims that this audit does
not support.

1. `:598-599` — "forcing `a` to fit both the 6-20 Msun slope and the 3-6 Msun
   region, degenerate with the bump parameters (flow, mp_low, msigma_low)".
   The floor does not relieve this.  σ(`a`) is unchanged to 4% whether the
   floor is on or off (table above), because the degeneracy is at 6–20 M☉,
   above the floor.  What the floor actually does in the 3–6 M☉ region is make
   `a`'s effect there *exactly* proportional to `log_flow`'s rather than merely
   similar.

2. `:602-604` — "Flattening below mco_floor keeps support down to mco_min (so
   the lower edge of the mass function is set by the mbh_min turn-on window,
   not the CO cutoff)".  The PISN component's low edge is set by
   `log_smooth_turnon(mco, mco_min=4, width=0.05)` at `:724-726`, i.e. by the
   CO cutoff at 4 M☉, which is *above* `mbh_min` = 3.  Between 3.0 and ~3.8
   M☉ the mass function is 99% bump; the `mbh_min` window and the `mco_min`
   turn-on both bite, and the CO cutoff is the higher of the two.

Neither is a code bug — the code does what it does correctly, and the same
floor is in the reference implementation (`src/intensity_models.py:31-35`), so
the mock and the inference use identical densities.  Only the docstring's
justification needs correcting.

## 7. Does this explain the evo3 posterior?

Yes, quantitatively, down to the parameter values.

### 7.1 The chain sits on the predicted ridge

The KL-optimal compensating point at `a` = +0.0071, computed with no reference
whatever to the chain:

| | refit prediction | evo3 posterior median |
|---|---|---|
| `log_flow` | −0.962 | **−0.935** |
| `msigma_low` | 3.569 | **3.444** |
| `mp_low` | 9.505 | **9.650** |
| `log_fpl` | −0.580 | −0.474 |
| `beta` | −2.426 | −2.397 |
| `c` | 2.365 | 2.228 |
| `b` | 0.267 | 0.406 |
| `mpisn` | 33.290 | 31.866 |

Every entry is inside the corresponding posterior 1σ.

### 7.2 The observed correlations are the predicted ones

From `runs/endO5_fullcosmo_evo3/O5_fullcosmo_evo3.nc` (2 chains × 1800 draws):

| pair | ρ |
|---|---|
| `log_flow` – `msigma_low` | +0.903 |
| `a` – `log_flow` | **−0.865** |
| `log_flow` – `mp_low` | −0.780 |
| `a` – `msigma_low` | **−0.774** |
| `a` – `mp_low` | +0.627 |
| `a` – `log_fpl` | −0.477 |

Eigen-decomposition of the correlation matrix of (`a`, `b`, `log_flow`,
`mp_low`, `msigma_low`, `beta`, `mpisn`, `dmbhmax`, `c`, `log_fpl`): the
largest eigenvalue is **3.807** out of a trace of 10, with eigenvector

```text
log_flow +0.488   mp_low -0.431   a -0.429   msigma_low +0.407
c +0.280   beta +0.263   mpisn -0.194   b +0.152   dmbhmax +0.130   log_fpl +0.035
```

Restricted to (`a`, `log_flow`, `msigma_low`), the top eigenvalue is 2.695 of
3 — **90% of the variance in one direction**, eigenvector
(−0.565, +0.593, +0.574).  Signs match the Fisher axis of §3 for `log_flow`,
`msigma_low`, `mp_low` and `log_fpl`; the posterior direction is ~3–4× steeper
because it mixes several Fisher axes and is evaluated ~1 unit away from truth.

### 7.3 The offset is prior pull along the flat direction

The truth (−0.9426) is 1.65 prior-σ *below* the prior mean of 2.35.  Moving
truth → posterior median:

| | Δ log prior |
|---|---|
| `a` | **+0.6690** |
| `b` | +0.0668 |
| `mp_low` | −0.0510 |
| `msigma_low` | −0.0387 |
| `mpisn` | −0.1380 |
| `dmbhmax` | −0.1320 |
| `c` | −0.0565 |
| `log_flow`, `log_fpl` (uniform) | 0 |
| **total** | **+0.3197** |

Against a log-likelihood cost of **0.230** (§2).  Net **+0.44 in favour of
`a` ≈ 0**.  The sampler is doing the arithmetically correct thing on a
likelihood that carries essentially no information about `a`.

The lower edge of the reported interval is prior-set too: relative to `a` = 0,
the prior penalty at `a` = −0.77 is 0.53 and at the floor −1.65 is 1.31, while
the likelihood cost is ~0.08 throughout.  The upper edge at +0.65 is the
likelihood wall.

### 7.4 What it is *not*

Not selection effects — those are in the KL, and including them sharpens the
picture rather than explaining it away.  Not model misspecification — the mock
was generated through the same module and defaults
(`scripts/reweight_res.py:12,321`; `mco_floor` default 6.0 at
`src/intensity_models_fast.py:1205`), so the generative and inference mass
functions are identical.

Not *primarily* the modest ESS either — but ESS matters for how the result is
phrased.  Bulk ESS is 14–140 (`a`: 22.4, r̂ = 1.118; `mpisndot`: 14.6;
`sigma`: 138), so the width 0.42 is probably under-estimated and the
"truth outside 95%" statement should be read as ~1σ-level evidence, not 2σ.
The robust part of §7.1 is the *location* agreement, not the interval.

## 8. In-flight confirmation runs

Two runs designed to test the σ(`a`)-vs-bump-width prediction end to end are
executing now.  **Their outcome is open; nothing below reports a result.**

| | `runs/endO5_broadbump` (Slurm 6791077) | `runs/endO5_narrowbump` (Slurm 6791202) |
|---|---|---|
| data | existing `endO5_val2`, symlinked | freshly reweighted mock |
| true `msigma_low` | 4 | **2** |
| prior | `runs/priors/gwtc5_massonly.prior` | same |
| free | mass function + `beta`, `lam`, `dkappa`, `zp` | same |
| pinned | cosmology, `mpisndot` = 0 | same |
| nobs | 9000 | 9000 |
| predicted σ(a) | ≈ 1.5 (prior-dominated) | ≈ 0.29 |

Everything else — prior, `nobs`, `n_pe` = 4000, 2 chains × 1800 draws,
injection pool — is identical between the two, so any difference in `a`
recovery is attributable to the bump width and nothing else.  Pinning the
cosmology and `mpisndot` is deliberate: this is a source-frame mass-function
question, and it removes the `h`/`Omh2`/`w` nuisance directions that held evo3
to bulk ESS ~15.

**What would confirm the analysis.**  `a` prior-dominated in `broadbump` (wide
posterior centred near the prior, truth not meaningfully excluded by the
likelihood) *and* `a` recovered in `narrowbump` with σ ≈ 0.3, at which width
the evo3 offset of ~0.95 would be a > 3σ effect.

**What would falsify it.**  `a` still lost at `msigma_low` = 2 — in which case
the problem is not the bump's breadth but something structural in the mass
model that this audit has mislocated.  Equally falsifying: `a` recovered
cleanly in `broadbump`, which would mean the evo3 failure was free
cosmology / `mpisndot` / poor mixing rather than the mass model.

**Why width 2 and not 1.**  Population draws come from rejection sampling
against a fixed 212,041,265-row injection pool
(`notes/2026-08-07-injection-pool-rejection-sampling.md`), so acceptance ≈
mean(w)/max(w) and a peakier target costs draws.  Predicted from a 40M-row
subsample and confirmed by two real reweights:

| true `msigma_low` | 4.0 | 3.0 | 2.5 | 2.0 | 1.5 | 1.0 |
|---|---|---|---|---|---|---|
| detections | 250,144 | 203,786 | 163,335 | 112,921 | 67,631 | 32,021 |
| max nobs (= nsel/4) | 31,268 | 25,473 | 20,417 | 14,115 | 8,454 | 4,003 |

`run_inf.py` uses half the selection rows and the model's `neff_sel` hinge is
at `4 * nobs`.  A first attempt at `msigma_low` = 1 gave 363,816 draws /
35,247 detections (predicted 365,543 / 32,021), leaving 17,624 usable selection
rows against a 36,000 hinge — the guard would have been permanently active, so
that run was cancelled.  Width 2 gave 113,299 detections and
neff_sel = 56,650, a 1.57× margin, while still sitting at Fisher σ(a) = 0.289.
The pool cannot be stretched further: `n_total` is already clamped to the
file's 212,041,265 rows, and reaching val2's draw count at width 1 would need
a ~14 billion row pool.

**Caveat on the pair.**  The narrow-bump mock has 2.2× noisier selection Monte
Carlo than val2 (113,299 vs 249,653 detections).  `neff_sel` clears the
threshold with margin, but the two runs are not matched in selection-MC
quality, and a difference in `a` recovery should be checked against
`min_neff` / `mc_var_loglike` / `neff_sel` before being attributed to the bump
width.  Second, `gwtc5_massonly.prior` keeps
`msigma_low = TruncatedNormal(4.0, 2.0, low=0.5, high=8.0)`, so in the
narrow-bump run the truth sits 1σ *below* the prior mean; any residual pull
toward a broad bump would work against recovering `a`, making that test
conservative rather than optimistic.

## 9. Recommendations

1. **Do not report `a` as measured in this configuration, and do not treat its
   offset as a bug.**  The evo3 posterior on `a` is prior times a one-sided
   likelihood wall.  Quote it as prior-dominated or profile it out.
   *Tradeoff:* none; this is the honest statement.

2. **Reparameterize away from the degenerate direction.**  Either sample the
   Fisher-null combination explicitly, or — more interpretable — sample the
   slope of `dN/dm` over a fixed observable window,
   `s ≡ d ln(dN/dm)/d ln m` at 15–20 M☉, and derive `a`.
   *Tradeoff:* loses the direct CO-IMF reading and needs a re-derived prior;
   but it turns a 30×-degraded parameter into a measured one and should
   materially improve the NUTS geometry (evo3 saturated `max_tree_depth`
   behaviour and reached ESS 14–140 largely because of this ridge).

3. **If tightening priors, tighten `log_flow` first.**  From the Fisher, with
   the run priors σ(a) = 1.50; `msigma_low` sd 0.5 → 1.22; `mp_low` sd 0.5 →
   1.46; **`log_flow` sd 0.3 → 0.88**; all three tight → 0.82.
   *Tradeoff:* an informative `log_flow` prior is hard to justify
   astrophysically, and even the aggressive combination only halves σ(a) —
   priors cannot rescue this.

4. **Consider capping `msigma_low`.**  Currently
   `TruncatedNormal(4, 2, low=0.5, high=8)`.  A Gaussian with σ = 4 at
   μ = 9.1 is not a bump; it is a second broad continuum with 78% of its mass
   in [6, 20] M☉ and 1.1% below zero mass.  Restricting to `msigma_low` ≲ 2.5,
   or reparameterizing as `msigma_low/mp_low ≲ 0.3`, makes `a` measurable at
   the 0.3 level.  *Tradeoff:* a real model restriction — if the true low-mass
   structure is broad, this biases `flow`/`mp_low` and pushes the excess into
   `a` instead.  Only do this if the narrow-bump interpretation is physically
   intended.

5. **Do not spend effort on `mco_floor`.**  Keep it at 6 for the
   integrability and quadrature reasons it was introduced for, but amend the
   docstring (`src/intensity_models_fast.py:596-604`) per §6.

6. **Use the profile, not the marginal, as the standing diagnostic.**  The
   N·KL profile of §2 costs ~5 minutes on CPU and is the right check to run
   before interpreting any shifted `a` in future mocks.

## 10. Caveats

- Both metrics assume **perfect per-event measurement** and **cosmology fixed
  at truth**.  The real run has 4000 PE samples per event and free `h`,
  `Omh2`, `w`.  Both relaxations only widen the posterior, so the compensation
  residuals here are a **lower bound on flatness** — the degeneracy is at
  least this bad.
- N·KL is the *expected* log-likelihood loss; a single realization fluctuates
  by O(√(2·N·KL)).  Irrelevant at N·KL = 0.23; at 2.8 it means "~2σ, not
  2.4σ".
- The Fisher σ(a) = 2.55 is a local quadratic at truth.  The true profile is
  asymmetric — flat down to the prior floor, walls above `a` ≈ +0.5 — so 2.55
  overstates the upward width.  **The refit profile of §2, not the Fisher σ, is
  the primary evidence.**
- Refits are L-BFGS-B from truth with box bounds equal to the prior supports.
  At large Δa (`a` ≥ 2) `b` pins at its −2.1 floor, so those rows are
  conservative; a wider `b` prior would lower them somewhat.
- Posterior correlations come from a chain with bulk ESS 14–140 and r̂ up to
  1.118 (`a`) over the sampled scalars.  They are indicative only.  The
  robust comparison is the *location* agreement of §7.1, not the exact
  correlation coefficients or the interval width.
- Small wiggles in the autodiff sensitivity curves (∂ ln P_pisn/∂a bounces
  between 0.18 and 1.07 near m = 4–4.5) are interpolation artifacts of the
  514-node log-`mbh` grid against the 0.2-M☉-wide `mco_min` turn-on.  They do
  not affect any integrated quantity, and the KL/Fisher use the same tables the
  sampler does.
- The σ(a)-vs-`msigma_low` and σ(a)-vs-`mco_floor` tables regenerate the *truth*
  at each setting and re-derive the Fisher there.  They answer "how well would
  `a` be measured in a universe like this", which is the design question; they
  are not statements about fitting the existing val2 data with a wrong floor or
  width.
- No tracked source file was modified by this audit; `runs/` was accessed
  read-only; no Slurm command other than reads was issued.

## Reproduction

All scratch scripts live in the session scratchpad, not the repo.  The four
computations are:

1. **Shape decomposition** — build `LogDNDM` at the
   `scripts/pop_configs/mock_O5_noevo.txt` truth, evaluate on a log grid over
   [3, 70] M☉, and take `jax.jacfwd` of `log dN/dm` w.r.t. each parameter.
2. **KL / Fisher** — read `runs/endO5_fullcosmo_evo3/sel_noevo.h5`
   (first half, 124,826 rows), form
   `log u_i(Λ) = LogDNDMDQDV(m1_i, q_i, z_i | Λ) + J_i − log pdraw_sel_i` with
   `J` from `FlatwCDMCosmology.z_and_log_jacobian` at truth, then
   `KL(Λ) = Σ w0_i (f_i(0) − f_i(Λ)) + logsumexp(log u(Λ)) − logsumexp(log u(0))`
   with `w0 ∝ u(truth)`.  Fisher = `jax.hessian(KL) * 9000`.
   Note that `KL` is invariant to rescaling `dN`, so `log_norm` need not be
   tracked.
3. **Compensation profile** — L-BFGS-B on `KL` at fixed `a`, jitting
   `value_and_grad` once as a function of `(a, y)` so all `a` values reuse one
   compile.
4. **Posterior** — `az.from_netcdf` on
   `runs/endO5_fullcosmo_evo3/O5_fullcosmo_evo3.nc`, read-only.

Run everything with `uv run python` and float64
(`jax.config.update("jax_enable_x64", True)`); CPU is sufficient (~5 min per
computation on 32 cores).

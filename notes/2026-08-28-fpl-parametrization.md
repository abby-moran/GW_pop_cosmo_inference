# The `fpl` railing and the power-law attachment

*2026-08-28.  Status: open question, parked for a later session.  Prompted by
the real-data runs `abbys_runs/GWTC5_gc_reparam_noevo` (fpl railed at its
cap of 2, 27.5% of draws at the bound) and `runs/realGWTC5_noevo_fullsel`
(cap widened to 10; fpl railed again, 27.2% at the bound, median 8.6).
All numbers below are from those two posteriors.*

## The concern

With `fpl` large, the ~$30\,M_\odot$ feature can be fit by the power-law
tail's smooth turn-on edge at $m_{\rm BH}^{\max}$ instead of the PISN
pile-up Gaussian — a degeneracy between the bump and the power-law break —
and a tail carrying most of the mixture weight sounds unlike standard LVK
results.  Should we go back to attaching the power law differently?

## What `fpl` is now, and why it changed

The mass function is a proper simplex mixture of three **unit-area**
components with weights $1 : f_{\rm low} : f_{\rm pl}$, normalized by
$\log(1 + f_{\rm low} + f_{\rm pl})$: (i) continuum + PISN pile-up
(truncated at $m_{\rm BH}^{\max}$), (ii) the low-mass bump, (iii) a
power-law tail on $[m_{\rm BH}^{\max}, \infty)$ with slope $c$, smoothly
turned on with a **hard-coded 5% fractional width** (`log_smooth_turnon`).

Before commit `e7ad354` (2026-07-21) the tail was *height-anchored*:
$p_{\rm tail}(m) = f_{\rm pl}\, \mathrm{mix}(m_{\rm BH}^{\max})\,
(m/m_{\rm BH}^{\max})^{-c}$, so $f_{\rm pl} = 1$ meant "tail starts as tall
as the continuum at the join" — scale-free, with a natural $O(1)$ prior
range.  That scheme was removed because the total integral of the mass
function drifted with $f_{\rm low}$, $z$, and the shape parameters; the
drifting normalization was silently absorbed into other hyperparameters and
**broke MCMC recovery of injected populations** (see
`notes/2026-08-06-join-point-machinery-removed.md`).  The simplex mixture
fixed that.  But the prior cap $\log f_{\rm pl} < \ln 2$ — sensible under
height-anchoring ("tail at most twice the continuum height") — survived the
semantics change unexamined.  Under the simplex, the natural scale of
$f_{\rm pl}$ is the *area ratio* of tail to continuum, which real data put
well above 2.  **The cap is a fossil; the railing is its symptom.**

## Findings (Abby's run, cap = 2)

1. **The degeneracy is real but is a reparametrization ridge, not two
   physical solutions.**  Strongest coupling: $\log f_{\rm pl}$ vs
   $m_{\rm BH}^{\max}$ ($r = -0.50$), then `mpisn_ref` ($r = -0.36$).  The
   ridge is one-sided and *saturating*: conditional medians of
   `mpisn_ref`/$m_{\rm BH}^{\max}$ slide down $\sim 2\,M_\odot$ with rising
   $f_{\rm pl}$, then flatline; in the high-$f_{\rm pl}$ half the
   correlations vanish ($r \approx 0$).  No multimodality.
2. **Component swap at the peak — the suspicion is mechanically correct.**
   The tail supplies $\sim 70\%$ of the density *at* the $\sim 31\,M_\odot$
   peak for the highest-$f_{\rm pl}$ draws vs $\sim 36\%$ for the lowest:
   at high $f_{\rm pl}$ the "PISN feature" is mostly the turn-on edge, not
   the Gaussian.  Correspondingly `sigma`, `dmbhmax` are prior-dominated
   (posterior/prior sd 0.55, 0.92) — **at all $f_{\rm pl}$**, so widening
   didn't newly destroy them.
3. **The physical spectrum is invariant along the ridge.**  The
   $q$-integrated $m_1$ marginal is statistically identical between the
   lowest-300 and highest-300 $f_{\rm pl}$ draws (e.g. $f(m_1 > 40)$:
   4.92% vs 4.91%; high-mass log-slopes inside each other's 90% CIs).
4. **The data genuinely want $f_{\rm pl} > 2$**: the data-fit term climbs
   monotonically $\sim 3$ nats across $f_{\rm pl}$ deciles and is still
   rising at the old cap.  Widening was the right call for the noevo run.
5. **One real physical change along the ridge:** the feature's *asymmetry*
   flips.  Low $f_{\rm pl}$: sharp **falling** edge (truncated pile-up,
   log-slope to $-10$).  High $f_{\rm pl}$: shallow fall ($\approx -c$)
   with the sharpness relocated to the **rising** edge — whose fractional
   width is the hard-coded 5% constant.  Note `sigma`'s prior floor is also
   0.05, so *both* branches have a model-imposed minimum feature sharpness
   of $\sim 5\%$ ($\sim 1.5\,M_\odot$ at $31\,M_\odot$).

## LVK comparison: normal spectrum, unusual coordinates

The simplex weights live on unit-area components with different supports,
and the pairing $\beta \approx -3.1$ does the job of LVK's $\alpha$, so
$w_{\rm pl} = 0.5$–$0.85$ is bookkeeping, not astrophysics.  Translated to
observables (Abby's posterior, $z \approx 0$ marginal):
$\mathrm{d}N/\mathrm{d}m_1(35)/\mathrm{d}N/\mathrm{d}m_1(20) = 0.89$
$[0.66, 1.22]$; $f(m_1 > 45) = 3.0\%$ $[2.0, 4.1]$; log-slope over
$40$–$80\,M_\odot$ of $-5.3$; peaks at $\sim 9.5$ and $\sim 31\,M_\odot$ —
all consistent with GWTC-3 PLP at the 1–2× level.  The only mild tension,
$f(>45)$ at the top of LVK's model-systematics range, goes in the direction
O4-era catalogs have moved anyway.

## The new run confirms the structural problem

`realGWTC5_noevo_fullsel` (full 488839-row selection set, cap = 10):
$f_{\rm pl}$ median 8.6, **27.2% of draws within 1% of the $\ln 10$ bound**
— the same pile-up fraction as at $\ln 2$.  The likelihood does not turn
over between 2 and 10.  The ridge drift came along: `mpisn_ref`
$27.5 \to 25.5$, $m_{\rm BH}^{\max}$ $31 \to 28$ vs Abby's run.  All four
chains agree at the rail (not a stuck-chain artifact; $\hat R \le 1.005$).
Per the earlier decision rule this is the **structural red flag: do not
widen again** — a ratio has no natural upper edge.
(Separately: the full selection set moved `neff_sel`'s median past the
$4 N_{\rm obs}$ hinge, but its minimum still dips below — the guard is
mildly active in the heavy-tail corner.)

## Recommendations (ranked)

1. **Do not revert to height-anchoring.**  It doesn't fix the shape
   degeneracy (pile-up vs turn-on edge is a shape swap, not amplitude
   bookkeeping) and it reintroduced-by-construction the normalization drift
   that demonstrably biased injection recovery.
2. **Reparametrize the tail weight** instead of capping the ratio:
   stick-breaking on the simplex weight, e.g. sample
   $\mathrm{logit}\, w_{\rm pl}$ with
   $w_{\rm pl} = f_{\rm pl}/(1 + f_{\rm low} + f_{\rm pl}) \in (0, 1)$, and
   make $f_{\rm pl}$ a deterministic; or at minimum replace the hard
   uniform edge with a soft prior (e.g. $\log f_{\rm pl} \sim
   \mathcal{N}(0, 1.5)$).  Caution: the existing `logit_fpl` template in
   `intensity_models_fast.py` maps $f_{\rm pl}$ itself into $(0,1)$ — that
   is *tighter* than the old cap; the logit must act on $w_{\rm pl}$.
3. **Fix the cosmology priors before any cosmology run — this is the
   actionable bug.**  `gwtc5_cosmo.prior`, `real_dat_cosmo*.prior`, and the
   evolution variants all still cap $\log f_{\rm pl}$ at $\ln 2$.  Real
   data will rail there, and the ridge's compensating drift
   ($\sim 2\,M_\odot$ in `mpisn_ref` and $m_{\rm BH}^{\max}$) is exactly
   the source-frame mass scale the standard-siren measurement leans on: a
   cap-induced offset in the feature location propagates into $H(z)$.
4. **Audit the hard-coded 5% turn-on width** (`log_smooth_turnon`) before
   trusting cosmology error bars: on the high-$f_{\rm pl}$ branch the
   feature's sharp side is model-fixed, mirroring `sigma`'s floor on the
   other branch.  Feature *location* stays data-driven (good for $H_0$
   centrals); feature *sharpness* — which sets the precision — is
   floor-limited on both branches.  Cheap mock sensitivity test: width
   $0.05 \to 0.10$.  Consider tying the turn-on width to `sigma` so one
   sampled parameter controls sharpness on both branches.

**Strongest counterargument to (2)+(3):** the $\ln 2$ cap in the cosmology
priors may act as an accidental regularizer keeping the fit on the pile-up
branch, where the feature's sharpness is honest (sampled via `sigma`)
rather than model-fixed.  Uncapping on real data will likely move the
posterior onto the turn-on-edge branch, which could either degrade the
cosmology constraint or make it spuriously stable.  If that trade is real,
the answer is still not the fossil cap — it is doing (2) and (4)
*together*, so that the sharpness is sampled on both branches.

## Status / next steps

- `runs/realGWTC5_noevo_fullsel` is mechanically clean (max $\hat R$ 1.005,
  1 divergence, healthy BFMI) but `mpisn`, `sigma`, `dmbhmax` should not be
  quoted from it: the fit sits on the turn-on-edge branch against the
  $\ln 10$ rail.
- Pending decision: implement stick-breaking $w_{\rm pl}$ (+ optional
  sigma-tied turn-on width), then rerun noevo and revisit the cosmology
  priors.

# Height-capped power-law tail: two flavors of the fix

*2026-08-29.  Status: design note, no code changed yet.  Follow-up to
`notes/2026-08-28-fpl-parametrization.md` and the re-evaluation session of
2026-08-29: the tail's turn-on edge can currently rise above the continuum
and impersonate the PISN pile-up (60–76% of the density at the ~31 M_sun
peak in the highest-`fpl` draws of `abbys_runs/GWTC5_gc_reparam_noevo`).
This note works out the mathematics of the proposed fix — anchor the tail
height to the continuum at the join and cap it — in its two flavors, and
where they are and are not equivalent.*

## Notation and current model

Let $m$ be source-frame primary/secondary BH mass, $q$ the mass ratio, $z$
redshift.  The mass function (`LogDNDM.__call__`,
`src/intensity_models.py:177-212`; identically
`intensity_models_fast.py:678-703`) is a simplex mixture of three shapes:

$$p(m \mid z) \;=\; \frac{p_{\rm pisn}(m \mid z) \;+\; f_{\rm low}\, p_{\rm low}(m) \;+\; f_{\rm pl}\, p_{\rm tail}(m \mid z)}{1 + f_{\rm low} + f_{\rm pl}},$$

- $p_{\rm pisn}$: tabulated PISN continuum + pile-up, normalized to unit
  area per $z$ (`:180-182`), truncated above $m_{\rm BH}^{\max}(z)$;
- $p_{\rm low}$: unit Gaussian bump at $m_{p,\rm low}$, width
  $\sigma_{\rm low}$ (`log_normalized_gaussian`, `:71-72`);
- $p_{\rm tail}(m \mid z) = \frac{c-1}{m_j}\left(\frac{m}{m_j}\right)^{-c} s(m)$,
  the closed-form-normalized power law (`log_normalized_power_law_tail`,
  `:75-76`) times the smooth turn-on sigmoid (`log_smooth_turnon`, `:44-46`)
  $$s(m) = \left[1 + e^{-(m - m_j)/(w\, m_j)}\right]^{-1}, \qquad w = 0.05 \text{ (hard-coded)},$$
  where $m_j \equiv m_{\rm BH}^{\max}(z) = m_{\rm PISN} + \dot m_{\rm PISN}\,\tfrac{z}{1+z} + \Delta m_{\rm BH}^{\max}$
  is the join point.  $c > 1$ is the tail slope.  The slow module zeroes the
  tail below $m_j$ (`:186`); the fast module's `smooth_tail_edge=True`
  default keeps the sigmoid-only suppression (`intensity_models_fast.py:684-687`).

$f_{\rm low}, f_{\rm pl}$ are sampled *area ratios* relative to the
continuum's weight of 1.  Nothing bounds the tail's **height**: for
$f_{\rm pl} \gtrsim 1$ the turn-on edge at $m_j$ becomes the tallest feature
of the spectrum.

## The height-anchored constructions

arXiv:2312.03973 (Eq. 6) attaches the tail as
$\delta(m \mid M_{\rm BH,max})\, f_{\rm pl}\, \frac{dN}{dm}\big|_{M_{\rm BH,max}} (m/M_{\rm BH,max})^{-c}$
with prior $f_{\rm pl} \in [0.01, 0.5]$: the tail's height is *defined* as a
fraction of the continuum's height at the join, and the prior guarantees it
stays subdominant there.  The pre-`e7ad354` code (`git show
e7ad354^:src/intensity_models.py`, `join_point_terms` + `__call__`) was the
same construction:

$$p_{\rm tail}^{\rm old}(m \mid z) = f_{\rm pl}\, \mu_j(z)\, \left(\frac{m}{m_j}\right)^{-c} s(m), \qquad \mu_j(z) \equiv p_{\rm pisn}(m_j \mid z) + f_{\rm low}\, p_{\rm low}(m_j),$$

where $\mu_j$ ("`log_mix_at_join`") is the height of the rest of the
mixture at the join.  Since $s(m_j) = 1/2$, the density ratio
tail/continuum is exactly $f_{\rm pl}/2$ *at* $m_j$, and the tail's global
maximum is $\approx 0.7\, f_{\rm pl}\, \mu_j$ (weakly $c$-dependent,
reached at $m \approx 1.1\, m_j$).  So the old cap $f_{\rm pl} \le 2$
bounded the edge at roughly the continuum height; the paper's
$f_{\rm pl} \le 0.5$ keeps it strictly subdominant.

## The tail's analytic area and $\kappa(c)$

For the pure power law, $\int_{m_j}^\infty (m/m_j)^{-c}\, dm = m_j/(c-1)$
for $c > 1$.  The sigmoid perturbs this by a factor that depends only on
$c$ (given the fixed $w = 0.05$ and the edge convention):

$$\kappa(c) \;\equiv\; \frac{(c-1)}{m_j} \int (m/m_j)^{-c}\, s(m)\, dm \;=\; (c-1) \int x^{-c}\, \tilde s(x)\, dx, \qquad \tilde s(x) = \left[1 + e^{-(x-1)/w}\right]^{-1},$$

integrated over $x \ge 1$ (hard edge, slow module) or $x \ge m_{\rm BH}^{\min}/m_j$
(smooth edge, fast module; the integral to $x = 0$ diverges for $c \ge 1$
because $\tilde s \to e^{-1/w}$ saturates, but the `mbh_min` window cuts it
and the residual dependence on $m_{\rm BH}^{\min}/m_j$ is negligible).
Numerically ($w = 0.05$):

| $c$ | 1.5 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $\kappa$ hard edge | 0.984 | 0.969 | 0.941 | 0.916 | 0.893 |
| $\kappa$ smooth edge ($x_0 = 0.1$) | 1.003 | 1.009 | 1.026 | 1.055 | 1.095 |

Compute once by quadrature on a grid in $c$ and `jnp.interp` (or evaluate
per $z$ on the existing `mbh_grid` with `log_trapz_grid`).  Note in passing:
the current code's "unit-area" tail actually has area $\kappa(c)$ — a benign
$z$-independent wrinkle of the same kind discussed below.

Anchored tail area, exactly:

$$A_{\rm tail}(z) \;=\; \int p_{\rm tail}^{\rm old}\, dm \;=\; r\, \mu_j(z)\, \frac{m_j(z)}{c-1}\, \kappa(c) \;\equiv\; \kappa(c)\, f_{\rm eff}(z), \qquad f_{\rm eff}(z) \equiv r\, \mu_j(z)\, \frac{m_j(z)}{c-1},$$

where from here on $r$ denotes the sampled height ratio (the old
$f_{\rm pl}$'s role) and $f_{\rm eff}$ is the equivalent simplex area-ratio.

## Flavor (a): anchored tail + per-$z$ normalization

$$p_a(m \mid z) \;=\; \frac{p_{\rm pisn}(m \mid z) + f_{\rm low}\, p_{\rm low}(m) + r\, \mu_j(z)\, (m/m_j)^{-c}\, s(m)}{1 + f_{\rm low} + A_{\rm tail}(z)}.$$

Unit total area at every $z$ by construction (up to the `mbh_min` window,
exactly as now), and the tail/continuum height ratio at the join is
$r\,s(m) \le r$ at every $z$.  Cost: one grid interpolation for
$\mu_j(z)$ (the deleted `join_point_terms`) plus $\kappa(c)$, both per
$z$-evaluation, negligible and smooth.

## Flavor (b): keep the simplex, make $f_{\rm pl}$ deterministic

Keep `__call__` untouched and set, at a reference redshift $z_{\rm ref}$
(the existing `zref = 0.001`),

$$f_{\rm pl} \;:=\; f_{\rm eff}(z_{\rm ref}) \;=\; r\, \mu_j(z_{\rm ref})\, \frac{m_j(z_{\rm ref})}{c-1} \qquad \text{(a scalar; sample } r \text{, record } f_{\rm pl} \text{ as a deterministic)}.$$

(Using $A_{\rm tail} = \kappa f_{\rm eff}$ instead changes the join-height
ratio to $\kappa r$ — the $f_{\rm eff}$ form keeps $r$'s height semantics
exact; the difference is a constant absorbed as below.)

## Where the flavors agree and differ

**noevo ($\dot m_{\rm PISN} = 0$): algebraically identical posteriors.**
The PISN grid, $\mu_j$, $m_j$ are $z$-independent, so
$f_{\rm eff}(z) = f_{\rm eff}(z_{\rm ref})$ for all $z$ and
$$\frac{p_a(m \mid z)}{p_b(m \mid z)} \;=\; \frac{1 + f_{\rm low} + f_{\rm eff}}{1 + f_{\rm low} + \kappa\, f_{\rm eff}},$$
a hyperparameter-dependent but $m$-, $z$-, data-independent constant.  The
likelihood is $R$-marginalized: a constant $C(\Lambda)$ added to
$\log dN$ adds $n_{\rm obs}\, C$ to the event terms
(`src/intensity_models.py:456-463`) and $-n_{\rm obs}\, C$ via the
selection factor $-n_{\rm obs} \log \mu_{\rm sel}$ (`:469-472`); it cancels
exactly, `neff_sel` (`:483`) is invariant, and $R = n_{\rm obs}/\mu_{\rm sel}$
(`:489`) compensates so all rate deterministics are invariant too.  This is
also why the normalization drift that motivated `e7ad354` cannot bite
either flavor: only *$z$-dependent* normalization survives the
cancellation, and in noevo there is none.

**Evolving case ($\dot m_{\rm PISN} \ne 0$): they genuinely differ.**  In
(b) the scalar weight makes the actual join-height ratio at redshift $z$
$$r(z) = r \cdot \frac{\mu_j(z_{\rm ref})\, m_j(z_{\rm ref})}{\mu_j(z)\, m_j(z)},$$
so the cap is exact only at $z_{\rm ref}$ and drifts with $z$ (through the
evolving join location and pile-up height; plausibly $O(10\%)$ for
GWTC-scale $\dot m_{\rm PISN}$, quantifiable per run from the existing 30-point
$z$ grids).  Flavor (b) remains a perfectly well-normalized model — the
area mismatch is the same constant as above — but $r$'s *meaning* and the
structural cap degrade away from $z_{\rm ref}$.  Flavor (a) keeps both
exact at every $z$; its per-$z$ denominator is the piece that handles the
one normalization component that does not cancel.

## Prior on $r$

$r \in (0, r_{\max}]$; physically, $r$ is the tail's height at the join as
a fraction of the continuum height there (limit of completed turn-on; the
ratio is $r/2$ at $m_j$ itself, and the tail's maximum is
$\approx 0.7\,r\,\mu_j$).  Candidates: $r_{\max} = 0.5$ (the paper's cap;
tail strictly subdominant everywhere near the join) or $r_{\max} = 1$
("tail may just reach the continuum").  Uniform in $r$ vs log-uniform
(the old prior was log-uniform on $[10^{-2}, 2]$): log-uniform preserves
the old behavior near 0; uniform puts more mass at the cap.  For the
highest-$f_{\rm pl}$ draws of the current posterior the realized join
ratio is 1.35–2.0, i.e. outside either cap — that branch is excluded by
construction, which is the point.

## Implementation deltas

- **Both flavors:** resurrect the join-height lookup from `e7ad354^`
  (`join_point_terms`: interp of the PISN grid at $(m_j(z), z)$ minus
  `log_Z_pisn_at_z`, plus $f_{\rm low}$ times the bump — the bump term is
  numerically nil at $m_j \sim 30$ but keep it for fidelity).  The fast
  module already has `mbhmax_at_z`, `_interp_from_log`, `_log_Z_from_z`
  (`intensity_models_fast.py:653-667`).  Prior files: replace
  `log_fpl = Uniform(...)` with `r` (or `log_r`); map it in
  `get_deterministic_parameters` (`intensity_models.py:401-408`,
  `intensity_models_fast.py:994-1001`) and record
  `numpyro.deterministic('fpl', ...)` so downstream plotting keeps working.
  Caution: the existing `logit_fpl` template (fast `:994-995`) squashes
  into $(0,1)$ via sigmoid — reuse only deliberately for $r_{\max} = 1$.
  Selection-side consistency (CLAUDE.md rule) is automatic: everything
  lives inside `LogDNDM`, which both the event and selection terms call.
  Update `scripts/test_fast_equivalence.py`.
- **Flavor (a) only:** change the mixture block (`intensity_models.py:190-208`,
  fast `call_from_logs:678-703`): tail term becomes
  $\log r + \log \mu_j(z) - c \log(m/m_j) + \log s$, denominator becomes
  $\mathrm{log1p}(f_{\rm low} + A_{\rm tail}(z))$; add the $\kappa(c)$
  table.
- **Flavor (b) only:** no `__call__` change.  Compute
  $f_{\rm eff}(z_{\rm ref})$ inside `LogDNDM.__post_init__` after
  `setup_interp` (the grid exists by then and $\mu_j$ does not depend on
  $f_{\rm pl}$), or in the model-building layer.
- `fpl` values in old run outputs are area-ratios and not comparable to $r$.

## Open choices (not recommendations)

1. $r_{\max}$: 0.5 (paper-faithful, strictly subdominant) vs 1.
2. Prior shape: uniform vs log-uniform in $r$.
3. Flavor (a) vs (b): (b) is a smaller diff and exactly equivalent for the
   noevo program; (a) is the only one whose cap and semantics survive
   $\dot m_{\rm PISN} \ne 0$, which the cosmology runs use.
4. Whether to also promote the turn-on width $w$ from a hard-coded 0.05 to
   a sampled parameter (or tie it to $\sigma$) — orthogonal to the cap, but
   it is what makes the residual just-above-join shape honest.

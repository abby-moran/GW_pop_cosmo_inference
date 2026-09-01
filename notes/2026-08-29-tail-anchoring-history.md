# History of the tail anchoring vs. mass-scale evolution

*2026-08-29.  Status: historical record, no code implications by itself.
Companion to `notes/2026-08-29-height-capped-tail-parametrization.md`
(the flavor-(a)/(b) design note) and
`notes/2026-08-28-fpl-parametrization.md` (why the tail attachment is
being reconsidered at all).  Dug up because the preferred fix anchors the
tail's mixture weight at a reference redshift while the PISN continuum
evolves, and we wanted to know how the original study handled that
interaction.  Terminology as in the design note: **per-z anchoring** means
the tail's amplitude is tied to the continuum height at the join
$\mu_j(z)$ evaluated at each sample's own redshift (height ratio constant
in $z$); **reference-z anchoring** means the amplitude is fixed from the
join height at a single $z_{\rm ref}$, so the height ratio evolves with
$z$ (flavor (b)).*

## The paper (arXiv:2312.03973v3)

Eq. 6 attaches the tail as
$\delta(m \mid M_{\rm BH,max})\, f_{\rm pl}\, \frac{dN}{dm_{1G}}\big|_{M_{\rm BH,max}} (m/M_{\rm BH,max})^{-c}$
with no redshift subscripts anywhere.  Section III.4 introduces the mass-scale
evolution "$M_{\rm tr}(z) = M_{\rm tr}(z{=}0) + \dot M_{\rm tr}(1 - 1/(1+z))$"
with "$M_{\rm BH,max}(z) - M_{\rm tr}(z) = {\rm const}$", but never mentions
$f_{\rm pl}$, the 2G tail, or the power law — the text is silent on whether
the anchor is re-evaluated per $z$.  The tail's physical motivation is
hierarchical (2G) mergers, assumed to "only subdominantly contribute";
no argument is given anywhere for how the 2G amplitude should scale with
$z$ relative to the 1G pile-up.  Prior: $f_{\rm pl}$ log-uniform with a
height-ratio cap of 0.5 (code lower bound $10^{-3}$; the text quotes 0.01).

## The public code

`github.com/jacobgolomb/BumpCosmology`, a fork of `farr/BumpCosmology`
(showyourwork article, `ms.tex` author Will M. Farr) — unmistakably the
ancestor of this repository.  Caveat: all public branch tips predate the
arXiv submission (fork `main` 2023-06-09; upstream `main` 2023-09-26), so
this is development-stage code, not the final paper code.  `main` has a
fully $z$-independent mass function (`LogDNDM.__call__(m)`, no
`mpisndot`), so the question cannot arise there.  Branch `will-changes`
has `LogDNDM_evolve` with **live per-z anchoring**, the **identical**
logistic taper `log_smooth_turnon(m, mmin, width=0.05)` (hard-coded 5%
fractional width) this repo inherited, and rate-free normalization (no
area integral anywhere).

## Commit archaeology (full clone, both remotes, all branches)

- **`4ad8428` (2023-06-13, Golomb, "big updates, fixed jacobian, ...")** —
  `LogDNDM_evolve` is born with **live reference-z anchoring**, and the
  tail's break location frozen too:

  ```python
  self.log_pl_norm = jnp.log(self.fpl) + self.interp_2d(self.mbhmax, self.zref)   # zref = 0
  ...
  log_dNdm = jnp.logaddexp(log_dNdm, -self.c*jnp.log(m/self.mbhmax) + self.log_pl_norm + log_smooth_turnon(m, self.mbhmax))
  ```

  At this commit only `mpisn` evolved; `mbhmax` was still a scalar.
- **`72df719` (2023-06-20, "changed dmbhmax to vary with z")** —
  $m_{\rm BH}^{\max}(z)$ starts evolving in the PISN grid while the tail
  stays pinned at the $z=0$ break with the $z=0$ anchor height: an
  internally inconsistent hybrid that persisted through August.
- **June–August 2023** — run-and-debug traffic on the evolve model, which
  was wired into the numpyro inference model from birth
  (`4ad8428:...:545`): "changed prior on mpisndot" (`7e984ff`),
  nan_to_num/gradient fixes (`145e249`, `67dbc80`), "changed some prior"
  (`9f8607a`), "changed to rejection sampling" (`b205277`), six "debug"
  commits (Jul 10), "fixed normalization?" (`8e3e9de`).  No posterior
  artifacts are committed, but prior tuning and NaN/gradient fixes imply
  actual sampling runs under reference-z.
- **`f341f7e` (2023-09-01, Golomb, "pushing several changes")** — the
  switch to per-z, with an *upgraded* reference-z variant (correct evolved
  `mbhmax_at_zref`) written at the same moment but committed already
  commented out:

  ```
  -        self.log_pl_norm = jnp.log(self.fpl) + self.interp_2d_dndmpisn(self.mbhmax, self.zref)
  +        #mpisn_at_zref = self.mpisn + self.mpisndot * (1 - 1/(1 + self.zref))
  +        #mbhmax_at_zref = mpisn_at_zref + self.dmbhmax
  +       # self.log_pl_norm = jnp.log(self.fpl) + self.interp_2d_dndmpisn(mbhmax_at_zref, self.zref)
  ...
  +        mbhmax_at_samples = jnp.interp(z, xp = self.z_array, fp = self.mbhmaxs)
  +        log_dNdmbhmax_at_samples = self.interp_2d_dndmpisn(mbhmax_at_samples, z)
  +        log_dNdm = jnp.logaddexp(log_dNdm, jnp.log(self.fpl) + -self.c*jnp.log(m/mbhmax_at_samples) + log_dNdmbhmax_at_samples + log_smooth_turnon(m, mbhmax_at_samples))
  ```
- **`33b08f1` (2023-12-22)** deletes the commented block.  It was never
  uncommented on any branch of either remote, through the most recent tip
  `origin/reformat` (2025-05-29) — the evident ancestor of this repo's
  initial commit, whose entire pre-`e7ad354` history (see
  `git show e7ad354^:src/intensity_models.py`, `join_point_terms`) is
  likewise per-z.

## Conclusions (interpretation flagged where it is interpretation)

(i) Reference-z anchoring was the *original* semantics of the evolve model
and demonstrably ran (June–August 2023).  (ii) Everything of publication
vintage — the paper repo from Sep 2023 onward and this repo's entire
pre-`e7ad354` history — used per-z anchoring.  (iii) The switch was
deliberate (the consistent reference-z alternative was written down at the
moment of the switch and left disabled) but **undocumented**: no commit
message, comment, or text records a reason.  (iv) *Interpretation:* the
June reference-z version was **not** flavor (b) — it froze the break
location as well as the amplitude, which became inconsistent once
$m_{\rm BH}^{\max}(z)$ evolved, and cleaning up that inconsistency is a
plausible alternative explanation for the September switch; the commented
Sep-2023 block is the closest historical match to flavor (b), and in their
rate-free, non-simplex model it is effectively equivalent to it.  (v) Net:
no source documents an argument *against* reference-z / flavor (b);
adopting it departs from historical practice but contradicts nothing on
record.  Will Farr is the person who might remember the actual reasoning.

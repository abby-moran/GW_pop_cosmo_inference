# Implementation plan: `tail_anchor` switch (simplex / ref_z / per_z)

*2026-08-29.  Status: plan, not yet implemented.  Self-contained recipe for an
implementer.  Rationale lives in the companion notes and is not repeated here:
`notes/2026-08-28-fpl-parametrization.md` (why the tail attachment is being
reconsidered), `notes/2026-08-29-height-capped-tail-parametrization.md`
(the mathematics of the two flavors; symbol definitions), and
`notes/2026-08-29-tail-anchoring-history.md` (what the paper and its code did).*

## 1. Goal & scope

Add a three-mode static switch `tail_anchor` to both population modules:

- `"simplex"` (default): current behavior, bit-identical — `fpl` is the sampled
  area-ratio mixture weight.
- `"ref_z"` (flavor (b)): sample the join-height ratio $r$; the tail's scalar
  simplex weight is the deterministic
  $f_{\rm eff}(z_{\rm ref}) = r\,\mu_j(z_{\rm ref})\,m_j(z_{\rm ref})/(c-1)$.
- `"per_z"` (flavor (a)): sample $r$; the tail is height-anchored at every $z$
  with per-$z$ denominator $\mathrm{log1p}(f_{\rm low} + f_{\rm eff}(z))$.

Here $m_j(z) = m_{\rm pisn} + \dot m_{\rm pisn}\,z/(1+z) + \Delta m_{\rm bhmax}$
is the join point and $\mu_j(z)$ the mixture (continuum + bump) height there.
$r$ is sampled via `log_r`; `fpl` is recorded as a `numpyro.deterministic` in
all modes so downstream scripts keep working.  $\kappa(c)$ is **deferred**.

## 2. Design decisions locked in

- Static string field threaded exactly like `smooth_tail_edge`
  (`intensity_models_fast.py:606`, ini key at `run_inf.py:91`, kwarg at
  `run_inf.py:194` → `pop_cosmo_model` → `build_population_model` → `LogDNDM`).
- `"simplex"` default ⇒ every existing config, run, and test is bit-identical.
- $\kappa(c)$ dropped in v1: it is a $z$-independent constant given the fixed
  turn-on width, so it cancels exactly in the R-marginalized likelihood
  (`src/intensity_models.py:456-472`); revisit only for exact reported fractions.
- $f_{\rm eff}$ convention (no $\kappa$) so $r$ keeps exact join-height
  semantics: tail/continuum density ratio $= r\,s(m)$ at the join, $s(m_j)=1/2$.
- The sampled amplitude travels in the existing `fpl` constructor slot in both
  modules (interpreted per `tail_anchor`), so `_mass_params` and every
  constructor signature stay single-amplitude.

## 3. Ordered implementation steps

Line anchors verified against the working tree on 2026-08-29.

**Step 1 — shared join-height helper, slow module** (`src/intensity_models.py`).
Add to `LogDNDM` (near `log_Z_pisn_at_z`, :174-175) the resurrected
`join_point_terms` logic from `git show e7ad354^:src/intensity_models.py`
adapted to the current class:

```python
def log_mix_at_join(self, mj, z):
    log_pisn = jnp.where(mj >= self.log_dndm_pisn.mbh_grid[-1], -np.inf,
                         self.interp_2d_dndmpisn(mj, z) - self.log_Z_pisn_at_z(z))
    if not self.use_low_bump:
        return log_pisn
    return jnp.logaddexp(log_pisn,
        safe_log(self.flow) + log_normalized_gaussian(mj, self.mp_low, self.msigma_low))
```

Check: unit-evaluate at a hand-checkable point (bump term is ~0 at $m_j\sim30$,
so it should match `interp_2d_dndmpisn - log_Z` to float tolerance).

**Step 2 — slow `LogDNDM`.**  Fields (:131-151): add
`tail_anchor: str = "simplex"` next to `use_low_bump` (:150).  In
`__post_init__` (:153-155), after `setup_interp()`:

```python
if self.tail_anchor == "ref_z":
    zr = jnp.asarray(self.zref)
    mj = self.mpisn + self.mpisndot*(1 - 1/(1+zr)) + self.dmbhmax
    self.fpl = self.fpl * jnp.exp(self.log_mix_at_join(mj, zr)) * mj / (self.c - 1)
```

(incoming `self.fpl` holds $r$; after this line it is $f_{\rm eff}(z_{\rm ref})$
and the simplex `__call__` path needs no change for ref_z).  In `__call__`
(mixture block :190-208), add the per_z branch: replace the scalar
`log_denom`/`log_w_pl` with

```python
log_f_eff = safe_log(self.fpl) + self.log_mix_at_join(mbhmax_at_samples, z) \
            + jnp.log(mbhmax_at_samples) - jnp.log(self.c - 1)
log_denom = jnp.log1p(self.flow + jnp.exp(log_f_eff))   # per-z arrays
log_w_pl  = log_f_eff - log_denom
```

with `log_w_pisn = -log_denom`, `log_w_low = safe_log(self.flow) - log_denom`,
tail *shape* unchanged (`log_p_pl`, :186-188).  Mirror in the
`use_low_bump=False` branch (:201-208).  Thread the field through slow
`LogDNDMDQDV` (fields :243-267, constructor call :271-274),
`build_population_model` (≈:419-428), and slow `pop_cosmo_model`
(signature :430, `build_population_model` call :448).
Check: `tail_anchor="simplex"` output bit-identical to current code.

**Step 3 — slow deterministic mapping** (`get_deterministic_parameters`,
fpl block :401-408).  Before the existing chain, add:

```python
if 'log_r' in sample or 'r' in sample:
    r = sample.get('r', None)
    if r is None:
        r = numpyro.deterministic('r', jnp.exp(sample['log_r']))
    out['fpl'] = r      # amplitude slot carries r in ref_z/per_z modes
```

(`fpl` as a *deterministic weight* is recorded in Step 6.)  Caution from the
design note: the existing `logit_fpl` template maps `fpl` itself into $(0,1)$
via sigmoid (:401-402, fast :994-995) — do not reuse it for `r` unless
$r_{\max}=1$ is chosen deliberately.

**Step 4 — fast `LogDNDM`** (`src/intensity_models_fast.py`).  Fields
(:583-608): add `tail_anchor: str = "simplex"` next to `smooth_tail_edge`
(:606).  Add the fast `log_mix_at_join` from existing helpers
(`_interp_from_log` :646-651, `_log_Z_from_z` :657-661):

```python
def log_mix_at_join(self, mj, z, log1p_z):
    log_pisn = jnp.where(mj >= self.mbh_grid[-1], -jnp.inf,
        self._interp_from_log(jnp.log(mj), z, log1p_z) - self._log_Z_from_z(z, log1p_z))
    ...  # bump term as in Step 1
```

ref_z branch in `__post_init__` (:609-611, after `setup_interp`; use
`zr = self.zref`, `log1p_z = jnp.log1p(zr)`).  per_z branch in
`call_from_logs` (:678-703): `mbhmax_at_samples` is already an argument;
compute `log_f_eff` per sample (one extra 2D gather, shared by the m1/m2
mixture weights since they share $z$) and use the per-$z$ denominator as in
Step 2.  Thread the field through fast `LogDNDMDQDV` (fields :743-764,
constructor :766-772), `build_population_model` (:1028-1035), and
`pop_cosmo_model` signature (:1065-1068).

**Step 5 — fast tabulated path** (this is the production path:
`tabulate_mass_function`/`tabulate_selection` default on, :1131-1134).
In `_build_mass_table` (:1152-1165) pass `tail_anchor=ld.tail_anchor` into the
inner `LogDNDM(...)` constructor (alongside `use_low_bump`/`smooth_tail_edge`,
:1159-1160).  In `_mass_params` (:1167-1170) the amplitude slot is
`sample['fpl']` — after Step 3 this already holds $r$ in the r-modes, so **no
change**; `_linearize_table` (:1184-1186) then differentiates the table w.r.t.
$r$ automatically and the scatter-free/Pallas kernels are untouched.
Check: `test_tabulated_path`/`_zdep` style FD gradient check on d/d`log_r`.

**Step 6 — record `fpl` deterministic.**  In both `pop_cosmo_model`s, after
`build_population_model`: in the r-modes record
`numpyro.deterministic('fpl', <f_eff(zref)>)` (for ref_z this is
`log_dN.log_dndm.fpl` post-`__post_init__`; for per_z compute $f_{\rm eff}$ at
`zref` via the same helper) so corner/diagnose scripts keep a comparable `fpl`.

**Step 7 — `run_inf.py` plumbing.**  Read
`tail_anchor = run.get("tail_anchor", fallback="simplex")` next to :91; pass
it at :194 (`mcmc.run`) and at the `recentering_baselines` call :176.
**Drive-by fix:** :176 currently forwards `use_low_bump` but not
`smooth_tail_edge` — add both flags there (harmless today only because both
defaults are True).  The `.nc` self-documentation is automatic
(:203-205 embeds every `[run]` key; :209-213 embeds prior/pop-config text).

**Step 8 — mock path.**  `scripts/reweight_inj.py:348-352` builds the slow
`LogDNDMDQDV(**pop_params)` filtered by `getfullargspec` against the pop
config, so adding the constructor field (Step 2) makes flavors selectable from
the pop config with no reweight-script change: set `tail_anchor = ref_z` (or
`per_z`) and `fpl = <r truth>` (the amplitude slot carries $r$; drop a comment
in the config).  Verify the config parser leaves the string un-coerced.

**Step 9 — CLAUDE.md.**  Fix the stale test path
(`scripts/test_fast_equivalence.py` → `scripts/testing_scripts/test_fast_equivalence.py`)
and add one line: `tail_anchor = simplex|ref_z|per_z`; in the r-modes the
prior samples `log_r` and `fpl` becomes a derived parameter.

## 4. Test plan (ordered)

1. **Default regression:** `cd scripts/testing_scripts && uv run python
   test_fast_equivalence.py` — all existing checks green, expectation
   *bit-identical* (default `"simplex"` touches no numeric path).
2. **Extended suite** (add to `test_fast_equivalence.py`, 812 lines, harness
   patterns at `test_population` :184 and `test_tabulated_path_zdep` :345):
   slow≡fast sweeps for `ref_z` and `per_z` at `mpisndot = 0` and `≠ 0`;
   FD gradient of the fast potential w.r.t. `log_r` through the 2-D tabulated
   path; and the **noevo identity**: at `mpisndot = 0`,
   $f_{\rm eff}(z) = f_{\rm eff}(z_{\rm ref})$ exactly, so for identical
   $(r, \ldots)$ the `ref_z` and `per_z` potentials must agree to float
   tolerance (not merely up to a constant) across a random parameter sweep.
3. **Smoke runs:** clone an existing noevo ini (e.g.
   `scripts/run_configs/mock_O5_ne.ini`) three ways
   (`tail_anchor = simplex|ref_z|per_z`, short chains); simplex must reproduce
   the reference posterior; ref_z vs per_z posteriors statistically identical.

## 5. Science A/B runs

Following existing naming (`scripts/pop_configs/mock_O5_*.txt`,
`scripts/run_configs/mock_O5_*.ini`, `scripts/priors/O5_*.prior`):

- Pop configs: `mock_O5_evo_refz.txt`, `mock_O5_evo_perz.txt` — evolving
  truth (`mpisndot ≠ 0`, e.g. reuse the evo7 scale), same $r$ truth in both.
- Prior file `O5_evo_tail_r.prior`: replace the `log_fpl` line with
  `log_r = Uniform(np.log(1e-2), 0.0)` — the user chose $r_{\max} = 1$ and
  log-uniform in $r$ (2026-08-29), matching the template
  `scripts/priors/real_dat_noevo_fullsel_r.prior`.
- Run configs, the 2×2 + regression:
  `mock_O5_evo_refz_fitrefz.ini`, `mock_O5_evo_refz_fitperz.ini`,
  `mock_O5_evo_perz_fitrefz.ini`, `mock_O5_evo_perz_fitperz.ini`,
  plus one noevo rerun as the free cross-flavor consistency check.
- Decision outputs: recovery of the truth in the matched fits (both flavors
  self-consistent?); size/direction of the bias in the crossed fits
  (`mpisn_ref`, `mbhmax`, `sigma`, `mpisndot`, and the $z$-dependence of the
  spectrum's high-mass edge) — this quantifies how much the anchoring
  convention matters at GWTC-scale evolution, which is the number the user
  needs to pick a flavor for the cosmology runs.

## 6. Choices (resolved by the user, 2026-08-29)

1. $r_{\max} = 1$ ("tail may just reach the continuum"), enforced via the
   prior bounds, not hard-coded.
2. Prior shape: log-uniform in $r$, `log_r = Uniform(np.log(1e-2), 0.0)`.
3. Turn-on width $w$: deferred (hard-coded 0.05 unchanged this pass).
4. Scope: switch + tests only.  The §5 science A/B configs and the
   cosmology-prior conversion are a follow-up pass after code review.

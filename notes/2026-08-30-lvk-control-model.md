# LVK "PowerLaw + 2 Peaks" control model (2026-08-30)

Infrastructure for control runs with an LVK-like mass model -- the GWTC-5
Default "PowerLaw + 2 Peaks" (arXiv 2605.27226) ignoring spins -- swapped in
for the PISN model, with everything else (cosmology inference, Madau-Dickinson
redshift evolution, frame conversion, selection integral, n_eff guards, R
sampling, float32 recentering, output format) identical.

## Files

- `src/intensity_models_lvk.py` -- the model module.  Imports all shared
  machinery from `intensity_models_fast` unchanged; only the mass intensity is
  new.  Zero edits to `intensity_models_fast.py`.
- `scripts/run_inf_lvk.py` -- entry point (light copy of `run_inf.py`; the
  PISN-only ini keys `use_low_bump` / `smooth_tail_edge` / `tail_anchor` do
  not exist).  Stamps `posterior.attrs["mass_model"] = "lvk_pl2p"`.
- `scripts/priors/lvk_gwtc5_control.prior`, 
  `scripts/run_configs/realGWTC5_lvk_control.ini`,
  `scripts/call_inf_lvk_control.sh` -- real-data GWTC-5 control run (259
  events, cosmology fixed to Planck; same data symlinks as the
  `realGWTC5_noevo_259ev_*` runs).
- `scripts/plot_ppd.py` -- dispatches on the `mass_model` attr via the
  `FAMILIES` registry (absent attr = PISN, unchanged behavior).  Both slice
  and `--marginal --lvk` modes work; `plot_trace.py` works unmodified.
  `plot_corner.py` / `diagnose_run.py` are NOT yet LVK-aware.
- `scripts/testing_scripts/test_lvk_model.py` -- validation (float64
  reference for the mass pdf, Zq table accuracy, the selection-consistency
  identity, finite gradients incl. dead events, recentering exactness,
  optional `--mcmc N` smoke run).

## Model

p(m1): broken power law (two segments m^-alpha_1 / m^-alpha_2, continuous at
`mbreak`, unit area on [mmin, mmax]) + two Gaussians truncated to
[mmin, mmax], stick-breaking weights (1-f_peaks, f_peaks*f_p1,
f_peaks*(1-f_p1)) matching the popsummary release's (lam_0, lam_1); the whole
mixture is multiplied by the existing smoothing window
`mmin_log_smooth_turnon` (= the LVK S(m | mmin, delta_m)) and cut at mmax.

p(q|m1) = q^beta S(q m1) / Zq(m1) on [mmin/m1, 1].  The m1-dependent
normalization Zq is REQUIRED (it does not cancel into R) and has no closed
form with S included, so it is tabulated per likelihood call (`LogQNorm`:
256 log-m1 nodes x 256-point trapezoid, static axis [2, 500] Msun).  One
instance per call is shared by the event term, the selection term, and the
log_norm reference evaluation, satisfying
`notes/2026-08-08-tabulated-selection-consistency.md`.  This is the ONLY
model-specific numerical approximation; the mass function itself is
closed-form, so there is no mass tabulation (the model uses the direct
evaluation path).

Conventions kept identical to the PISN model:

- point normalization: m1 dN/dm1 dq dV dt == 1 at (mref=30, qref=1,
  zref=0.001), so R has the same meaning and the PPD slice deterministics
  (`mdNdmdVdt_fixed_qz` etc., same `coords` grids) are directly comparable
  across model families.
- p(m1) is NOT renormalized after smoothing: the deficit is a
  parameter-dependent overall constant, which cancels exactly in the
  R-marginalized likelihood prod_i mu_i / mu_sel^nobs (and matches the LVK
  convention where the mixture fractions are defined pre-smoothing).
- unlike `LogDNDMDQDV` there is no `+ log(m1)` Jacobian and no second
  mass-function factor: p(m1) q^beta already is a density in (m1, q).

Derived parameters (`intensity_models_lvk.get_deterministic_parameters`) are
tolerant: `kappa = lam + dkappa` and `Om = Omh2/h^2` fire only when the source
keys are present; no PISN rules.  Fixed bounds (mmin/mmax/delta_m) are prior
floats; freeing one later is a prior-line swap (the LogQNorm axis is static so
traced mmin/delta_m/beta all work).

## Later additions (same day)

- `pairing` option (static flag, ini key `pairing`): "lvk" (default, q^beta
  with the Zq normalization) or "mt" (the PISN model's total-mass pairing of
  the same LVK mass function, no Zq table; beta becomes the total-mass
  exponent).  Runs: `realGWTC5_lvk_mtpair` (mass_model attr `lvk_pl2p_mt`).
  Comparison triplet: PISN+mt (realGWTC5_noevo_259ev_*) vs LVK+mt (mtpair)
  isolates the mass model; LVK+mt vs LVK+q^beta (lvk_control) the pairing.
- `realGWTC5_lvk_mminfree`: mmin = Uniform(3, 10), delta_m = Uniform(0.01,
  10) sampled (LVK-faithful), plus the audit-corrected mass priors
  (mpp_1 U(5,20), mpp_2 U(25,60), beta U(-2,7), f_peaks = Beta(2,1) which
  reproduces LVK's Dirichlet(1,1,1) mixture prior exactly).  NOTE the
  release's lam_0 is the POWER-LAW fraction (= 1 - f_peaks), not the peak
  fraction.
- LogQNorm fix: the trapezoid row spacing is computed analytically
  ((1-qlo)/(n_q-1) in log space), not via log(jnp.diff(q)) -- with traced
  mmin, rows with qlo -> 1-1e-6 have spacing below the float32 ULP at 1.0,
  diff underflows to exact zeros and d/d(mmin) is nan at every draw.
  test_lvk_model.py section 8 guards this.

## Deliberate differences from the LVK release

- one shared (mmin, delta_m), fixed -- the release samples two
  (mlow, delta_m) pairs (their roles unverified; likely primary/secondary).
- redshift evolution stays Madau-Dickinson (lam, dkappa, zp) -- LVK used
  PowerLawRedshift to z ~ 1.9, so R(z) comparisons are approximate above zp.
- prior bounds for mpp_*/sigpp_*/mbreak were read off the release posterior
  ranges (mbreak rails at 50, sigpp_2 at 10); transcribe from the paper if
  evidence-sensitive comparisons are needed.

## PISN-side pairing switch (2026-08-31): the 2x2 is complete

The ORIGINAL PISN model now has the same static `pairing` switch, so the
missing cell (PISN mass + q^beta pairing) exists:

- `intensity_models_fast.LogDNDMDQDV` / `intensity_models.LogDNDMDQDV`
  (slow twin), `build_population_model` and `pop_cosmo_model` take
  `pairing = "mt"` (default, the original total-mass pairing, bit-identical
  to before) or `"lvk"`:
  log_dN = log_dndm(m1,z) + beta log q + log S(q m1 | mbh_min, delta_m)
  - log Zq(m1) + log_dndv(z) - log_norm -- no second mass-function factor,
  no + log(m1) Jacobian; beta becomes the q exponent.  `LogQNorm` moved from
  `intensity_models_lvk` into `intensity_models_fast` (the lvk module now
  imports it) so there is ONE Zq implementation; one instance per model
  object serves the event term, the selection term, `_normalize` and the PPD
  slices alike.  In the tabulated path the "lvk" branch of the `_log_weights`
  closure does ONE mass-table lookup (m1 only) plus the exact q terms; the
  closure serves events and selection, so the tabulated-selection-consistency
  invariant holds by construction (verified as an identity).  The Zq lookup
  stays on the plain replica-trick backward (not `_linearize_table`).
- `run_inf.py`: ini key `pairing` (default mt), forwarded to
  `recentering_baselines` and `mcmc.run`; stamps
  `posterior.attrs["mass_model"] = "pisn"` (mt) or `"pisn_lvkpair"` (lvk).
  Absent attr still means pisn downstream, so old .nc's are unaffected.
- `plot_ppd.py`: FAMILIES entry `pisn_lvkpair` (fast module,
  `{"pairing": "lvk"}`).
- `scripts/testing_scripts/test_pisn_qpair.py` -- validation (exact algebraic
  recomposition, bit-identity of the mt default, slow-vs-fast for lvk,
  tabulated-vs-direct + the selection identity in both modes, finite
  potential/grads under all three tail_anchor modes).

Run definitions: `scripts/priors/real_dat_noevo_qpair_r.prior` (copy of
`real_dat_noevo_fullsel_r.prior` with ONLY `beta = Uniform(-4, 7)`, the
lvk_control q-exponent prior), `scripts/run_configs/
realGWTC5_noevo_259ev_qpair.ini` (clone of the refz ini + `pairing = lvk`),
`runs/realGWTC5_noevo_259ev_qpair/` (same data symlinks),
`scripts/call_inf_qpair.sh` + `scripts/call_plots_qpair.sh`.

The completed 2x2 comparison design:

| | mt pairing | q^beta pairing |
|:-|:-|:-|
| PISN mass | realGWTC5_noevo_259ev_{simplex,refz,perz} | realGWTC5_noevo_259ev_qpair |
| LVK mass | realGWTC5_lvk_mtpair | realGWTC5_lvk_control |

Rows isolate the pairing function at fixed mass model; columns isolate the
mass model at fixed pairing (the q^beta column shares the identical
beta prior).

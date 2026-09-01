# Recentering reference point: failure diagnosis and fixes (2026-08-31)

## What went wrong in `realGWTC5_noevo_259ev_qpair_h`

Real-data runs have no truth config, so `truth_params = {}` in
`scripts/run_inf.py` and `recentering_baselines()`
(`src/intensity_models_fast.py`) fills **every** reference parameter from a
prior draw at fixed seed 0. Freeing `h` in the prior shifted NumPyro's
per-site RNG stream, and the new seed-0 draw landed on `c = 0.321`.

In `tail_anchor = "ref_z"` mode, `log_f_eff` contains `- log(c - 1)`, which
is NaN for `c <= 1` (the power-law tail is non-normalizable there — its area
diverges). The derived `fpl` became NaN, all 259 event reference
log-likelihoods hit the `_LOG_ZERO_FLOOR = -1e6` floor, and
`log_pdraw_sel_scale` came out at −1000021. The float32 recentered potential
then carried a constant ~1e6-magnitude offset per event — roughly 31 lost
bits of log-likelihood resolution — and NUTS froze (step size ~1e-4, every
tree at max depth). The log **did** warn: "recentering_baselines: 259
event(s) have (near-)zero likelihood at the reference point", but a warning
was too easy to miss and the run burned a full allocation.

Chain: RNG-stream shift → seed-0 reference draw with `c < 1` → NaN `fpl` →
per-event −1e6 floors → float32 resolution loss → NUTS stalled.

## Fix 1 — explicit `[ref_params]` ini section

`scripts/run_inf.py` (and `scripts/run_inf_lvk.py`, same convention) now
accept an optional `[ref_params]` section in the run-config ini:

```ini
[ref_params]
h = 0.674
c = 4.844
...
```

Simple `name = float` lines. The values are merged **on top of** the
truth/prior-draw mechanism: named sites are pinned via `numpyro.substitute`
inside `recentering_baselines`; any sampled site not listed still comes from
the seed-0 prior draw, exactly as before. With no section, behavior is
unchanged.

Rules:

- Only **sampled** sites of the prior may appear (validated; a fixed-float
  or unknown name raises `ValueError`). Fixed floats are `deterministic`
  sites, which `numpyro.substitute` would silently override, and derived
  deterministics (`fpl`, `mpisn`, ...) would be inconsistent with their
  parents.
- Parameter-name case is preserved (a dedicated case-sensitive
  `ConfigParser` pass), so e.g. `Omh2` works.
- Good values: posterior medians of a healthy run of the same family.

`run_configs/realGWTC5_noevo_259ev_qpair_h.ini` and
`run_configs/realGWTC5_noevo_259ev_qpair_mpisndot.ini` now carry
`[ref_params]` with the posterior medians of the healthy fixed-h baseline
`runs/realGWTC5_noevo_259ev_qpair` (plus `h = 0.674` / `mpisndot = 0.0`
respectively).

## Fix 2 — the `c <= 1` region no longer produces NaN

`log(c - 1)` appears in `log_f_eff` (ref_z/per_z tail anchoring) and in
`log_normalized_power_law_tail(_from_log)` (the tail density itself, all
anchor modes). Both are now guarded with the double-`jnp.where` pattern
(`_guarded_log_cm1` in `src/intensity_models_fast.py`, mirrored in the slow
twin `src/intensity_models.py`): for `c <= 1` the result is floored at
`_LOG_ZERO_FLOOR` (so `fpl -> 0`, continuum-only model, finite likelihoods
and zero — not NaN — gradients); for `c > 1` the result is **bit-identical**
to the unguarded expression (`jnp.where` selects the exact same computed
value), so the fast/slow and mt-default equivalence tests still pass.

The `c` prior bounds were deliberately **not** changed (finished runs used
them and the posterior sits near `c ≈ 4`); the tail is simply
non-normalizable for `c <= 1` and the floor keeps prior-draw reference
points finite.

With this guard alone, the original failing seed-0 draw (`c = 0.321`) now
yields 0 dead events and a finite `log_mu_sel = 3.26` — the freeze mechanism
is closed even without `[ref_params]`.

## Fix 3 — dead reference events are a hard error

`recentering_baselines` (both `intensity_models_fast` and
`intensity_models_lvk`) now raises `RuntimeError` instead of warning when
more than `max_dead_events` (default 0) events sit at/below the −1e6 floor
(NaN counts as dead), or when `log_mu_sel` at the reference is non-finite or
floored. The message reports the dead count, echoes the reference draw's
parameter values, and points to `[ref_params]`. Healthy runs report zero
dead events (checked against the `realGWTC5_259ev_qpair*` logs), so a single
dead event is fatal by default; pass `max_dead_events=N` to loosen.

## Verification (2026-08-31, CPU)

- Fixed h ini with `[ref_params]`: 0/259 dead events, loglike_ref in
  [−17.1, −0.33], `log_pdraw_sel_scale = 2.864` (healthy family runs:
  3.82–3.88).
- Guard: `c > 1` bit-identical to the old expression (checked at
  c = 1.0001, 1.5, 4, 4.844, 8); `c <= 1` finite with finite gradients.
- `testing_scripts/test_pisn_qpair.py` and
  `testing_scripts/test_fast_equivalence.py` pass (incl. the mt-default
  bit-identity check).
- Fix 3 demonstrated: a synthetic sub-`mbh_min` event raises the
  `RuntimeError` with the reference point echoed.

See also: `notes/2026-09-01-h-divergences-float32.md` — even with a healthy
reference point, the float32 recentered potential retains enough roughness
to cause the ~2.9% divergence rate of this run's h-free sampling (float64
twin: 0 divergences).

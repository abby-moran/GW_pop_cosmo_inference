# Fullcosmo evo4–evo7: what the production runs taught us

*2026-08-10.  Companion to `2026-08-09-scatter-free-vjp.md`,
`2026-08-09-pivot-reparam.md`,
`2026-08-09-log-fpeak-parametrization.md`,
`2026-08-09-low-mass-bump-width-identifiability.md`, and
`mass-model-audit.md`.  All diagnoses via `scripts/diagnose_run.py`;
corners via `scripts/plot_corner.py`.*

## Setup lineage

Shared sampler settings unless noted: 2 chains × 1800 draws, `n_pe=4000`,
`max_tree_depth=10`, `dense_mass=True`, target accept 0.8, 2×H100.
Scatter-free VJP on from evo4 onward.

| run | job | unique change | data / truth | prior | wall |
|---|---|---|---|---|---|
| **evo3** | 6786762 | baseline: dense mass + depth 10 | `endO5_val2`, `msigma_low=4`, 9000 evt | `gwtc5_fullcosmo_evo.prior` | 16.97 h |
| **evo4** | 6790523 | scatter-free VJP only | same as evo3 | same | 15.72 h |
| **evo5** | 6792525 | pivot (`mpisn_ref` @ `zpivot=0.75`) | same as evo3 | `gwtc5_fullcosmo_evo_pivot.prior` | ~15.7 h |
| **evo6** | 6792535 | pivot + sample `log_fpeak` (not `log_flow`) | same as evo3 | `gwtc5_fullcosmo_evo_pivot_fpeak.prior` | ~15.7 h |
| **evo7** | 6793061 | pivot + `log_fpeak` + **narrow bump** | `endO5_w15pool`, `msigma_low=1.5`, 8000 evt | `gwtc5_fullcosmo_evo_pivot_fpeak_w15.prior` (`msigma_low` ∈ [0.5, 2.5]) | 12.3 h |

Outputs live under `runs/endO5_fullcosmo_evo{N}/`.

## Diagnostics at a glance

| | evo3 | evo4 | evo5 | evo6 | evo7 |
|---|---|---|---|---|---|
| OVERALL | FAIL | FAIL | FAIL | FAIL | FAIL\* |
| min bulk ESS | ~15 | 5 | 5 | 6 | **334** |
| max r-hat | ~1.12 | 1.35 (`a`) | 1.37 (`msigma_low`) | 1.27 (`c`) | **1.01** |
| depth-10 saturation | 100% | 100% | 100% | 100% | **40%** |
| divergences | 0 | 0 | 0 | 0 | **45 (1.25%)** |
| step size (per chain) | — | ~8e-4 / 1e-3 | 1.5e-3 / 3e-4 | 2e-3 / 3e-4 | **5e-3 / 7e-3** |
| bump / `a` verdict | FAIL (broad) | FAIL | FAIL | FAIL | **OK** (width ~1.6) |
| post. median `msigma_low` | — | 3.25 | 3.36 | 3.29 | **1.59** |

\*evo7's sampler sections are healthy; OVERALL FAIL is from divergences and
(after the diagnose_run update) the narrow-feature selection-tilt check.
ESS / r-hat / geometry / bump sections all pass or are healthy NOTEs.

Corners:
`runs/endO5_fullcosmo_evo{4,5,6,7}/O5_fullcosmo_evo{N}_corner.png`.

## Finding 1 — Scatter-free VJP is a wall-time win, not a geometry fix

evo3 vs evo4 were identical science setups.  Wall time dropped ~8%
(16.97 → 15.72 h; ~15.6–15.7 s/it vs ~16.5–17.0).  That is smaller than the
A6000 grad-only bench (1.24×) because leapfrog time is not all gradient
kernels.  Mixing stayed broken (100% depth saturation, unusable ESS).
Expected: potential bit-identical, grads agree to ~2×10⁻⁵
(`2026-08-09-scatter-free-vjp.md`).

## Finding 2 — On the width-4 mock, pivot and `log_fpeak` do not unlock NUTS

evo5 and evo6 were meant as global sampling-efficiency tests of the
reparameterizations that looked helpful in short benches
(`2026-08-09-pivot-reparam.md`: ~2.3× better worst grads/ESS on 2000 events).
At full scale on the **same** `endO5_val2` / `msigma_low=4` truth:

* Both still saturate depth 10 in **100%** of iterations with min bulk ESS
  **5–6**.
* The dominant correlation direction remains the
  `a`–amplitude–`msigma_low`–`mp_low` block (evo5: |r|(`a`,`log_flow`)=0.88;
  evo6 softens pairwise |r|(`a`,`log_fpeak`) to ~0.77 but ESS does not move).
* Diagnose still flags the bump as too broad (Fisher σ(`a`) ≳ 1; posterior
  median width ~3.3).
* Zero divergences; `sigma` piles on the prior floor (22% / 33% of draws
  within 1% of 0.05).

**Conclusion:** with a broad true bump, reparameterizing the cosmology /
PISN-evolution / bump-amplitude coordinates is not enough.  The
conditioning problem named in evo3/evo4 diagnoses is the bump continuum
eating `a`, as audited in `mass-model-audit.md` and
`2026-08-09-low-mass-bump-width-identifiability.md`.  Short-bench
pivot gains on `mpisndot` are real but swamped at production scale by this
ridge.

## Finding 3 — Narrowing the bump (evo7) fixes mixing

evo7 keeps pivot + `log_fpeak` and the same sampler knobs, but changes the
**mock**: truth `msigma_low=1.5`, prior capped at 2.5, fresh rejection-
sampled pool (`endO5_w15pool`, ~371M rows → ~125k sel), `evt_end=8000`.

Results:

* min bulk ESS **334**, every free parameter r-hat ≤ **1.01**.
* Depth cap hit in only **40%** of iterations; adapted step size ~6×
  larger than evo4.  Diagnose says depth 10 / `nmcmc` could be lowered.
* Bump section **OK**: median width 1.59, Fisher σ(`a`) ~0.22; `a` retains
  independent leverage (truth quantile ~0.36).
* Wall ~12.3 h at ~11.6 s/it (helped by fewer events and shorter trees).

So the production-scale efficiency story is: **fix the bump width first**;
pivot/`log_fpeak` are then free to help secondary directions.  evo5/6 show
that applying those reparams *without* fixing the bump leaves ESS at the
evo4 floor.

## Finding 4 — evo7's remaining issues are selection noise and mild divergences

**Divergences (45 / 3600 = 1.25%).**  Diffuse across the posterior (not
concentrated at the `sigma` floor).  Consistent with the highest
`mc_var_loglike` of the series (~3.85–4.16 of budget 5): roughness of a
noisy surface, not a geometry cliff.

**`sigma` truth miss.**  Truth 0.0539 vs posterior median 0.130 (quantile
0.001), with ESS 334 so the miss is meaningful.  Root cause is Monte Carlo
noise in the width15 selection set, not a code bug or sampler failure:

* Conditional likelihood scans (all other params at truth) peak at
  `sigma≈0.09` on evo7 inputs and at truth on evo6 inputs; tabulation off
  reproduces this to 0.01 nats.
* Swapping selection sets moves the bias with the selection file
  (width15 PE + val2 sel recovers truth; val2 PE + width15 sel biases).
* Bootstrap `nobs·sd(Δ log_μ_sel)` over the posterior 16–84% of `sigma` is
  ~3 nats — the same order as the entire likelihood signal separating
  truth from 0.09.  The Farr `neff_sel` hinge (2.0× OK here) only bounds
  global normalization, not a sharp-feature tilt.
* With the likelihood flattened, the prior
  `TruncatedNormal(0.1, 0.1, low=0.05)` (median ~0.14) finishes the job.

`diagnose_run.py` now has a narrow-feature selection-tilt check for this
failure mode (`2026-08-09-run-diagnostics.md`).

**`dmbhmax` prior-dominated** (sd ratio 0.92) — do not quote as measured.

## Practical takeaways

1. **Do not use the width-4 (`msigma_low=4`) mock for fullcosmo sampler /
   reparam efficiency tests.**  It guarantees an unidentifiable `a`–bump
   ridge that masks everything else.  Cap `msigma_low` ≤ 2.5 in priors and
   mock truths intended for geometry work
   (`utils.BUMP_MSIGMA_LOW_MAX`).
2. **evo7 is the informative geometry run** in this series.  evo5/6 are
   negative controls: reparam alone ≠ mixing on a bad mass model.  Do not
   quote `sigma` (or other tilt-flagged narrow features) from evo7 until
   the selection set is enlarged or drawn from the proposal pool without
   truth-rejection.
3. **Scatter-free VJP stays on** — modest, validated wall saving; keep
   documenting speed separately from ESS.
4. **Next levers on the evo7-like setup:** grow / redesign the selection
   set for sharp features; optionally raise `n_pe` if divergences persist;
   reconsider the informative `sigma` prior shape (keep the 0.05 floor);
   optionally drop `max_tree_depth` / `nmcmc` now that saturation is only
   40%.
5. **Pivot + `log_fpeak` remain justified** by short benches and by evo7's
   healthy cosmology/evolution ESS, but their production-scale value should
   be measured on narrow-bump (or capped-bump) mocks, not on evo4's truth.

## Pointers

* Diagnose / corner artifacts: `runs/endO5_fullcosmo_evo{4,5,6,7}/`
* Configs: `scripts/run_configs/mock_O5_fullcosmo_evo{4,5,6,7}.ini`
* Pop configs: `mock_O5_noevo.txt` (width 4) vs `mock_O5_width15.txt`
  (width 1.5)
* Related notes listed in the header.

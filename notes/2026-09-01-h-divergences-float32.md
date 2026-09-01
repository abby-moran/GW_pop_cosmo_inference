# h-free divergences are float32 potential roughness (2026-09-01)

*Experiments on the real-data h-free run `realGWTC5_noevo_259ev_qpair_h`
(259 events, PISN + q^beta pairing, `tail_anchor = ref_z`).  Verdict: the
~2.9% divergence rate is caused by float32 roughness of the recentered
potential, not by posterior curvature.  A float64 twin at identical settings
has 0/1200 divergences with the same step size.  Accept the rate at current
settings; exact h-pivot reparams double ESS(h) but do not reduce divergences.
Experiment harness lived in the session scratchpad (ephemeral); this note is
self-contained.*

## The problem

Freeing `h` (`TruncatedNormal(0.7, 0.2, low=0.4, high=1.2)`) raises the
divergence rate from 0.1–0.2% (fixed-h twins) to ~2.4–2.9%, with divergences
clustered at high h (top h-quintile: 4.7–8.2% vs ~1–3% elsewhere).
Convergence is otherwise healthy (R-hat <= 1.008, ESS > 800) and posteriors
are not materially biased.

## Investigation chain

1. **Ridge forensics** (earlier sessions): a nearly-linear h–mass-scale
   ridge — `log mp_low ~ -0.14 log h` (R² = 0.63), `log mpisn_ref ~ -0.18
   log h`, `log mbhmax ~ -0.19 log h` — the sampled image of the physical
   map `m_det = m (1+z(dL_eff; h))` with dL_eff ≈ 0.9 Gpc (mp_low) / 1.3 Gpc
   (mpisn_ref, mbhmax).  Not a funnel, not a prior-boundary trap, and *not*
   a sigma-floor interaction (divergent draws are not preferentially near
   sigma = 0.05: 21% below 0.07 vs 24% overall).
2. **dense_mass = true** (earlier sessions): whitened the ridge, divergences
   persisted (and 3/12 chains suffered step-size collapse).
3. **This session's suite**: real data, production potential + `[ref_params]`
   recentering, 600 warmup + 600 draws x 2 chains, fixed seed.  Candidates:
   `target_accept` 0.9/0.95, exact h-pivot reparams (below), the exact
   `1+z_eff(h)` map, combinations, and a float64 control.

## Results (full production data, 1200 draws per run)

| config | divergences | div. h-rank | leapfrogs/draw | ESS(h) |
|---|---|---|---|---|
| control f32, 2 seeds | 31 + 38 = 2.9% | 0.75 / 0.64 | 39.9 | 565 |
| control f32, `scatter_free_tables=False` | 34 (2.8%) | 0.69 | 33.7 | — |
| h-pivot (mp_low + mpisn_ref), 2 seeds | 43 + 52 = 4.0% | 0.58 / 0.46 | 37.5 | 827 |
| exact `1+z_eff(h)` map | 37 (3.1%) | 0.63 | 33.8 | 1201 |
| **control float64** | **0 (0.0%)** | — | 37.0 | 715 |

On softened data (2000/7000 PE samples, 350k/1.43M selection rows) the
pivot looked like a 2.4x win (7 vs 17); it did not transfer to the full
data — chain-to-chain divergence scatter is super-Poisson, so single-seed
comparisons mislead.

## Settled mechanism

**Float32 roughness of the potential/gradients, not true curvature.**

- float64 at otherwise identical settings: 0/1200 divergences where f32
  gives 31–38 (≈35 expected under the f32 null), with the *same* adapted
  step size (0.11) and trajectory length — the true posterior supports that
  step cleanly.
- The f32 rate is invariant to the gradient path (Pallas scatter-free vs
  plain lookups: 34 vs 31/38) — it is the arithmetic width, not the kernel.
- Exact reparams that fully decorrelate h from the mass scales
  (corr(log h, log u) ≈ 0, ESS(h) doubled) **redistribute** divergences away
  from high h (median h-rank 0.75 → ~0.5) but do not reduce the count.  The
  ridge sets *where* the f32 noise trips the energy-error threshold, not
  how often — same reason dense mass whitened the ridge yet failed.
- Raising `target_accept` never helped: 0.95 was *worse* (26 vs 17 on the
  softened data) at 2.1x leapfrog cost, and 0.9 reproduced the
  dense-mass-style step collapse on one chain.

## Unbiasedness (triple evidence)

1. Dense-mass twins agree with diagonal runs to < 0.05σ (earlier forensics,
   production length).
2. The exact-pivot twin — provably the identical posterior (log-density
   identity verified to 2e-5; construction below) — agrees to <= 0.11σ on
   all key parameters *with the divergences de-clustered from h*, directly
   ruling out h-tail undersampling.
3. The divergence-free float64 run agrees to <= 0.14σ (within its
   short-chain MC error; the least precise of the three checks).

## Practical guidance

- **Default: accept ~2.9%** at current settings (diagonal mass,
  `target_accept = 0.8`, `max_tree_depth = 7`, float32).  Read the
  divergence count of h-free runs as an f32-arithmetic artifact, not a
  sampling pathology, provided R-hat/ESS are healthy.
- **Divergence-free record when needed** (e.g. referee-proofing):
  `jax.config.update('jax_enable_x64', True)` before any array creation,
  plus `scatter_free_tables=False` (the Pallas kernels are f32-only; the
  plain path was not slower on an A6000).  Measured cost: 2.6x wall on an
  A6000 (17-min H100 production run → expected ~30–35 min; see next
  section).  The recentering machinery is unnecessary but harmless in x64.
  run_inf.py exposes this as `x64 = true` in the `[run]` section of the ini
  (enables `jax_enable_x64` and forces `scatter_free_tables=False`).
- **Do not** use `target_accept >= 0.9` or `dense_mass = true` on this
  potential (see warnings below).

## Expected f64 cost on H100

- The 2.6x A6000 slowdown says the workload is **memory-bandwidth-bound,
  not FLOP-bound**: the A6000's FP64:FP32 throughput is 1:64, so a
  compute-bound potential would have slowed ~30–60x.  The dominant cost is
  elementwise passes over the 1.43M-row selection arrays (and 259x7000 PE
  arrays), where f64 just doubles the bytes moved.
- On H100 PCIe (FP64:FP32 = 1:2, ~26 vs ~51 TFLOPS) the slowdown should
  therefore sit near the bandwidth floor of ~2x: the 17-min production
  h-free run becomes roughly **30–35 min**.
- Upshot: at these run lengths, always-f64 is an affordable policy call —
  and it retires the whole float32 guard ecosystem as a live concern
  (recentering baselines become belt-and-braces; failure modes like the
  frozen-run recentering incident become impossible rather than guarded).
  Counterweights: x64 bypasses the f32-only Pallas scatter-free kernels
  (the plain path was *faster* on the A6000 but is unbenchmarked in x64 on
  the H100 — the machine those kernels were written for, because scatter
  hurt there), and the 2x compounds for longer future runs (fullcosmo,
  larger catalogs).  Measure the true H100 cost with a benchmark twin
  (`x64 = true`) before adopting f64 as the default.

## The two-scale h-pivot (verified; ESS lever only)

Doubles ESS(h) per leapfrog at unchanged cost; adopt only if h-mixing ever
becomes the bottleneck.  Exact spec, preserving the posterior *exactly*:

- Sample `u_mp_low ~ Uniform(5.1293, 13.1941)` and
  `u_mpisn_ref ~ Uniform(20.6672, 41.6344)` instead of `mp_low`,
  `mpisn_ref`.  (Bounds = original truncated supports shrunk so the derived
  value stays in-support for every h in [0.4, 1.2].)
- Derive `mp_low = u_mp_low * h**-0.14`, `mpisn_ref = u_mpisn_ref *
  h**-0.18` (deterministics).
- Add for each scale a `numpyro.factor` carrying the **original** prior
  density at the derived value plus the Jacobian:
  `TruncatedNormal(9, 2, low=5, high=15).log_prob(mp_low) - 0.14*log(h)`
  and `TruncatedNormal(35, 5, low=20, high=49.1).log_prob(mpisn_ref)
  - 0.18*log(h)`.  (The flat u density is a constant; the factor makes the
  pushforward exactly the old prior.)
- `[ref_params]` remap for the current ini reference point:
  `u_mp_low = 9.3633`, `u_mpisn_ref = 30.7751` (= 9.895·0.674^0.14,
  33.04·0.674^0.18), replacing the `mp_low`/`mpisn_ref` lines.
- Verified: pivot log-density − original log-density + Jacobian is constant
  (2e-5 spread, float32); sampled posteriors agree to <= 0.11σ.  The exact
  `1+z_eff(h)` map (quadratic in log h fit to `dL(z; h, Om=0.315, w=-1) =
  dL_eff`) behaves the same — the extra curvature exactness buys nothing.

**Now productionized** (2026-09-01): the pivot is available as `h_pivot =
true` in the `[run]` section of the run-config ini (default false; optional
exponent overrides `h_pivot_gamma_mp_low = 0.14`, `h_pivot_gamma_mpisn_ref =
0.18`), implemented in `utils.sample_parameters_from_dict` and threaded
through both `pop_cosmo_model` twins by `run_inf.py` — no prior-file edits
needed.  The u-bounds are computed **at runtime** from the actual prior
objects (`utils.h_pivot_u_bounds`; the numbers above assume the qpair_h
prior's supports — e.g. the mpisndot-run prior's tighter `mpisn_ref` support
(23.5, 49.1) gives tighter bounds automatically), a fixed-h prior is a hard
error, and a fixed-float `mp_low`/`mpisn_ref` is skipped silently.
`mp_low`/`mpisn_ref` are recorded as deterministics under their original
names (posterior files unchanged apart from the new `u_*` sample sites), and
`run_inf.py` auto-remaps physical `mp_low`/`mpisn_ref` lines in `[ref_params]`
(and truth/init points) to the u-sites via `u = m * h**gamma`, so existing ini
`[ref_params]` blocks work unedited.  Regression coverage:
`testing_scripts/test_h_pivot.py` (density identity re-verified at 2e-4
spread across h in [0.45, 1.15] on synthetic data, both twins; runtime
u-bounds; error paths; pivot-off bit-identity).

## Incidental warnings

- `target_accept = 0.9` triggered warmup step-size collapse (step ~5e-4,
  R-hat > 2 on the frozen chain) twice on this potential — once here, once
  in the dense-mass runs.  Treat 0.8 as the safe operating point.
- `b` is weakly bimodal (~3–4% posterior mass at `b < -1`) and slow-mixing.
  Unrelated to the divergences; not chased here.

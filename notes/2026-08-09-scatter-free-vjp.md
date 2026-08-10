# Scatter-free table-lookup gradients (Pallas custom VJP)

*2026-08-09.  Branch `local/opt-pass-3`.  Removes the backward-pass scatter
kernels from the tabulated path of `src/intensity_models_fast.py` by routing
the table-value gradient through per-parameter tangent tables contracted in
custom Pallas kernels.  Companion notes: `2026-08-07-optimization-changelog.md`
(the whole series), `2026-08-07-mass-table-2d.md` (the 2-D mass table this
builds on).*

## The problem

Reverse-mode AD of "gather from a parameter-dependent table" is a scatter-add
of one value per data point back into the table.  After the 2-D mass table
landed, a production gradient step (9000 x 4000 PE + 1.7M selection samples,
h/Omh2/mpisndot free) spent **20.4 of 36.4 ms of device time in scatter
kernels** on an A6000 -- twelve scatters per step: 4 corners x {m1, m2} into
the (30 x 8192) mass table and 2 corners x {log1p z, J} into the fused dL
table, each for both the PE and selection sets.  The replica mitigation
(`SCATTER_REPLICAS = 32`) was already saturated: raising it to 64 made the
gradient *slower* (42.4 ms) and 128 much slower (66.3 ms) -- the extra copies
just evict everything else from L2.

## The idea

`d(potential)/d(table)` never needs to be materialized.  Every hot table is
built from a handful of scalar parameters theta:

| table | shape | traced parameters (all-free run) |
|---|---|---|
| mass function `f_tab` | (30, 8192) | a, b, c, mpisn, mpisndot, mbhmax, sigma, fpl, mp_low, msigma_low, flow (k=11) |
| fused dL `[log1p z, J+log dN/dV]` | (2048, 2) | Om, lam, kappa, zp (k=4; h enters only via the index and the scalar +2 log dH added after lookup) |

With tangent tables `U_k = dT/dtheta_k` -- computed once per likelihood call
by `jax.linearize` of the table build, i.e. k extra ~250k-entry table builds
amortized over ~38M point lookups -- the chain rule through the table values
collapses to

    theta_bar_k = sum_points  g * lerp(U_k, x)

a gather plus a reduction, with no atomics.  The lookup *position's* cotangent
(the dense chain through log1p(z), the mass axis, ...) is returned as usual by
the same `custom_vjp`, so the gradient is mathematically identical to ordinary
reverse mode up to float32 summation order.  The forward value is bit-identical
(same arithmetic).

## Why a Pallas kernel

Pure-XLA formulations of the contraction all lose:

* **einsum over materialized per-point tangent rows**: (npoints, k) floats of
  peak memory per lookup; measured +12 GiB and grad 37.5 -> 48.3 ms.
* **chunked `lax.scan`**: fixes memory (8.6 GiB) but serializes ~18 small
  kernels per lookup plus pad copies; grad 59.6 ms.
* **one reduction per parameter**: re-streams the 36M-point index/cotangent
  arrays k times.

The right shape for the computation is: stream each point once, keep the k
partial sums in registers, gather from tangent tables that stay hot in L2
(~1 MB per channel).  That is a ten-line Pallas kernel
(`_sf_theta_kernel_2d` / `_sf_theta_kernel_1d`), launched over blocks of
`_SF_BLOCK = 1024` points, followed by a (nprog, k) `jnp.sum`.  On
production-like clustered indices the kernel does one 36M-point lookup's
contraction in 4.0 ms where the equivalent replicated scatter takes 21 ms.
Notes from tuning:

* Block size 1024 beat 2048/4096/8192 (8.45 / 8.91 / 10.57 / 95.9 ms for the
  two mass lookups).
* A fused kernel doing the m1 + m2 lookups in one pass -- they share the z
  cell -- *loses* at every block size (register pressure kills occupancy):
  12.1 ms vs 8.45 ms.  The lookups stay separate calls.
* Tangent tables are kept channel-first, (k, nz, nm), so each channel is a
  contiguous table and the 2-D kernel needs no transpose.
* Uniform-random indices are a factor ~4 slower than realistic per-event
  clustered ones; benchmark index distributions matter.

CPU (tests, small data) falls back to a chunked-`lax.scan` XLA path; there is
no scatter contention to fix there.

Requires `absl-py` (imported by `jax.experimental.pallas`); added to
`pyproject.toml`.

## Measured (A6000, production scale, synthetic bench)

| configuration | grad before | grad after | peak mem before | after |
|---|---|---|---|---|
| h, Omh2, **mpisndot** free (2-D table) | 37.5 ms | **30.3 ms** (1.24x) | 9.37 GiB | 8.89 GiB |
| h, Omh2 free, mpisndot = 0 (1-D table) | 20.1 ms | **12.8 ms** (1.57x) | 6.12 GiB | 4.29 GiB |

Forward pass: ~4.6 -> ~5.0 ms (the k extra table builds under
`jax.linearize`).  Device-time breakdown after: scatter kernels 20.4 -> 0.4 ms;
Pallas kernels 9.8 ms; the remainder is the dense logsumexp/n_eff forward and
backward passes (~11 ms), which are plain bandwidth-bound streams over the
(nobs, nsamp) array and the natural next target if more is ever needed
(e.g. saving the forward's exp(w - max) for the backward).

At depth-10 trees (the `fullcosmo_evo3` sampler settings) the 2-D-table saving
is ~2 hours per 1000 leapfrog-saturated samples on the A6000.

### Full-scale H100 validation (`endO5_fullcosmo_evo3` vs `evo4`)

Apples-to-apples NUTS runs on 2×H100 (same prior, `endO5_val2` data, 9000
events, `n_pe=4000`, `max_tree_depth=10`, `dense_mass=True`, 2×1800 draws):

| run | code path | Slurm wall | s/it (chain finishes) |
|---|---|---|---|
| `endO5_fullcosmo_evo3` (job 6786762) | replicated-scatter VJP | 16.97 h | ~16.5–17.0 |
| `endO5_fullcosmo_evo4` (job 6790523) | scatter-free VJP (`local/opt-pass-3`) | 15.72 h | ~15.6–15.7 |

Wall speedup ~**8%** (~1.25 h).  That is smaller than the A6000 grad-only
bench (1.24×) because leapfrog time also includes dense-mass ops, host
overhead, etc.; the launch script's ~13–14 h projection was optimistic.

Mixing / geometry unchanged: both runs still saturate depth 10 in 100% of
iterations with unusable min bulk ESS (evo3 ~15, evo4 ~5 — chain noise, not
a VJP regression).  Expected: the potential is bit-identical and gradients
agree to ~2×10⁻⁵, so the posterior surface is the same.  Scatter-free VJP
is a real wall-time cut with correct gradients; it does not fix conditioning.

## Correctness

* Potential: bit-identical with `scatter_free_tables` on/off (same forward
  arithmetic), both table layouts, GPU and CPU.
* Gradients: agree with the replicated-scatter backward to <= 2.2e-5 relative
  (float32 summation order), all 17-18 sampled parameters, truth and
  edge-heavy points, both layouts, GPU and CPU (CPU run with `_SF_CHUNK`
  forced small so the scan path is exercised).
* `scripts/test_fast_equivalence.py` gained test 9
  (`test_scatter_free_vjp`) asserting exactly this, and the full suite passes.

## Knobs

* `pop_cosmo_model(..., scatter_free_tables=None)`: default on (tabulated
  path only).  `False` restores the replicated-scatter backward.
* `scripts/bench_model.py --no_sfvjp` for A/B timing.

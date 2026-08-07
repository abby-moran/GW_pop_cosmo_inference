What I changed

No existing file was modified. Everything is new, side-by-side, so you can A/B before committing to anything:

┌──────────────────────────────────┬───────────────────────────────────────────────────────────────────────────┐
│               File               │                                  Purpose                                  │
├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ src/intensity_models_fast.py     │ Optimized drop-in for intensity_models — same public API                  │
├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ scripts/bench_model.py           │ Benchmarks the gradient (what NUTS pays) + --diagnose traces every site   │
├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ scripts/test_fast_equivalence.py │ Numerical equivalence vs. the original, scored against float64 references │
├──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ scripts/bench_sampling.py        │ End-to-end NUTS comparison — written but not yet run                      │
└──────────────────────────────────┴───────────────────────────────────────────────────────────────────────────┘

Measured, at 9000 × 4000 PE + 1.7M selection samples (A6000)

┌─────────────────────┬──────────┬──────────┬───────┐
│                     │ original │   fast   │       │
├─────────────────────┼──────────┼──────────┼───────┤
│ potential (forward) │ 123.4 ms │ 3.3 ms   │ 37×   │
├─────────────────────┼──────────┼──────────┼───────┤
│ grad (per leapfrog) │ 476.6 ms │ 35.3 ms  │ 13.5× │
├─────────────────────┼──────────┼──────────┼───────┤
│ peak GPU memory     │ 30.0 GiB │ 14.9 GiB │ 2.0×  │
└─────────────────────┴──────────┴──────────┴───────┘

Worst case at max_tree_depth=7: 61 h → 4.5 h. Equivalence: potential agrees to 1e-6 relative, all 17 gradient components to ~3e-4, and the fast version is more accurate than the original against float64 references (PISN grid 24×, cosmology Jacobian 24×).

Where the speed came from

1. Scatter contention in the backward pass (91 → 35 ms). Differentiating a gather from a small table makes the VJP a scatter-add of 36M values into ~514 slots. Measured in isolation: 0.5 ms forward, 13.7 ms backward, and worse for smaller tables. Fix: keep 32 copies of each table so adjacent points hit different copies; the VJP of the broadcast sums them for free.
2. No searchsorted in the hot path. Every table here is log- or log1p-uniform, so the grid index is closed-form. jnp.interp was running a 10-iteration binary search — ten gather passes over 36M elements — five times per evaluation.
3. One fused cosmology lookup instead of three. The whole Jacobian block is J(u) + 2·log(dH) with u = log(dL) − log(dH); with Om and w fixed the tables are compile-time constants.
4. Max-subtracted linear trapezoid for the PISN mco integral (one exp instead of logaddexp+logsumexp), with the mco axis moved last so the reduction is contiguous.
5. mpisndot == 0 detected statically → the PISN grid is built with 1 z-slice instead of 30 and the interpolation collapses to 1-D.
6. Single-pass logsumexp + n_eff; removed nested @jax.jit decorators; softplus instead of log(logistic(...)).

Bugs found

Two are serious and I demonstrated both directly:

1. @jax.jit on get_deterministic_parameters silently drops kappa, mbhmax, fpl, flow from your output. numpyro.deterministic records its site during the inner jit trace; on a cache hit the sites vanish. Confirmed on a minimal model — mcmc.get_samples() has no kappa at all — and on your real model: derived sites present = [], missing = ['flow','fpl','kappa','mbhmax'].
2. nan_to_num fixes values but not gradients. When an event's weights all underflow, neff = exp(2·(−inf) − (−inf)) is NaN; nan_to_num returns a finite value with a NaN gradient, which NUTS treats as a divergence. The original model cannot even initialize_model when one event is out of support; the fast version returns a finite potential and clean gradients. This is a plausible second-order cost — persistent divergences collapse the step size — but I have not yet confirmed that in a real NUTS run.

Also worth your attention:

3. neff_criteria uses criteria=nobs, but a per-event n_eff can never exceed nsamp. With 4000 < 9000 the penalty is unsatisfiable, so it contributes ≈ −11 with a permanently nonzero gradient driven by the single worst event. I kept the original default and exposed neff_criterion.
4. dVC/dz is off by up to a factor of 2 for z ≲ 0.005 (0.68 in log) — linear interpolation of a z²-like function across the first grid cell. Irrelevant above z ≈ 0.05.
5. safe_log(x, eps=1e-300) is a no-op in float32; 1e-300 underflows to 0.
6. nan_to_num(log_like_per_event, nan=0) silently treats a NaN event as likelihood 1.
7. join_point_terms computes log_mix_at_join and discards it — free at runtime (DCE'd) but either dead code or a missing continuity feature.
8. mco_floor=6.0 exceeds mco_min=4.0, so the CO power law is flat between 4 and 6. Suspicious, not verified.
9. loglik_array_dim + neff are two (9000,) arrays per sample ≈ 260 MB of output; now behind store_per_event=False.
10. mco_min in a prior file is silently ignored — build_population_model never passes it.

Open questions before I go further

- Should I swap run_inf.py over to the fast module (one import line), or leave that to you?
- neff_criterion: keep nobs, or change to a genuine per-event target?
- Want me to run bench_sampling.py to confirm the step-size/divergence effect end-to-end?

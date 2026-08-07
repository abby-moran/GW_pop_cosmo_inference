# Profiling JAX / numpyro models: a field guide

*2026-08-07.  How to measure what actually matters in a numpyro model, and
the JAX-specific pitfalls we hit in this codebase.  Companion to
`2026-08-07-jax-performance-improvements-explained.md`, which explains the
mental model (tracing, XLA, VJPs) that this guide leans on.*

## Part 1 — measure the right thing

### The unit of cost is one gradient of the potential

NUTS spends its life calling `value_and_grad(potential)` — up to
2^max_tree_depth times per sample (128 for our tree depth 7).  Nothing else
is worth profiling until you know this number.  Get the potential exactly as
numpyro builds it:

```python
from numpyro.infer.util import initialize_model
mi = initialize_model(key, model, model_args=..., model_kwargs=...,
                      dynamic_args=False, init_strategy=...)
pe_fn, z0 = mi.potential_fn, mi.param_info.z     # z0: unconstrained params
vg = jax.jit(jax.value_and_grad(pe_fn))
```

(`scripts/bench_model.py` wraps all of this, including a synthetic dataset
whose samples land inside the model's support — benchmarking on uniform junk
puts everything in the -inf tails and hides the real cost.)

### Three timing rules

1. **Block.**  JAX dispatches asynchronously; without
   `.block_until_ready()` you time the *enqueue*, not the work.
2. **Separate compile from steady state.**  The first call traces and
   compiles (20 s for this model); it tells you nothing about per-step cost.
   Time the first call separately, then take the min/median of several
   repeats.
3. **Benchmark the gradient, not the forward pass, and not sub-expressions
   in isolation.**  Forward-only profiling undercounted our real cost ~3x
   (the backward pass has its own kernels, e.g. scatter-adds) and misses
   work that only exists inside the full graph.

Rough sanity check against reality: (per-gradient time) x (steps/sample,
from `mcmc.get_extra_fields()['num_steps']`) x (number of samples) should
reproduce your wall-clock.  Ours does (9.5 ms x 127 x 3600 ~ 75 min/chain).

### Memory

```python
st = jax.devices()[0].memory_stats()
print(st['peak_bytes_in_use'] / 2**30, "GiB")
```

Peak memory in a gradient is dominated by *residuals* (forward
intermediates kept for the backward pass), so it scales with the number of
big intermediate arrays your forward pass creates — another reason fewer
passes over the data helps.  If you must trade compute for memory, that's
what `jax.checkpoint` is for (we didn't need it).

### Finding the hot spot

Two approaches, in order of effort:

- **Bisection** (what found everything in this code): time
  `value_and_grad` of partial potentials — events term only, selection term
  only, grid build only (`scripts/bench_breakdown.py`).  Crude but
  unambiguous, and it composes: ~477 ms original -> 91 ms after the
  elementwise/interp fixes -> 35 ms after the scatter fix -> bisection then
  showed the events term still dominating -> tabulating the mass function
  brought it to 18 ms.
- **The JAX profiler** (`jax.profiler.trace(logdir)` + TensorBoard, or
  Nsight) shows per-kernel times.  Useful once you need to know *which
  kernel* inside a term is slow (that's how a 13.7 ms scatter-add in the
  backward pass gets caught red-handed).

### Verify while you optimize

Every rewrite needs a referee.  `scripts/test_fast_equivalence.py` is the
pattern: compare against the untouched original *and* against float64
dense-grid references (so you know which implementation is actually more
accurate), compare gradients component by component, and — the check people
skip — compare AD gradients against **finite differences of the potential
itself**.  AD is exact for the *expression you wrote*, which is not always
the function you *meant*: our hard cutoff at m = mbhmax gave AD gradients
10-30% different from FD because moving a parameter moves probability mass
across a step that AD cannot see.  A model that samples fine can still mix
poorly for exactly this reason.

## Part 2 — pitfalls (each one bit this codebase)

### 1. Side effects inside `jit` vanish on cache hits

`numpyro.deterministic(...)` inside an inner `@jax.jit`-ed helper records
its site during the tracing call only; when the compiled function is
replayed, the site silently disappears — our runs were missing `kappa`,
`mbhmax`, `fpl`, `flow` from the output with no error.  Rules: never put
numpyro primitives (sample/deterministic/factor) under your own `jit`; and
don't decorate small helpers with `@jax.jit` at all — inside the outer jit
it buys nothing and creates exactly this hazard.

### 2. `nan_to_num` fixes values, not gradients

If `x` is NaN/inf, `jnp.nan_to_num(f(x))` has a finite *value* but its
*gradient* still propagates NaN from upstream (`exp(2*(-inf) - (-inf))` and
friends).  NUTS sees a NaN gradient as a divergence.  The fix is to guard
the *inputs* with `jnp.where` so the bad branch is never evaluated —
sometimes needing the "double-where" trick:

```python
safe = jnp.where(alive, x, 1.0)          # inner where: valid dummy input
out  = jnp.where(alive, f(safe), floor)  # outer where: select result
```

Both branches of a `where` are *computed* (and differentiated); the guard
works because `f(safe)` is finite everywhere, not because the dead branch is
skipped.  Related: `-inf - (-inf)`, `0 * inf`, `inf / inf` anywhere in the
forward pass will surface as NaN in the backward pass even if the forward
value looks fine.

### 3. Respect float32

JAX defaults to float32 on GPU.  Constants tuned for float64 silently break:
`safe_log(x, eps=1e-300)` was a no-op because 1e-300 underflows to exactly
0 in float32.  Use dtype-aware epsilons (`np.finfo(dtype).tiny`).  Also
prefer max-subtracted formulations over trusting the dynamic range.
Conversely, don't reflexively enable x64 — it doubles memory and roughly
halves GPU throughput; fix the numerics instead.

### 4. Trace-time vs run-time values (`ConcretizationTypeError`)

You cannot ask for the numerical value of a tracer (`float(jnp.max(x))`,
`if x > 0:`) inside a jitted function — shapes and graph structure must be
decided at trace time.  Compute static quantities (table bounds, grid sizes)
from the raw *numpy* inputs before they're converted to JAX arrays, and keep
genuinely static configuration (fixed hyperparameters) as Python numbers so
`if` statements on them specialize the graph for free.

### 5. Everything in the model body runs per leapfrog step

A numpyro model is re-executed (as a compiled graph) at every gradient
evaluation.  `jnp.log(data)`, Jacobian factors of the data, anything
parameter-independent: hoist it out of the model (or compute it once and
close over it).  Watch dtype promotion rules too — a stray Python int can
promote arrays (and under `jax_enable_x64` even break dtype-matching ops,
as our `lax.rem(int32, int64)` did).

### 6. Ops whose *backward* is the problem

The VJP of a gather (table lookup) is a scatter-add; into a small table it
serializes on atomics and can cost 25x the forward gather.  Replicating the
table (32 copies, adjacent points hitting different copies) fixed it here.
More generally: when the gradient is mysteriously slower than ~2-3x the
forward pass, suspect a specific backward kernel and go look at the profile.

### 7. Hidden per-element algorithms

`jnp.interp` runs a binary search per point; `searchsorted` likewise.  On
uniform / log-uniform grids replace them with closed-form index arithmetic.
Chains of `logaddexp`/`logsumexp` do redundant exp/log/max plumbing —
subtract one max and reduce in linear space.

### 8. Recompilation

Every new shape/dtype (and every new value of a *static* argument)
recompiles the whole graph — for this model, ~20 s a pop.  Symptom: the
"occasional" multi-second call.  Keep shapes fixed (pad if needed) and
static arguments few-valued.  `jax.log_compiles(True)` tells you when and
why recompiles happen.

### 9. Dead code is free — use that fact

XLA removes anything not needed for the outputs.  Diagnostics recorded with
`numpyro.deterministic` cost *nothing* during sampling (the potential
doesn't return them; they're evaluated only in postprocessing), so record
generously — `min_neff`, `mc_var_loglike` — instead of adding penalty terms
"to be able to see the value".  The flip side: code you *think* is doing
something may be DCE'd (the discarded `log_mix_at_join` looked load-bearing
for a year while costing nothing and doing nothing).

### 10. Kinks and hard cutoffs

`min`, `max`, `abs`, `jnp.where`-as-step: AD returns a one-sided derivative
at the kink (fine, measure zero), but a *step discontinuity in the density*
is a modeling bug for HMC — the gradient is blind to it everywhere (see the
AD-vs-FD check above), and hinge-shaped penalties produce gradient jumps
that cause divergences right where the penalty activates.  Prefer smooth
constructions: `softplus` instead of `min(0, .)` hinges, smooth turn-ons
instead of hard cutoffs — the cost is invisible next to anything else in
the model.

## A checklist for the next optimization pass

1. `bench_model.py` (or equivalent): compile time, forward, gradient, peak
   memory, at production scale.  Write the numbers down.
2. Estimate wall-clock: gradient x 2^tree_depth x samples.  Does it match?
3. Bisect the gradient cost by term; profile kernels only where bisection
   points.
4. After *every* change: equivalence vs reference, gradient components,
   AD vs FD of the potential, and the dead-event / out-of-support edge case.
5. Re-measure.  Keep the harness in the repo — the next person will need it.

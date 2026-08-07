# Why the fast model is fast: each JAX optimization, explained

*2026-08-07.  A pedagogical tour of `src/intensity_models_fast.py` for
someone who uses JAX but hasn't looked under the hood.  Numbers are for
production scale (9000 events x 4000 PE samples, 1.7M selection samples) on
an A6000 unless stated.*

## A 60-second model of how JAX actually runs your code

Three facts explain almost every optimization below.

**1. Your Python runs once; the GPU runs a compiled graph.**  Under
`jax.jit` (and numpyro jits the whole potential for you), JAX calls your
Python function a single time with abstract *tracers* — objects that record
shape and dtype but hold no values — and writes down every jnp operation
into a graph (a "jaxpr").  XLA compiles that graph into fused GPU kernels;
subsequent calls with the same shapes/dtypes replay the compiled binary and
never touch your Python again.  Consequences:

- Python-level work (`if mpisndot == 0:`, building numpy constants, string
  logic) is **free** — it happens once, at trace time.  This is why we can
  choose completely different graph structures based on static properties.
- Python *side effects* (printing, appending to lists, registering numpyro
  sites inside an inner `@jax.jit`) happen only on the tracing call and
  silently vanish afterwards — the source of a real bug we fixed.
- Anything not needed for the outputs is deleted by the compiler
  (dead-code elimination), so "computed but unused" costs nothing.

**2. The gradient is another graph, and it's what NUTS pays for.**  NUTS
evaluates `value_and_grad(potential)` at every leapfrog step — up to
2^max_tree_depth = 128 times per sample.  Reverse-mode AD builds the
backward graph mechanically: every operation gets a *vector-Jacobian
product* (VJP), and intermediate values ("residuals") from the forward pass
are kept in GPU memory until the backward pass consumes them.  So the
gradient costs ~2-3x the forward pass in FLOPs, its memory footprint is set
by the residuals, and — crucially — *the VJP of a cheap operation can be
expensive* (see gathers below).

**3. At these sizes the GPU is memory-bandwidth-bound.**  The big arrays
are (9000, 4000) float32 ~ 144 MB.  Most of our operations are elementwise,
so runtime ~ (number of passes over the big arrays) x (bytes per pass) /
bandwidth.  The two ways to be faster: touch the big arrays fewer times
(fusion, precomputation, fewer passes), and make each touch cheap
(no transcendentals or binary searches per element, contiguous access).

With that, the optimizations:

## 1. Replicated lookup tables (the scatter-contention fix)

The model interpolates several small tables (e.g. the 514-node PISN mass
grid).  The forward `gather` — read `table[i]` for 36M different `i` — is
almost free.  But its VJP is a **scatter-add**: 36M gradient contributions
must be *added into* ~514 table slots.  On a GPU those adds are atomic
operations, and when thousands of threads target the same handful of slots
they serialize.  Measured in isolation: 0.5 ms forward, **13.7 ms
backward** — and it gets *worse* for smaller tables (fewer slots, more
contention).

Fix: keep `SCATTER_REPLICAS = 32` copies of each table, and have adjacent
points (adjacent lanes of a GPU warp) index different copies, so concurrent
atomics land on different memory addresses.  The forward pass broadcasts the
table into its replicas; the VJP of a broadcast is a sum over replicas,
which XLA does efficiently — so the 32 partial scatters get combined for
free.  Backward time for the gathers dropped ~10x.

Lesson: *in JAX the cost of an op includes the cost of its VJP.*  Gathers
from small tables are the canonical trap.

## 2. Closed-form grid indices instead of binary search

`jnp.interp(x, grid, vals)` does not know the grid is uniform, so it runs a
~10-iteration binary search per point — ten dependent gather passes over
36M elements, five times per potential evaluation.  Every grid in this model
is uniform in `log(x)` or `log1p(x)`, so the cell index is one formula:
`i = (log(x) - log(x0)) / dlog`.  One fused elementwise kernel replaces the
search entirely.

Lesson: convenience functions hide per-element algorithms.  If you know the
grid structure, exploit it.

## 3. One fused cosmology lookup instead of three

The likelihood needs `z(dL)`, and a Jacobian combination
`log(dVC/dz) - log(ddL/dz) - 2 log1p(z)` for every sample.  The original
interpolated three separate tables (three index computations, three
gathers).  Since `Om` and `w` are fixed in these runs, the tables are
compile-time constants, so we precompute a single table of the *combined*
Jacobian on a log-uniform axis in `log(dL)` and fetch `(log1p z, J)` in one
lookup.  Bonus: the rebuilt table is denser at low z, fixing a factor-2
error in dVC/dz below z ~ 0.005 that the original's coarse first cell
produced.

Lesson: fold every per-sample computation you can into precomputed tables,
and fetch them together.

## 4. Max-subtracted trapezoid for the PISN integral

Building the PISN grid integrates over mco in log space.  The original
chained `logaddexp`/`logsumexp` ops — each one is exp + log + max plumbing,
and chains of them make long dependent graphs (long backward graphs too).
Numerically, all that machinery buys is protection against overflow, which
you can get once: subtract the running max `M`, do a plain linear trapezoid
on `exp(x - M)`, and add `M` back after the log.  One `exp`, one reduction.
The mco axis was also moved to be the *last* (contiguous) axis so the
reduction reads coalesced memory.

Lesson: log-space safety has a cost; pay it once per reduction, not per
operation.  And reductions want the reduced axis contiguous.

## 5. Static shortcuts: `mpisndot == 0`

When `mpisndot` is pinned to 0 in the prior, the single-mass function has no
z-dependence.  Because that pin is visible at *trace time* (it's a Python
float, not a tracer), plain Python `if` statements can build a structurally
different graph: the PISN grid gets 1 z-slice instead of 30, and its
interpolation collapses from 2-D to 1-D.  No runtime branching is involved —
the compiled graph simply never contains the z axis.

Lesson: static configuration is free in JAX; use Python control flow on it
aggressively.  (Only *traced* values need `jnp.where`/`lax.cond`.)

## 6. Fused logsumexp + n_eff over the weight array

Per event we need `logsumexp(w)`, and the n_eff diagnostic needs
`logsumexp(2w)` — the original computed them in separate passes over the
(9000, 4000) array, each with its own max-subtraction.  `_logsumexp_and_neff`
does one max, one `exp`, and two cheap reductions of the same intermediate,
i.e. one pass over the 144 MB instead of several.  It also handles dead
events with `where`-guards so values *and gradients* stay finite (see the
profiling note for why `nan_to_num` cannot do this).

## 7. The tabulated mass function (the big structural win)

Even after all of the above, the dominant cost was evaluating the mass
function `log dN/dm` — a mixture of a PISN component (table lookups), a
Gaussian peak, a power-law tail with smooth turn-ons — at 72M sample masses
(m1 and m2 for every PE and selection sample), with transcendentals per
mixture component per point.

But when mpisndot = 0 the mass function depends only on m, not z.  So: per
likelihood call, evaluate it **once** on an 8192-node log-uniform mass grid
(8192 evaluations instead of 72M), then reduce every per-sample evaluation
to a 1-D linear interpolation — one closed-form index plus one replicated
gather.  Gradients flow through the table values into the hyperparameters
automatically; AD differentiates the table build (cheap, it's 8192 points)
and the lerp (linear).

This roughly halved the gradient time again (35 -> 18 ms) and cut peak
memory (the residuals of 72M-point transcendental chains disappear).  It
also *improved* the gradients: linear interpolation smears the model's step
discontinuity at m = mbhmax over one grid cell, which together with
`smooth_tail_edge=True` makes the AD gradient of the potential agree with
finite differences (with the original's hard edge, d/dh, d/dmpisn,
d/ddmbhmax were 10-30% wrong — AD cannot see probability mass crossing a
hard step as a parameter moves).

Lesson: the best JAX optimization is often mathematical, not mechanical —
find the low-dimensional structure (here: a 1-D function evaluated 72M
times) and factor it out.

## 8. Hoisting constants out of the sampler

`log(m1)`, `log(q)`, `log(pdraw)`, the selection Jacobian factors — all
depend only on the *data*, not on the sampled parameters, yet the original
recomputed them inside the potential, i.e. at every leapfrog step.  Computed
once at model-build time, they are baked into the compiled graph as
constants.  Three fewer `log` passes over the big arrays per step.

Lesson: inside a numpyro model, everything runs per leapfrog step.  Ask of
every line: does this depend on a sampled parameter?  If not, move it out.

## What we deliberately did *not* do

- No float64: float32 is ~2x faster and 2x smaller on GPU, and the
  max-subtracted formulations keep it accurate (verified against float64
  references — the fast module is closer to them than the original was).
- No `jax.checkpoint` (rematerialization): after the tabulation the
  residuals fit comfortably (~6 GiB peak), so trading compute for memory
  wasn't needed.
- No multi-GPU sharding: one gradient is 9.5 ms on an H100; chains are the
  natural parallel unit.

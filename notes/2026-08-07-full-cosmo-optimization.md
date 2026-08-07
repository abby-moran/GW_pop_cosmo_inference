# Full-cosmology optimization (Om, w free)

Companion to `2026-08-07-optimization-changelog.md`.  That pass specialized
the hot path for the production setup where only `h` is sampled.  This note
covers what changes when `Om` and `w` are free too, and what we did about it.

## Why the h-only wins mostly carry over

Most of the earlier work does **not** depend on cosmology being fixed:

- Mass-function tabulation is gated only on `mpisndot == 0`.
- The fused `dL → (log1p z, J)` lookup still indexes by
  `u = log(dL/dH)`; `h` enters only as scalar shifts.
- Scatter replication, fused logsumexp/n_eff, neff-penalty redesign, and
  float32 recentering are cosmology-agnostic.

What *does* get more expensive: when `Om`/`w` are tracers, the cosmology
(and, on the tabulated path, the combined cosmology+rate-density) tables
stop being compile-time constants.  Reverse-mode AD through a gather from a
traced table becomes a scatter-add of one cotangent per data point — the
same failure mode the replication trick was built for, now paid again for
every cosmology channel.

## Scatter-free custom VJP

Linear interpolation is linear in the table values, so for a table
`T(u; θ)` that depends on a handful of scalars `θ`,

```
d/dθ_k lerp(T, t)  =  lerp(dT/dθ_k, t)  +  (T[i1]-T[i0]) * dt/dθ_k
```

The second term is the ordinary index cotangent (kept on the normal AD
path).  The first is what reverse mode normally realizes as a scatter into
`T`; we instead:

1. Build the primal table and its forward-mode tangents `dT/dθ_k` once per
   likelihood call (`_build_table_with_tangents`, K = number of *traced*
   parameters among the table's inputs — statically fixed ones need no
   tangent).
2. Look up with `_table_lookup_fewparam`, a `jax.custom_vjp` whose backward
   pass gathers from the tangent tables and reduces.  The primal table is
   `stop_gradient`'d so the scatter never happens.

Channels are interleaved into one `(n, C)` array (`log1p z` and `J`, or on
the tabulated path `log1p z` and `J + log dN/dVdt`), so one index
computation and adjacent memory serve both.  `2 log(dH)` is added *after*
the lookup (exact: lerp weights sum to 1), which keeps `h` out of the
table's parameter list.

This is only a win while `K` is small.  The cosmology / rate-density tables
have `K ≤ 6` (`Om, w, zmax, lam, kappa, zp`).  The tabulated *mass*
function still uses replicated scatter on purpose: it depends on ~a dozen
sampled parameters, where K gather+reduce passes lose to 2 replicated
scatters.

Even when `Om`/`w` are fixed, the tabulated path's combined
cosmology+rate-density table still uses the custom VJP for `(lam, kappa,
zp)` — a free win on the production setup.

Reference implementation without the custom VJP:
`src/im_fast_baseline.py` (A/B with
`bench_model.py --module im_fast_baseline --cosmo_free`).

## Closed-form setup indexing

`_dimless_dl_tables` builds the dimensionless tables.  The z-grid is
log1p-uniform, so cell indices into it are arithmetic (replacing
`jnp.interp`'s binary search for the lower-edge and `dc(z)` lookups).  The
inverse `z(u)` lookup still uses `jnp.interp`: `dL/dH(z)` is a
model-dependent monotone table with no closed-form index, and it runs once
per likelihood call over ~2k nodes — negligible next to the data arrays.

## Omh2 reparameterization

With `(h, Om, w)` all free, the data mainly constrain the distance–redshift
relation over the detected range, so the three parameters are strongly
degenerate and NUTS pays in tree depth.  A prior file may sample the
physical density `Omh2 = Om * h^2` instead of `Om`;
`get_deterministic_parameters` then records

```
Om = Omh2 / h^2
```

as a `numpyro.deterministic`.  `Omh2` is the CMB-constrained combination and
is much less degenerate with `h` than `Om` itself.  Example prior lines:

```
h = TruncatedNormal(0.7, 0.2, low=0.4, high=1.2)
Omh2 = TruncatedNormal(0.143, 0.05, low=0.02, high=0.4)
w = TruncatedNormal(-1.0, 0.3, low=-2.0, high=-0.3)
```

Do **not** also set `Om` in the prior when using `Omh2` (the deterministic
is only installed when `Om` is absent from the sampled dict).

## Measured (A6000, production scale, 2026-08-07)

9000 events × 4000 PE samples + 1.7M selection, float32, Om/w free,
tabulated mass function on:

| | grad (min) | forward | peak mem |
|---|---|---|---|
| custom VJP (`intensity_models_fast`) | 16.5 ms | 4.3 ms | 5.7 GiB |
| replicated scatter (`im_fast_baseline`) | 18.6 ms | 4.7 ms | 6.0 GiB |

~11% on the leapfrog step.  Replication already made the scatters tolerable;
the custom VJP is the right architecture when Om/w are free and is a free
win for `(lam, kappa, zp)` even when they are not.  The bigger lever once
Om/w are free is conditioning (Omh2 reparameterization above), which shows
up in tree depth / ESS rather than ms/leapfrog.

## How to exercise

```
# gradient cost with Om, w free (custom VJP path)
uv run python bench_model.py --module intensity_models_fast --cosmo_free

# same, with Omh2 parameterization
uv run python bench_model.py --module intensity_models_fast --cosmo_free --omh2

# A/B against the pre-custom-VJP snapshot
uv run python bench_model.py --module im_fast_baseline --cosmo_free

# equivalence + AD-vs-FD + Omh2 chain rule
uv run python test_fast_equivalence.py
```

## What we deliberately did not do

- Precomputed 3-D tables over `(u, Om, w)`: would make tables constants again
  but introduces a discretization error and 8 gathers per lookup.  The
  custom VJP is exact and needs no error budget.
- Reparameterizing `w` (e.g. via `H(z_pivot)`): worth exploring on short
  chains for ESS/wall-time, but it changes the sampled posterior
  coordinates and is left to the prior file / analysis choice.

# 2-D mass-function table: fast path for sampled mpisndot

*2026-08-07.  Branch `local/mass-table-2d`.  Extends the tabulated
mass-function evaluation in `src/intensity_models_fast.py` from 1-D (log m)
to 2-D (z x log m) so that sampling `mpisndot` no longer falls back to the
direct per-sample evaluation.  Companion notes:
`2026-08-07-optimization-changelog.md` (the full series),
`2026-08-07-jax-performance-improvements-explained.md` (why tabulation wins).*

## The problem

The single-mass function `log_dndm(m, z)` depends on redshift only through

    mpisn(z) = mpisn + mpisndot * z / (1 + z)

With `mpisndot` pinned to 0 the function is z-independent, so the fast model
evaluates it once per likelihood call on an 8192-node log-m grid and every
per-sample evaluation over the (nobs, nsamp) PE array and the selection set
becomes a single 1-D lerp.  That tabulation carried roughly a 4x speedup.

Sampling `mpisndot` used to disable it: every one of the ~38M points went
through the direct evaluation (bilinear gather into the PISN grid, tail,
bump, window, two logaddexp chains), costing ~67.5 ms per gradient and
~20 GiB peak (vs ~16.5 ms / ~5.7 GiB tabulated) at production scale
(nobs=9000, nsamp=4000, nsel=1.7M) on an H100.

## The fix

The z dependence is a smooth one-parameter shift, so the table simply gains
a z axis:

- **Grid.** `f_tab[i, j] = log_dndm(m_j, z_i)` with `m_j` the same 8192-node
  log axis as before and `z_i` the *same* `n_z = 30` log1p-uniform nodes the
  PISN mass grid already uses (`LogDNDM.z_array`).  Building the table costs
  30 x 8192 = 245k direct evaluations once per likelihood call -- less than
  1% of the 38M per-sample evaluations it replaces.  Because the z nodes
  coincide with the PISN grid slices, the PISN component of the table is
  exact at every node: no interpolation error stacks on top of the z-interp
  the direct path already does.

- **Lookup.** Per sample: closed-form fractional index on the log-m axis
  (as before), plus the z cell from `_Log1pAxis.cell_and_frac` with the
  *same linear-in-z weights* the direct path's PISN interpolation uses, then
  one bilinear lerp (`_gather_lerp2d`, which also handles the replicated
  scatter mitigation for the backward pass).  The m1 and m2 lookups share
  the z cell, so the extra per-sample cost over the 1-D case is one `expm1`
  and two more gathers.

## Selection stays on the direct path (important!)

First attempt used the 2-D table for the selection set too, and the
equivalence test failed by ~1.3 nats at nobs=400.  Per-factor tracing showed
the per-event log likelihoods agreed to 0.009 nats total; the entire
difference was the selection factor.  The z-lerp of the *combined*
log-density (logaddexp of pisn + tail + bump, minus the lerped log Z)
carries an O(3e-3) systematic bias in `log_mu_sel`, and `selfactor`
multiplies `log_mu_sel` by nobs -- a negligible per-point error becomes an
O(nobs * 3e-3) parameter-dependent distortion (~27 nats of potential
variation at nobs=9000).

The per-event likelihood terms have no such amplification (each event's
log-lse moves by ~5e-4 typ, 0.02 max), and the selection set is only ~5% of
the points.  So in the 2-D case the table serves the (nobs, nsamp) PE array
and the selection weights keep the direct evaluation.  With that split the
tab-vs-direct potential difference at nobs=400 drops to 9e-3 nats and every
per-parameter gradient agrees to <= 0.6% (smooth_tail_edge=True pair).

## AD vs FD with mpisndot free: FD is the fragile side

`test_tabulated_path_zdep` checks AD gradients against central finite
differences of the same potential.  Two FD pitfalls specific to this
configuration, both verified to affect the *direct* path identically (so
they are model structure, not table artifacts):

1. The PISN remnant map's `where(mco < mpisn(z_i), ...)` branch kinks the
   potential whenever some `mpisn(z_i)` crosses an mco grid node.  With 30 z
   slices the kinks are ~9e-3 Msun apart in mpisn, so an FD step of 3e-3
   (fine for the mpisndot=0 test) straddles them: FD swings
   81 -> 69 -> 30 -> 41 over eps = 1e-3 .. 3e-2 while AD sits at 82.6 --
   identically for direct and tabulated.  The test uses eps = 1e-3, where FD
   and AD agree to ~2-3%.

2. With the hard tail edge (`smooth_tail_edge=False`) an h step moves every
   sample across the moving discontinuity at 30 distinct mbhmax(z_i)
   positions, so d/dh FD is not a usable reference at any step size; that
   check is informational-only in the z-dependent test.  The recommended
   `smooth_tail_edge=True` path must pass AD-vs-FD for all of
   {h, mpisn, dmbhmax, mpisndot} (measured: 0.1-3.3%).

## Interface

- `pop_cosmo_model(tabulate_mass_function=...)` default (None) now means
  *enabled always*: 1-D table when mpisndot is statically 0, 2-D otherwise.
  `False` still selects the direct per-sample evaluation everywhere.
- `scripts/bench_model.py --no_tab` forces the direct path for A/B timing.
- `scripts/test_fast_equivalence.py` gained `test_tabulated_path_zdep`
  (test 7): potential + all-parameter gradient equivalence tab2d-vs-direct,
  finite-grad check, and AD-vs-FD on the edge parameters including mpisndot.

## Measured (float32, nobs=9000, nsamp=4000, nsel=1.7M, mpisndot free)

Same GPU, back-to-back A/B via `bench_model.py --module intensity_models_fast
--mpisndot_free [--no_tab]`:

| | direct (`--no_tab`) | 2-D table (default) | |
|---|---|---|---|
| potential (forward) | 5.5 ms | 5.1 ms | ~1x |
| gradient (per leapfrog) | 68.5 ms | 38.0 ms | 1.8x |
| peak GPU memory | 20.4 GiB | 9.4 GiB | 2.2x |

Potential values at the truth point agree to 1e-4 relative (0.1 nats at
nobs=9000).  For reference, the mpisndot-pinned 1-D table sits at ~16.5 ms
on the same hardware: the 2-D case keeps paying for the 30-slice PISN grid
build (~16M-element trapezoid per call plus its backward), the 245k-point
table build through the direct evaluation, the direct selection set, and
two extra gathers + one expm1 per sample for the bilinear lookup.  Sampling
mpisndot now costs ~2.3x the pinned case instead of ~4x.

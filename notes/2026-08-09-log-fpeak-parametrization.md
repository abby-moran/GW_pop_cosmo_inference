# Peak-height parametrization of the low-mass bump (`log_fpeak`)

*2026-08-09.  Follow-up to
`notes/2026-08-09-low-mass-bump-width-identifiability.md`.*

## Definition

`get_deterministic_parameters` (both `intensity_models_fast.py` and the
reference `intensity_models.py`) now accepts a fourth bump-amplitude
parametrization in the prior file:

```text
log_fpeak = log_flow - log(msigma_low)
```

i.e. `flow = exp(log_fpeak) * msigma_low`.  `exp(log_fpeak)/sqrt(2 pi)` is
the bump's peak density relative to the unit-weight PISN component; up to
that constant, `log_fpeak` is the amplitude the data actually measure.
`log_flow` and `flow` are recorded as numpyro deterministics, so downstream
tooling and comparisons with older chains keep working.

## Why

The data constrain the bump's peak height, not its integrated weight, so
sampling `log_flow` builds in an amplitude-width correlation.  Measured
consequences (Fisher at the width-2 truth / narrowbump posterior):

| | `log_flow` coords | `log_fpeak` coords |
|---|---|---|
| ρ(amp, width), Fisher | +0.83 | +0.54 |
| ρ(amp, width), chain | +0.67 | +0.43 |
| ρ(a, amp), chain | −0.39 | −0.25 |

Alternatives tested and rejected: amplitude as contrast over the local PISN
continuum at `mp_low` (worse — the transform injects `a`, `b`, `mpisn` into
the amplitude coordinate; ρ(a, amp) → −0.97 and 2× worse conditioning).
Reparametrization does not change identifiability at all (σ(a) is exactly
invariant); the width cap is what fixed `a`.

## Prior

`runs/priors/gwtc5_massonly_fpeak.prior` replaces
`log_flow = Uniform(np.log(1e-3), np.log(2))` with

```text
log_fpeak = Uniform(np.log(1e-3), np.log(4))
```

and applies the `msigma_low` cap (`TruncatedNormal(1.5, 1.0, low=0.5,
high=2.5)`) cleared by the broad/narrow confirmation runs.  This prior is
flat in `log_fpeak`, **not** flat in `log_flow` (they differ by the
`log msigma_low` Jacobian, uniform-shifted per width); remember the
difference when comparing to pre-fpeak runs — a deliberate choice, not an
oversight.

## Plumbing

`run_inf.load_true_vals` and `plotting.load_true_vals` derive
`log_fpeak = log_flow - log(msigma_low)` from truth configs, so
`init_to_value`, `recentering_baselines` and corner-plot truth markers work
under either parametrization (both look up sites by name and ignore extra
keys).

## Non-interference

The change is purely additive: existing prior files hit the unchanged
`log_flow` branch bit-for-bit, so the in-flight evo4 job (Slurm 6790523,
reads `gwtc5_fullcosmo_evo.prior`) is unaffected even if it restarts.  No
existing prior file or run config was modified.  Other prior files can adopt
the parametrization by making the same two-line change.

Sampling `log_flow` still works but prints a one-shot deprecation warning
(`utils.warn_log_flow_deprecated`) from `run_inf.py` at prior load and from
`get_deterministic_parameters` when that branch is taken.

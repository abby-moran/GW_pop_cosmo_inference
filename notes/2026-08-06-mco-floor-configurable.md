# `mco_floor` is now settable from the config files

*2026-08-06.  Applies to `src/intensity_models_fast.py` only.  Defaults are
unchanged: runs that don't mention `mco_floor` behave exactly as before.*

## What `mco_floor` is

The CO-core IMF in this model is the broken power law of Golomb, Isi & Farr
(2024, arXiv:2312.03973, Eq. 2) — indices `-a` below the fixed break at
20 Msun and `-b` above — except that `log_dNdmCO` **flattens the power law
below `mco_floor` (default 6 Msun)**: the density is constant on
`[mco_min, mco_floor]`.

This is deliberate, even though it looks odd next to `mco_min = 4` (the
original source even carried a `# TODO: why is mco_floor needed?`).  Since
the remnant map is the identity below `mpisn`, CO cores of 4–6 Msun feed BH
masses of ~4–6 Msun directly, and an un-floored power law (the prior lets
`a` reach ~6) would diverge toward the arbitrary cutoff at `mco_min`.  That
would

- pile density at the cutoff, letting the total rate and the selection
  normalization be controlled by the least observable corner of the model
  (for `a > 1` the power law is non-integrable toward zero mass);
- force `a` to fit both the 6–20 Msun slope and the 3–6 Msun region, making
  it degenerate with the low-mass bump parameters (`flow`, `mp_low`,
  `msigma_low`) that were added to this model precisely to describe the
  low-mass structure; and
- concentrate the `mco` quadrature's integrand in its first few grid cells.

Flattening (rather than truncating) keeps support down to `mco_min`, so
low-mass secondaries stay inside the population and the lower edge of the
mass function is set by the `mbh_min` turn-on window, not by the CO cutoff.
The full argument lives in the `log_dNdmCO` docstring.

## What changed

`mco_floor` used to be a hardcoded default argument of `log_dNdmCO` that
nothing ever passed.  It is now piped end to end, exactly like `mco_min`:

```
prior / pop-config file
  -> build_population_model:  mco_floor=sample.get('mco_floor', 6.0)
    -> LogDNDMDQDV (field, default 6.0)
      -> LogDNDM (field)
        -> LogDNDMPISN (field)
          -> log_dNdmCO_from_log(..., mco_floor=...)   # the CO-IMF integrand
```

Because it is a real dataclass field on `LogDNDMDQDV`, it also survives the
`getfullargspec` filter that `reweight_res.py` applies to population-config
keys — so the same key works in both halves of the workflow.

## How to use it

Add one line to the relevant config; omit it to get the old behavior:

- **Inference** (`run_inf.py`): add `mco_floor = 6` to the `.prior` file.
  Fixed floats there become `numpyro.deterministic` sites, so the value is
  also recorded in the MCMC output.
- **Population generation** (`reweight_res.py`): add `mco_floor=6` to the
  `pop_configs/*.txt` file.

If you change it, change it in **both** places — the generated population
and the inference model should use the same mass function (same rule as for
`smooth_tail_edge` and `mco_min`).

## Gotchas

- **Fast module only.**  `intensity_models.py` and `intensity_models_coup.py`
  still hardcode 6.0; they are kept frozen as the reference implementations
  for the equivalence tests.
- **`mco_floor <= mco_min` is legal but inert-ish**: at `mco_min` the
  flattening is disabled entirely (the pure power law extends to the
  cutoff), and below `mco_min` the floor does nothing because the
  smooth turn-on at `mco_min` already suppresses that region.  There is no
  validation of `mco_floor >= mco_min`.
- **Don't sample it casually.**  A distribution in the prior file would
  technically work (the code handles traced values — a fixed float from a
  prior file already arrives as a jnp scalar, which is why
  `log_dNdmCO_from_log` branches between `np.log` and `jnp.log`), but a
  sampled flattening scale moves a kink of the integrand across the `mco`
  grid, and nothing about that has been tested for sampler health.

## History / verification

- Before this change, `mco_min` in a prior file was silently ignored by
  `build_population_model`; that was fixed earlier in the same optimization
  pass, and `mco_floor` now follows the identical pattern.  All existing
  pop configs set `mco_min=4` and no `mco_floor`, so nothing changes for
  them.
- `scripts/test_mco_floor.py` checks the whole chain: the flattening is
  actually applied, an explicit `mco_floor = 6` in a prior file reproduces
  the default potential, `mco_floor = 10` / `= 4` measurably change it (i.e.
  the value is not silently dropped), and each dataclass layer receives the
  value.  `scripts/test_fast_equivalence.py` passes unchanged.
- Related note: `2026-08-06-join-point-machinery-removed.md` documents
  another piece of archaeology in the same mass function.

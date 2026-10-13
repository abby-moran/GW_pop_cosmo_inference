# GWTC-4/5 PE prior fix: UniformSourceFrame distance prior

## Provenance: what changed on `main` (ported 2026-10-12, commit 0b96ec3)

This note was written on the `lvk-powerlaw-z` branch, where the pre-fix
code really was off by a factor (1+z)(ddL/dz)^2 per sample (see the
branch history below).  **That was never the state of `main`.**  Before
the port, `main`'s `get_samples_from_event` already computed

    prior = dVc/dz * m1_det / (ddL/dz * (1+z))        (Planck18, file's `redshift` column)

with both 1/(1+z) and 1/(ddL/dz) in the denominator, i.e. the correct
functional form.  The "The bug" / "Impact" / "Quarantine" sections below
(2.1x factor, 190 of 259 events, voided runs) therefore do **not** apply
to PE files built on `main`: GWTC-4p0 / 4p1 / 5p0 files extracted with
`main`'s pre-port code were NOT off by the ~2x factor.  What 0b96ec3
actually changed for `main` is:

1. **Cosmology / redshift source.**  Planck18 evaluated at the file's
   `redshift` column is replaced by Planck15_LAL (H0 = 67.90, Om0 = 0.3065,
   bilby's `cosmology='Planck15_LAL'`) evaluated at z(dL) interpolated from
   the `luminosity_distance` column.  This is the cosmology the declared
   prior is actually defined in; the change is a ~1e-3-level, smooth
   per-sample correction (the within-event std of the old-vs-new log-ratio
   is 1.4e-3 median, 2.4e-3 max; see "Impact").
2. **Unmasked `(1+zs)` broadcast bug.**  The old `main` line divided by
   `(1+zs)` using the *unmasked* `zs` while `m1_det`, `dvcdz`, `ddl_dz` were
   already masked by `redshift < zmax`, so the expression raised a broadcast
   `ValueError` whenever any PE sample had z >= zmax.  Files for which it did
   not raise were computed correctly (shapes matched because no sample was
   cut).  The port computes everything from the masked arrays.
3. **GWTC5p1 added to the release tuple.**  Previously a `GWTC5p1` file
   name fell through to the O1-O3 branch, `prior = dL^2 * m1_det`.  That is
   a genuine prior error (PowerLaw(alpha=2) in dL instead of
   UniformSourceFrame) for any 5p1 file; no 5p1 file is known to have been
   used on `main` before the port.
4. **Plumbing.**  `group=` kwarg on `get_samples_from_event` (bypasses the
   waveform-preference lists; main keeps that preference ladder as the
   default), `group_used` is recorded, and `_check_gwtc45_analytic_prior`
   verifies the declared `luminosity_distance` / `chirp_mass` /
   `mass_ratio` bilby classes (raises on mismatch; one-line warning naming
   the group when the block or an entry is absent).

Everything below this line is the branch-side account, kept for the
derivation and the evidence; read its "old code" as the branch's.

---

## On the `lvk-powerlaw-z` branch (pre-fix code there)

*2026-09-03.  Branch `fix-selection-normalization`.  Bug in
`weighting.get_samples_from_event` affecting every real-data GWTC-5 run
made so far on that branch (all `runs/realGWTC5_*`, now quarantined in
`runs/bad_z_prior/`).  The selection side and the O1-O3 PE branch were
correct and are unchanged.*

## The bug (branch code)

`get_samples_from_event` returns `(m1_det, q, dL_Gpc, prior)`, where
`prior` must be the PE sampling-prior density in (m1_det, q, dL); the
population model divides the per-sample population density by it
(`intensity_models.py`, `pop_cosmo_model`).  For GWTC-4.x/5.x release files
the code computed

    prior = dVc/dz * m1_det * ddL/dz          (Planck18, file's own z column)

The files declare (`f[group]['priors']['analytic']`)

    luminosity_distance = bilby.gw.prior.UniformSourceFrame(cosmology='Planck15_LAL', ...)
    chirp_mass          = UniformInComponentsChirpMass(...)
    mass_ratio          = UniformInComponentsMassRatio(...)

UniformSourceFrame is p(z) ∝ dVc/dz / (1+z) (uniform in comoving volume and
source-frame time); to a density in dL it is divided by
ddL/dz = D_C(z) + (1+z) D_H / E(z).  Uniform-in-components masses give
p(m1_det, q) ∝ m1_det.  So the correct density is

    prior = m1_det * dVc/dz / ((1+z) * ddL/dz),   z = z(dL | Planck15_LAL)

i.e. the branch's old value was off by a factor (1+z) (ddL/dz)^2 per
sample.  Per-event constants (units, sr, bounds, normalization) cancel in
the hierarchical likelihood and are ignored.
(`main`'s pre-port code, by contrast, already had the correct form but
divided by an unmasked `(1+zs)` -- see item 2 of the provenance section
above; the port computes the factor from the masked arrays.)

## Fix

`src/weighting.py`: new `_gwtc45_pe_prior(m1_det, dL_Gpc)` computes z from
dL under `PLANCK15_LAL = FlatLambdaCDM(H0=67.90, Om0=0.3065)` (bilby's
'Planck15_LAL', NOT astropy Planck15 with H0=67.74, which is what the
files' `redshift` column corresponds to) via a 20000-point grid and
`np.interp`, then the formula above.  A guard,
`_check_gwtc45_analytic_prior`, re-opens the file and raises `ValueError`
if the declared `luminosity_distance` / `chirp_mass` / `mass_ratio` strings
are not UniformSourceFrame / UniformInComponentsChirpMass /
UniformInComponentsMassRatio; if the group has no `priors/analytic` block
(87 of the GWTC-5 Stable_Release-8 groups, mostly `C00:SEOBNRv5PHM`) or the
block lacks one of those entries, it prints a one-line warning naming the
file and group that the prior is being assumed.
The group-selection ladder now records `group_used` for this purpose; its
priority order is unchanged.

## Evidence

- Declared analytic priors in the GWTC-5 files (e.g. GW240413_022019, all
  three waveform groups) are exactly the three bilby classes above with
  `cosmology='Planck15_LAL'`, minimum=10, maximum=4000 Mpc.
- `prior / m1_det` for GW240413 (C00:IMRPhenomXPHM-SpinTaylor, 15179
  samples) divided by
  `bilby.gw.prior.UniformSourceFrame(10, 4000, cosmology='Planck15_LAL').prob(dL_Mpc)`
  is constant to 3.4e-4 relative (ratio 3261.70 - 3262.80) over
  dL = 0.12-0.80 Gpc.
- The `_cosmo` vs `_nocosmo` trap for GWTC-2.1/3: the `_cosmo` files were
  reweighted to a comoving-volume distance prior, but their
  `priors/analytic` block and `log_prior` column still show the original
  `PowerLaw(alpha=2)` distance prior.  The code uses the `_nocosmo` files,
  for which `prior = dL^2 * m1_det` is correct (verified unchanged, exact
  equality, on GW191103_012549 C01:Mixed).

## Impact (branch runs only; does not apply to `main`-built files)

190 of the 259 events in `PE_GWTC5_259ev.h5` are GWTC-4.1/5.0 (the 69
GWTC-2.1/3.0 events are unaffected).  The spurious factor
(1+z)(ddL/dz)^2 varies by 2.1x (median over events; range 1.1-3.3x)
between the 5th and 95th percentiles of a single O4 event posterior
(4.6x max/min over 7000 samples), up-weighting the high-z / low-m1_source
end of every affected event; fixing it tilts each O4 event to lower z and
higher m1_source, and correspondingly shifts the inferred mass scales and
redshift evolution.  Validation of the regenerated PE file against the
quarantined one: `m1`, `q`, `dl`, `evt`, `pe_file`, `pe_group` identical;
`pdraw` (stored as log) identical for the 69 O1-O3 events, changed for
all 190 O4 events; `pdraw_new - pdraw_old + log((1+z)(ddL/dz)^2)` is
constant per event up to a within-event std of 1.4e-3 (median; 2.4e-3
max) that is exactly the Planck18-at-file-z vs Planck15_LAL-at-z(dL)
evaluation difference (residual after subtracting it: 5e-5).  All conclusions
drawn from `runs/bad_z_prior/realGWTC5_*` should be treated as void; the
runs to redo first are `realGWTC5_lvk_control` and
`realGWTC5_noevo_259ev_qpair` (see their inis).

## Quarantine (branch)

The 21 affected run directories were moved verbatim to `runs/bad_z_prior/`
(with a README).  `runs/realGWTC5_noevo_259ev/` was regenerated on
2026-09-03 with `scripts/extract_realGWTC5_259.py` (same seed; `m1`, `q`,
`dl`, event labels identical to the quarantined file, only `pdraw` of the
190 O4 events changed).  The selection file `sel_GWTC5_fixed.h5` was
unaffected and is copied from the quarantined provenance directory.

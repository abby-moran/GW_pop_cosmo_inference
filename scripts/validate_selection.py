"""Regression check for weighting.extract_selection_samples on the GWTC-5
sensitivity mixture file (Zenodo 19500052).

Validates the fixed selection-normalization recipe:
  1. Detection mask (snr > 10 | min-over-searches FAR < 1/yr) before any m2
     cut recovers the documented per-epoch counts.
  2. With the true-m2 > 2.5 Msun cut, extract_selection_samples returns the
     expected number of rows, ndraw = total_generated, and finite pdraw.
  3. mu_sel recomputed in float64 at the posterior-median shape of
     runs/realGWTC5_noevo_fullsel lands at ~14 Gpc^3 yr (implied R ~ 17
     /Gpc^3/yr for 243 events), i.e. the ~460x rate bias is gone.

Usage (run from ``scripts/``)::

    uv run python validate_selection.py [--mixture-file PATH] [--reference-nc PATH]

The defaults point at the cluster copy of the mixture file and at
runs/realGWTC5_noevo_fullsel/realGWTC5_noevo_fullsel.nc; override them to
run elsewhere.  Exits nonzero on any failed assertion.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import weighting  # noqa: E402

MIXTURE_FILE = ('/mnt/home/amoran/GW_pop_cosmo_inference/runs/'
                'mixture-semi_o1_o2-real_o3_o4a_o4b-cartesian_spins_'
                '20260410130052UTC-clipped.hdf')
REFERENCE_NC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'runs', 'realGWTC5_noevo_fullsel',
                            'realGWTC5_noevo_fullsel.nc')
NOBS = 243

# Documented expectations (see the Zenodo 19500052 release notes and
# notes in the fix-selection-normalization investigation).
EXPECTED_NDRAW = 1_568_035_640
EXPECTED_DETECTED = 1_578_148            # (snr>10 | far<1/yr), no m2 cut
EXPECTED_DETECTED_BY_EPOCH = {
    'semi': 369_995,
    'O3a': 115_779,
    'O3b': 105_545,
    'O4a': 482_163,
    'O4b': 504_666,
}
EXPECTED_ROWS_M2CUT = 1_433_314          # after true mass2_source > 2.5
EXPECTED_MU_SEL = 14.0                   # Gpc^3 yr at posterior-median shape
MU_SEL_TOL = 1.0

# GPS boundaries of the observing epochs; rows outside all ranges are the
# semianalytic O1/O2 injections (which have no time_geocenter campaign).
EPOCH_RANGES = {
    'O3a': (1238166018, 1253977218),
    'O3b': (1256655618, 1269363618),
    'O4a': (1368975618, 1389456018),
    'O4b': (1396969218, 1422118818),
}


def check_detection_mask(mixture_file=MIXTURE_FILE):
    """Recompute the detection mask directly and check per-epoch counts."""
    with h5py.File(mixture_file, 'r') as f:
        searches = [s.decode() if isinstance(s, bytes) else s
                    for s in f.attrs['searches']]
        ev = f['events']
        snr = ev['semianalytic_observed_phase_maximized_snr_net'][:]
        tgeo = ev['time_geocenter'][:]
        far_cols = np.vstack([ev[f'{s}_far'][:] for s in searches])
        ndraw = f.attrs['total_generated']

    assert len(searches) == 13, f"expected 13 searches, got {len(searches)}"
    assert not np.isnan(far_cols).any(), "NaNs in FAR columns"
    far = far_cols.min(axis=0)
    detected = (snr > 10) | (far < 1.0)

    n_det = int(detected.sum())
    print(f"detected (snr>10 | far<1/yr), no m2 cut: {n_det}")
    assert n_det == EXPECTED_DETECTED, (
        f"detected count {n_det} != expected {EXPECTED_DETECTED}")

    epoch = np.full(len(snr), 'semi', dtype=object)
    for name, (lo, hi) in EPOCH_RANGES.items():
        epoch[(tgeo >= lo) & (tgeo < hi)] = name
    for name, expected in EXPECTED_DETECTED_BY_EPOCH.items():
        n = int(((epoch == name) & detected).sum())
        print(f"  {name}: {n}")
        assert n == expected, f"epoch {name}: {n} != expected {expected}"

    assert int(ndraw) == EXPECTED_NDRAW, (
        f"total_generated {ndraw} != expected {EXPECTED_NDRAW}")


def check_extract_selection_samples(mixture_file=MIXTURE_FILE):
    """Run the fixed extractor and check counts, ndraw, and sanity of pdraw."""
    (m1, q, z, a1, a2, ct1, ct2, pdraw, ndraw) = weighting.extract_selection_samples(
        mixture_file, nsamp=None, desired_pop_wt=None, mass_sel=2.5)

    print(f"extract_selection_samples: {len(m1)} rows, ndraw = {ndraw[0]:.0f}")
    assert len(m1) == EXPECTED_ROWS_M2CUT, (
        f"rows {len(m1)} != expected {EXPECTED_ROWS_M2CUT}")
    assert int(ndraw[0]) == EXPECTED_NDRAW, (
        f"returned ndraw {ndraw[0]} != expected {EXPECTED_NDRAW}")
    assert np.all(np.isfinite(pdraw)) and np.all(pdraw > 0), "bad pdraw values"
    assert np.all(np.isfinite(m1)) and np.all((q > 0) & (q <= 1)), "bad m1/q"

    # Assemble the run-file schema (same as real_dat_run.py's selection block).
    from astropy.cosmology import Planck18
    import astropy.units as u
    df = pd.DataFrame({'m1': m1, 'q': q, 'z': z, 'a1': a1, 'a2': a2,
                       'cos_tilt_1': ct1, 'cos_tilt_2': ct2,
                       'pdraw_m1sqz': pdraw, 'ndraw': ndraw})
    df['dm1sz_dm1ddl'] = weighting.dm1sz_dm1ddl(df['z'])
    df['pdraw_sel'] = df['pdraw_m1sqz'] * df['dm1sz_dm1ddl']
    df['m1d'] = df['m1'] * (1 + df['z'])
    df['dl'] = Planck18.luminosity_distance(df['z'].to_numpy()).to(u.Gpc).value
    return df


def check_mu_sel(sel_df, reference_nc=REFERENCE_NC):
    """Recompute mu_sel (float64) at the reference run's posterior median."""
    import arviz as az
    import diagnose_run as dr

    idata = az.from_netcdf(reference_nc)
    post_med = dr._posterior_median_dict(idata.posterior)
    sample = dr._canonical_pop_sample({}, post_med)

    # EXPECTED_MU_SEL was derived from realGWTC5_noevo_fullsel, which was
    # sampled under the total-mass ("mt") pairing (the lvk-powerlaw-z branch's
    # run_inf.py fallback; main's fallback is "q").  Use the pairing recorded
    # in the .nc attrs so the weights match the chain, and fall back to "mt"
    # when the attrs carry no pairing information at all.
    try:
        pairing = dr._pairing_of(idata)
    except Exception as exc:  # unreadable/unknown attrs: assume the mt run
        print(f"warning: could not read pairing from {reference_nc} ({exc}); "
              "assuming 'mt'")
        pairing = "mt"
    print(f"selection weights computed with pairing = {pairing!r}")

    lw = dr._sel_log_wts(sel_df, sample, use_low_bump=True, pairing=pairing)
    assert np.all(np.isfinite(lw) | (lw < np.inf)), "NaN selection log-weights"
    lw = lw[np.isfinite(lw)]
    m = lw.max()
    w = np.exp(lw - m)
    mu = np.exp(m) * w.sum() / EXPECTED_NDRAW
    neff = w.sum()**2 / (w**2).sum()
    print(f"mu_sel = {mu:.4f} Gpc^3 yr  (log_mu_sel = {np.log(mu):.4f}, "
          f"neff = {neff:.0f}, implied R = {NOBS/mu:.2f} /Gpc^3/yr)")
    assert abs(mu - EXPECTED_MU_SEL) < MU_SEL_TOL, (
        f"mu_sel = {mu:.3f} not within {MU_SEL_TOL} of {EXPECTED_MU_SEL}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Regression checks for weighting.extract_selection_samples "
                    "on the GWTC-5 sensitivity mixture file.")
    parser.add_argument(
        "--mixture-file", default=MIXTURE_FILE,
        help="GWTC-5 sensitivity mixture .hdf (Zenodo 19500052) "
             "(default: %(default)s)")
    parser.add_argument(
        "--reference-nc", default=REFERENCE_NC,
        help="ArviZ .nc of the reference run whose posterior median sets the "
             "mu_sel check (default: %(default)s)")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args()
    check_detection_mask(args.mixture_file)
    sel_df = check_extract_selection_samples(args.mixture_file)
    check_mu_sel(sel_df, args.reference_nc)
    print("\nAll selection-extraction regression checks passed.")

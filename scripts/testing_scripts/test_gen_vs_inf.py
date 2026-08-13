"""
Consistency test: generation pipeline vs inference model, point by point.

reweight_res.py records, for every detected injection, pdraw_sel
proportional to the population density at truth in (m1_det, q, dL), so if
the inference model is consistent with the generation model,
    resid := log_pop_inference(m1d, q, dl | truth) - log(pdraw_sel)
must be CONSTANT across all samples.  The constant is log_C = log(Z) +
log_w_max - log(n_total) for files made after the pdraw normalization fix
(pdraw_sel is the properly normalized drawing density), or log_w_max for
older files.  Any trend of `resid` with
m1, q or z localizes an inconsistency (population shape, cosmology Jacobians,
frame conversions, pdraw bookkeeping).

Usage: uv run python test_gen_vs_inf.py [--config run_configs/mock_O5_noevo_val.ini]
"""
import argparse
import configparser
import os
import sys

sys.path.append("../src/")

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

import intensity_models_fast as fast


def truth_sample(pop_config_path):
    """Build the parameter dict exactly as pop_cosmo_model sees it at truth."""
    tv = {}
    with open(pop_config_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                k, v = line.split("=", 1)
                try:
                    tv[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    # inference samples (dmbhmax, dkappa, log_fpl, log_flow) and maps them back;
    # the round trip is the identity, so just pass the raw values through.
    return tv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="run_configs/mock_O5_noevo_val.ini")
    p.add_argument("--nsub", type=int, default=200_000)
    args = p.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    run = cfg["run"]
    run_dir = os.path.join("../runs", run["run_dir"])

    sel = pd.read_hdf(os.path.join(run_dir, run["output_sel_file"]), key="true_parameters")
    rng = np.random.default_rng(7)
    idx = rng.choice(len(sel), size=min(args.nsub, len(sel)), replace=False)
    sel = sel.iloc[idx]
    print(f"{len(sel)} selection samples (of the full set)")

    tv = truth_sample(os.path.join("pop_configs", run["pop_config_file"]))
    print("truth:", {k: round(v, 4) for k, v in tv.items()})

    cosmo = fast.FlatwCDMCosmology(tv["h"], tv["Om"], tv["w"], zmax=tv["zmax"])
    ld = fast.build_population_model(tv, use_low_bump=True, smooth_tail_edge=True)

    m1d = jnp.asarray(sel["m1d"].to_numpy())
    q = jnp.asarray(sel["q"].to_numpy())
    dl = jnp.asarray(sel["dl"].to_numpy())
    z_true = sel["z"].to_numpy()
    m1_true = sel["m1"].to_numpy()

    # --- inference-side density, exactly as pop_cosmo_model computes it ------
    zs = cosmo.z_of_dL(dl)
    m1s = m1d / (1 + zs)
    log_pop_inf = (ld(m1s, q, zs)
                   + jnp.log(cosmo.dVCdz(zs))
                   - 2 * jnp.log1p(zs)
                   - jnp.log(cosmo.ddL_dz(zs)))

    resid = np.asarray(log_pop_inf) - np.log(sel["pdraw_sel"].to_numpy())

    good = np.isfinite(resid)
    print(f"\nresid = log_pop_inference - log(pdraw_sel)   "
          f"[expect constant: log_C for normalized files, log_w_max for legacy]")
    print(f"  non-finite resid: {np.sum(~good)} of {len(resid)}")
    r = resid[good]
    print(f"  mean {r.mean():+.4f}  std {r.std():.4f}  "
          f"min {r.min():+.4f}  max {r.max():+.4f}")

    # z inversion accuracy on its own
    dz = np.asarray(zs)[good] - z_true[good]
    print(f"  z_of_dL - z_true:  max|dz| {np.abs(dz).max():.2e}  "
          f"rms {np.sqrt((dz**2).mean()):.2e}")

    # localize any structure
    for name, x in (("m1_true", m1_true[good]), ("q", np.asarray(q)[good]),
                    ("z_true", z_true[good])):
        qs_ = np.quantile(x, np.linspace(0, 1, 9))
        print(f"\n  resid binned by {name}:")
        for lo, hi in zip(qs_[:-1], qs_[1:]):
            m = (x >= lo) & (x <= hi)
            print(f"    [{lo:9.3f}, {hi:9.3f}]  mean {r[m].mean():+.5f}  "
                  f"std {r[m].std():.5f}  n {m.sum()}")


if __name__ == "__main__":
    main()

"""
Fine-grained decomposition of the ~27ms "likelihood" bucket from
profile_model.py. That number is an aggregate of several genuinely
different operations; this script isolates each one so you know whether
the cost is cosmology, the PISN grid interpolation, the low-mass/power-law
shape terms, or the mixture bookkeeping -- rather than guessing.

Mirrors the exact computation in pop_cosmo_model's likelihood block:

    zs = cosmo.z_of_dL(dls)
    m1s = m1s_det / (1 + zs)
    log_wts = log_dN(m1s, qs, zs) - log_pdraw
              - 2*log1p(zs) - log(cosmo.ddL_dz(zs)) + log(cosmo.dVCdz(zs))

Broken into:
  (1) COSMOLOGY: z_of_dL, ddL_dz, dVCdz -- three independent jnp.interp
      calls into FlatwCDMCosmology's 1024-point interpolation table.
      Also includes the FUSED version (cosmo.full_cosmo_block), which
      shares index computation between ddL_dz and dVCdz since they
      interpolate against the same z-grid.
  (2) PISN GRID INTERP ONLY: just interp_2d_dndmpisn, called on m1 and m2
      (this is the map_coordinates bilinear lookup into the big grid).
  (3) SHAPE TERMS ONLY: log_p_low (Gaussian) + log_p_pl (power-law tail),
      pure elementwise math, no interpolation -- this should be cheap and
      serves as a sanity-check baseline.
  (4) FULL LogDNDM.__call__: for comparison against the sum of the above.
  (5) FULL LogDNDMDQDV.__call__ (i.e. full log_dN as called in the model):
      for comparison against (4)*2 plus mixture/dV bookkeeping.
  (6) FULL likelihood expression (cosmology + log_dN + the surrounding
      Jacobian terms), matching pop_cosmo_model exactly, as the top-level
      check that the parts sum to the whole from the original profiler.
      Includes both the ORIGINAL (unfused) and FUSED cosmology paths so
      the two can be compared directly.

Usage: same as profile_model.py
    python profile_model_detailed.py --config run_configs/mock_O5_noevo.ini
    python profile_model_detailed.py --synthetic
"""
import argparse
import time
import sys
import os
import json

sys.path.append('../src/')

import jax
import jax.numpy as jnp
import numpy as np


def block(x):
    return jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, x)


def timeit(fn, *args, n_repeat=7, name="", **kwargs):
    t0 = time.perf_counter()
    block(fn(*args, **kwargs))
    compile_time = time.perf_counter() - t0

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        block(fn(*args, **kwargs))
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    print(f"  [{name:45s}] compile+first: {compile_time:7.4f}s | "
          f"min: {times.min()*1000:7.3f}ms | median: {np.median(times)*1000:7.3f}ms")
    return dict(name=name, compile_and_first_run=compile_time,
                min=float(times.min()), median=float(np.median(times)))


def make_synthetic_data(nobs=9000, nsamp=4000, seed=0):
    rng = np.random.default_rng(seed)
    m1s_det = rng.uniform(10, 100, size=(nobs, nsamp))
    qs = rng.uniform(0.1, 1.0, size=(nobs, nsamp))
    dls = rng.uniform(100, 5000, size=(nobs, nsamp))
    log_pdraw = np.log(rng.uniform(1e-6, 1e-2, size=(nobs, nsamp)))
    return m1s_det, qs, dls, log_pdraw


def load_real_event_data(cfg_path):
    import configparser
    import pandas as pd
    import h5py

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    run = cfg["run"]
    base_runs_dir = "../runs"
    run_dir = os.path.join(base_runs_dir, run["run_dir"])
    pe_file = os.path.join(run_dir, run["output_file_PE"])
    prior_dir = "../runs/priors"
    prior_path = os.path.join(prior_dir, run["prior"])

    evt_start = run.getint("evt_start")
    evt_end = run.get("evt_end")
    evt_end = None if (evt_end is None or evt_end.lower() == "none") else int(evt_end)

    try:
        with h5py.File(pe_file, "r") as f:
            m1s = f["m1"][evt_start:evt_end]
            qs = f["q"][evt_start:evt_end]
            dls = f["dl"][evt_start:evt_end]
            pdraws = f["pdraw"][evt_start:evt_end]
    except (KeyError, OSError):
        pe_samples_mock = pd.read_hdf(pe_file, key="samples").iloc[evt_start:evt_end]
        m1s = np.asarray(pe_samples_mock["m1"].to_list())
        qs = np.asarray(pe_samples_mock["q"].to_list())
        dls = np.asarray(pe_samples_mock["dl"].to_list())
        pdraws = np.asarray(pe_samples_mock["pdraw"].to_list())

    pdraws = jnp.nan_to_num(pdraws, neginf=-1e30, posinf=1e30)
    return np.asarray(m1s), np.asarray(qs), np.asarray(dls), np.asarray(np.log(pdraws)), prior_path


def get_example_hyperparams():
    return dict(
        a=2.0, b=3.0, c=3.0,
        mpisn=45.0, mpisndot=0.0, dmbhmax=10.0,
        sigma=0.1, log_fpl=np.log(0.01),
        beta=0.0, lam=2.0, dkappa=1.0, zp=2.0,
        mp_low=10.0, msigma_low=2.0, log_flow=np.log(0.05),
        h=0.7, Om=0.3, w=-1.0,
        mbh_min=5.0, delta_m=2.5, zmax=20,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--nobs", type=int, default=9000)
    p.add_argument("--nsamp", type=int, default=4000)
    p.add_argument("--n_repeat", type=int, default=7)
    args = p.parse_args()

    import intensity_models as im

    if args.config:
        print(f"Loading real event data from config: {args.config}")
        m1s_det, qs, dls, log_pdraw, prior_path = load_real_event_data(args.config)
        from utils import get_priors_from_file, sample_parameters_from_dict
        import numpyro.handlers as handlers
        prior = get_priors_from_file(prior_path)
        with handlers.seed(rng_seed=0):
            sample = sample_parameters_from_dict(prior)
    else:
        print(f"Using synthetic data: nobs={args.nobs}, nsamp={args.nsamp}")
        m1s_det, qs, dls, log_pdraw = make_synthetic_data(args.nobs, args.nsamp)
        sample = get_example_hyperparams()

    print(f"Event array shape: {m1s_det.shape} ({m1s_det.size:,} points)\n")

    det = im.get_deterministic_parameters(sample)
    full_sample = dict(sample)
    full_sample.update(det)

    cosmo = im.FlatwCDMCosmology(full_sample['h'], full_sample['Om'], full_sample['w'], zmax=full_sample['zmax'])
    log_dN = im.build_population_model(full_sample)   # LogDNDMDQDV instance
    log_dndm = log_dN.log_dndm                         # LogDNDM instance (per-mass, pre-pairing)

    dls_j = jnp.asarray(dls)
    qs_j = jnp.asarray(qs)
    m1s_det_j = jnp.asarray(m1s_det)

    results = {}

    # ---------------------------------------------------------------
    print("=== (1) COSMOLOGY ONLY ===")
    print("Three independent jnp.interp lookups into a 1024-pt cosmology table,")
    print("plus the FUSED version (cosmo.full_cosmo_block) for comparison.\n")

    zs = cosmo.z_of_dL(dls_j)  # need this for everything downstream too

    results['cosmo_z_of_dL'] = timeit(
        jax.jit(cosmo.z_of_dL), dls_j, n_repeat=args.n_repeat, name="z_of_dL(dls)")
    results['cosmo_ddL_dz'] = timeit(
        jax.jit(cosmo.ddL_dz), zs, n_repeat=args.n_repeat, name="ddL_dz(zs)  [unfused]")
    results['cosmo_dVCdz'] = timeit(
        jax.jit(cosmo.dVCdz), zs, n_repeat=args.n_repeat, name="dVCdz(zs)  [unfused]")

    def unfused_cosmo_block(dls):
        zs = cosmo.z_of_dL(dls)
        return zs, cosmo.ddL_dz(zs), cosmo.dVCdz(zs)

    results['cosmo_all_three_unfused'] = timeit(
        jax.jit(unfused_cosmo_block), dls_j, n_repeat=args.n_repeat,
        name="ALL THREE cosmology calls, UNFUSED")

    # FUSED: calls the new cosmo.full_cosmo_block method directly, which
    # now does a single batched map_coordinates call for ddL_dz/dVCdz
    # (one kernel launch) instead of two independent jnp.interp calls.
    results['cosmo_all_three_fused'] = timeit(
        jax.jit(cosmo.full_cosmo_block), dls_j, n_repeat=args.n_repeat,
        name="ALL THREE cosmology calls, BATCHED (new)")

    m1s = m1s_det_j / (1 + zs)

    # ---------------------------------------------------------------
    print("\n=== (2) PISN GRID INTERPOLATION ONLY (interp_2d_dndmpisn) ===")
    print("The map_coordinates bilinear lookup into the big (n_z, n_m, n_m) grid.\n")

    results['pisn_interp_m1'] = timeit(
        jax.jit(log_dndm.interp_2d_dndmpisn), m1s, zs,
        n_repeat=args.n_repeat, name="interp_2d_dndmpisn(m1, z)  [one call]")

    m2s = qs_j * m1s
    results['pisn_interp_m1_and_m2'] = timeit(
        jax.jit(lambda m1, m2, z: (log_dndm.interp_2d_dndmpisn(m1, z),
                                    log_dndm.interp_2d_dndmpisn(m2, z))),
        m1s, m2s, zs, n_repeat=args.n_repeat,
        name="interp_2d_dndmpisn called for BOTH m1 and m2")

    # ---------------------------------------------------------------
    print("\n=== (3) SHAPE TERMS ONLY (no interpolation, pure elementwise) ===")
    print("Sanity-check baseline -- should be cheap.\n")

    def shape_terms_only(m, mu_low, sigma_low):
        return im.log_normalized_gaussian(m, mu_low, sigma_low)

    results['shape_gaussian'] = timeit(
        jax.jit(shape_terms_only), m1s, full_sample['mp_low'], full_sample['msigma_low'],
        n_repeat=args.n_repeat, name="log_normalized_gaussian (low-mass peak)")

    # ---------------------------------------------------------------
    print("\n=== (4) FULL LogDNDM.__call__ (single mass, e.g. m1 only) ===")
    print("Should be roughly (2, one-sided) + (3) + overhead.\n")

    results['full_logdndm_m1'] = timeit(
        jax.jit(log_dndm.__call__), m1s, zs,
        n_repeat=args.n_repeat, name="log_dndm(m1, z)  [full LogDNDM call]")

    # ---------------------------------------------------------------
    print("\n=== (5) FULL log_dN (LogDNDMDQDV.__call__): m1 + m2 + mixture/dV ===")
    print("This is exactly what's called inside pop_cosmo_model as log_dN(m1s, qs, zs).\n")

    results['full_log_dN'] = timeit(
        jax.jit(log_dN.__call__), m1s, qs_j, zs,
        n_repeat=args.n_repeat, name="log_dN(m1, q, z)")

    # ---------------------------------------------------------------
    print("\n=== (6) FULL LIKELIHOOD EXPRESSION (matches pop_cosmo_model exactly) ===")
    print("Shown for BOTH the original unfused cosmology path and the new")
    print("fused cosmo.full_cosmo_block path, so the real-world impact of")
    print("fusing is visible at the top level, not just in isolation.\n")

    def full_likelihood_block_unfused(m1s_det, qs, dls, log_pdraw):
        zs = cosmo.z_of_dL(dls)
        m1s = m1s_det / (1 + zs)
        log_wts = (log_dN(m1s, qs, zs) - log_pdraw
                   - 2 * jnp.log1p(zs) - jnp.log(cosmo.ddL_dz(zs))
                   + jnp.log(cosmo.dVCdz(zs)))
        return log_wts

    def full_likelihood_block_fused(m1s_det, qs, dls, log_pdraw):
        zs, ddl, dvc = cosmo.full_cosmo_block(dls)
        m1s = m1s_det / (1 + zs)
        log_wts = (log_dN(m1s, qs, zs) - log_pdraw
                   - 2 * jnp.log1p(zs) - jnp.log(ddl)
                   + jnp.log(dvc))
        return log_wts

    results['full_likelihood_expression_unfused'] = timeit(
        jax.jit(full_likelihood_block_unfused), m1s_det_j, qs_j, dls_j, jnp.asarray(log_pdraw),
        n_repeat=args.n_repeat, name="FULL likelihood expr, UNFUSED cosmology")

    results['full_likelihood_expression_fused'] = timeit(
        jax.jit(full_likelihood_block_fused), m1s_det_j, qs_j, dls_j, jnp.asarray(log_pdraw),
        n_repeat=args.n_repeat, name="FULL likelihood expr, FUSED cosmology (new)")

    # Keep this key for backwards compatibility with any downstream tooling
    # that reads profile_results_detailed.json and expects it.
    results['full_likelihood_expression'] = results['full_likelihood_expression_unfused']

    # ---------------------------------------------------------------
    print("\n=== SUMMARY: where does the ~27ms actually go, and did fusing help? ===\n")
    full = results['full_likelihood_expression_unfused']['min']
    full_fused = results['full_likelihood_expression_fused']['min']
    cosmo_cost = results['cosmo_all_three_unfused']['min']
    cosmo_cost_fused = results['cosmo_all_three_fused']['min']
    pisn_cost = results['pisn_interp_m1_and_m2']['min']
    full_dN = results['full_log_dN']['min']

    print(f"  Full likelihood expression (unfused cosmo): {full*1000:8.3f} ms  (100%)")
    print(f"  Full likelihood expression (fused cosmo):   {full_fused*1000:8.3f} ms  "
          f"({100*full_fused/full:5.1f}% of unfused)")
    print(f"    -> fusing cosmology saved: {(full-full_fused)*1000:8.3f} ms "
          f"({100*(full-full_fused)/full:5.1f}% of total)")
    print()
    print(f"    of which cosmology, unfused (3 interp calls): {cosmo_cost*1000:8.3f} ms  ({100*cosmo_cost/full:5.1f}%)")
    print(f"    of which cosmology, batched (1 map_coordinates call): {cosmo_cost_fused*1000:8.3f} ms  ({100*cosmo_cost_fused/full:5.1f}% of unfused total)")
    print(f"    of which full log_dN (mass function):   {full_dN*1000:8.3f} ms  ({100*full_dN/full:5.1f}%)")
    print(f"      of which PISN grid interp (m1+m2):    {pisn_cost*1000:8.3f} ms  ({100*pisn_cost/full:5.1f}% of total)")
    print()
    print("  If cosmology % is large: check whether the FUSED number above")
    print("  actually beats the unfused one. If not, the ~10ms/call floor is")
    print("  likely fixed per-op dispatch overhead rather than compute that")
    print("  fusing the interpolation math can remove.")
    print()
    print("  If PISN grid interp % is large: this is fundamental to your")
    print("  interpolation-based approach -- the fix is grid resolution (n_m)")
    print("  from the previous profiling round, not this call site.")

    with open("profile_results_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved raw numbers to profile_results_detailed.json")


if __name__ == "__main__":
    main()
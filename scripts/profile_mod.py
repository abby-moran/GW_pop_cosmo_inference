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
  (2) PISN GRID INTERP ONLY: just interp_2d_dndmpisn, called on m1 and m2
      (this is the map_coordinates bilinear lookup into the big grid).
  (3) JOIN-POINT TERMS: join_point_terms(z), which itself calls
      interp_2d_dndmpisn AGAIN (this is the suspected duplicate-work path
      -- computed independently for the m1 call and the m2 call even
      though z is identical between them).
  (4) SHAPE TERMS ONLY: log_p_low (Gaussian) + log_p_pl (power-law tail),
      pure elementwise math, no interpolation -- this should be cheap and
      serves as a sanity-check baseline.
  (5) FULL LogDNDM.__call__: for comparison against the sum of the above.
  (6) FULL LogDNDMDQDV.__call__ (i.e. full log_dN as called in the model):
      for comparison against (5)*2 plus mixture/dV bookkeeping.
  (7) FULL likelihood expression (cosmology + log_dN + the surrounding
      Jacobian terms), matching pop_cosmo_model exactly, as the top-level
      check that the parts sum to the whole from the original profiler.

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
    log_dndv = log_dN.log_dndv                          # LogDNDV instance (merger rate density)

    dls_j = jnp.asarray(dls)
    qs_j = jnp.asarray(qs)
    m1s_det_j = jnp.asarray(m1s_det)

    results = {}

    # ---------------------------------------------------------------
    print("=== (1) COSMOLOGY ONLY ===")
    print("Three independent jnp.interp lookups into a 1024-pt cosmology table.\n")

    zs = cosmo.z_of_dL(dls_j)  # need this for everything downstream too

    results['cosmo_z_of_dL'] = timeit(
        jax.jit(cosmo.z_of_dL), dls_j, n_repeat=args.n_repeat, name="z_of_dL(dls)")
    results['cosmo_ddL_dz'] = timeit(
        jax.jit(cosmo.ddL_dz), zs, n_repeat=args.n_repeat, name="ddL_dz(zs)")
    results['cosmo_dVCdz'] = timeit(
        jax.jit(cosmo.dVCdz), zs, n_repeat=args.n_repeat, name="dVCdz(zs)")

    def full_cosmo_block(dls):
        zs = cosmo.z_of_dL(dls)
        return zs, cosmo.ddL_dz(zs), cosmo.dVCdz(zs)

    results['cosmo_all_three'] = timeit(
        jax.jit(full_cosmo_block), dls_j, n_repeat=args.n_repeat,
        name="ALL THREE cosmology calls combined")

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
    print("\n=== (3) JOIN-POINT TERMS (join_point_terms) ===")
    print("Calls interp_2d_dndmpisn AGAIN internally -- this is evaluated")
    print("independently inside the m1 call AND the m2 call in LogDNDMDQDV,")
    print("even though z is identical between them. Isolating the cost of")
    print("ONE call here tells you the size of that duplicated work.\n")

    results['join_point_terms_once'] = timeit(
        jax.jit(log_dndm.join_point_terms), zs,
        n_repeat=args.n_repeat, name="join_point_terms(z)  [one call]")

    results['join_point_terms_twice'] = timeit(
        jax.jit(lambda z: (log_dndm.join_point_terms(z), log_dndm.join_point_terms(z))),
        zs, n_repeat=args.n_repeat,
        name="join_point_terms(z) called TWICE (current behavior)")

    # ---------------------------------------------------------------
    print("\n=== (4) SHAPE TERMS ONLY (no interpolation, pure elementwise) ===")
    print("Sanity-check baseline -- should be cheap.\n")

    def shape_terms_only(m, mu_low, sigma_low):
        return im.log_normalized_gaussian(m, mu_low, sigma_low)

    results['shape_gaussian'] = timeit(
        jax.jit(shape_terms_only), m1s, full_sample['mp_low'], full_sample['msigma_low'],
        n_repeat=args.n_repeat, name="log_normalized_gaussian (low-mass peak)")

    # ---------------------------------------------------------------
    print("\n=== (5) FULL LogDNDM.__call__ (single mass, e.g. m1 only) ===")
    print("Should be roughly (2, one-sided) + (3, one-sided) + (4) + overhead.\n")

    results['full_logdndm_m1'] = timeit(
        jax.jit(log_dndm.__call__), m1s, zs,
        n_repeat=args.n_repeat, name="log_dndm(m1, z)  [full LogDNDM call]")

    # ---------------------------------------------------------------
    print("\n=== (6) FULL log_dN (LogDNDMDQDV.__call__): m1 + m2 + mixture/dV ===")
    print("This is exactly what's called inside pop_cosmo_model as log_dN(m1s, qs, zs).\n")

    results['full_log_dN'] = timeit(
        jax.jit(log_dN.__call__), m1s, qs_j, zs,
        n_repeat=args.n_repeat, name="log_dN(m1, q, z)  [full LogDNDMDQDV call]")

    # ---------------------------------------------------------------
    print("\n=== (7) FULL LIKELIHOOD EXPRESSION (matches pop_cosmo_model exactly) ===\n")

    def full_likelihood_block(m1s_det, qs, dls, log_pdraw):
        zs = cosmo.z_of_dL(dls)
        m1s = m1s_det / (1 + zs)
        log_wts = (log_dN(m1s, qs, zs) - log_pdraw
                   - 2 * jnp.log1p(zs) - jnp.log(cosmo.ddL_dz(zs))
                   + jnp.log(cosmo.dVCdz(zs)))
        return log_wts

    results['full_likelihood_expression'] = timeit(
        jax.jit(full_likelihood_block), m1s_det_j, qs_j, dls_j, jnp.asarray(log_pdraw),
        n_repeat=args.n_repeat, name="FULL likelihood expr (cosmo + log_dN + Jacobian)")

    # ---------------------------------------------------------------
    print("\n=== SUMMARY: where does the ~27ms actually go? ===\n")
    full = results['full_likelihood_expression']['min']
    cosmo_cost = results['cosmo_all_three']['min']
    pisn_cost = results['pisn_interp_m1_and_m2']['min']
    join_cost = results['join_point_terms_twice']['min']
    join_dup_savings = results['join_point_terms_twice']['min'] - results['join_point_terms_once']['min']
    full_dN = results['full_log_dN']['min']

    print(f"  Full likelihood expression:              {full*1000:8.3f} ms  (100%)")
    print(f"    of which cosmology (3 interp calls):    {cosmo_cost*1000:8.3f} ms  ({100*cosmo_cost/full:5.1f}%)")
    print(f"    of which full log_dN (mass function):   {full_dN*1000:8.3f} ms  ({100*full_dN/full:5.1f}%)")
    print(f"      of which PISN grid interp (m1+m2):    {pisn_cost*1000:8.3f} ms  ({100*pisn_cost/full:5.1f}% of total)")
    print(f"      of which join_point_terms (x2 calls): {join_cost*1000:8.3f} ms  ({100*join_cost/full:5.1f}% of total)")
    print(f"        -> redundant 2nd join_point_terms call costs "
          f"~{join_dup_savings*1000:.3f} ms ({100*join_dup_savings/full:5.1f}% of total)")
    print(f"        -> that's the specific savings available from passing")
    print(f"           join_terms= through to avoid recomputation for m2")
    print()
    print("  If cosmology % is large: the fix is caching/precomputing zs,")
    print("  ddL_dz, dVCdz once per likelihood call rather than per-something-else,")
    print("  or increasing the cosmology interpolation table's coarseness")
    print("  tradeoff (ninterp=1024 currently) if it's the dominant cost.")
    print()
    print("  If PISN grid interp % is large: this is fundamental to your")
    print("  interpolation-based approach -- the fix is grid resolution (n_m)")
    print("  from the previous profiling round, not this call site.")
    print()
    print("  If join_point_terms duplication is a meaningful %: wiring")
    print("  join_terms= through LogDNDMDQDV.__call__ (as discussed) directly")
    print("  removes exactly that cost.")

    with open("profile_results_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved raw numbers to profile_results_detailed.json")


if __name__ == "__main__":
    main()
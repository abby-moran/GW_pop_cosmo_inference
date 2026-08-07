"""
Test F of the recovery-bias investigation: population-weighted PIT / coverage
test of the mock-PE machinery, no MCMC required.

Logic: conditioned on its observed data (which includes passing detection and
the mass cut, both functions of the data alone), each event's true parameters
are one draw from

    p(theta | d) propto L(d | theta) * pop_truth(theta),

i.e. from the event's PE samples (which claim density L(d|theta) flat in
(m1_det, q, dL)) reweighted by the population at truth.  Therefore, for any
1-D coordinate x, the population-weighted rank of the true value x* among the
PE samples,

    F = sum_j w_j 1[x_j <= x*] / sum_j w_j,   w_j = pop_truth(theta_j),

must be Uniform(0,1) across events.  A tilt in F for some coordinate means
the PE cloud is systematically offset/skewed in that coordinate relative to
its claimed density -- exactly the kind of defect that biases the
hierarchical fit while leaving n_eff diagnostics healthy.
"""
import argparse
import configparser
import os
import sys

sys.path.append("../src/")

import h5py
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

import intensity_models_fast as fast
from test_gen_vs_inf import truth_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="run_configs/mock_O5_noevo_val.ini")
    p.add_argument("--nobs", type=int, default=10000)
    args = p.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    run = cfg["run"]
    run_dir = os.path.join("../runs", run["run_dir"])

    with h5py.File(os.path.join(run_dir, run["output_file_pe"]), "r") as f:
        m1d = f["m1"][: args.nobs]
        q = f["q"][: args.nobs]
        dl = f["dl"][: args.nobs]
    nobs = m1d.shape[0]

    sel = pd.read_hdf(os.path.join(run_dir, run["output_sel_file"]),
                      key="true_parameters").iloc[:nobs]
    true_m1d = sel["m1d"].to_numpy()
    true_q = sel["q"].to_numpy()
    true_dl = sel["dl"].to_numpy()

    # alignment sanity: the PE cloud should sit near its event's truth
    med = np.median(m1d, axis=1)
    r = np.corrcoef(np.log(med), np.log(true_m1d))[0, 1]
    print(f"alignment check: corr(log median PE m1d, log true m1d) = {r:.4f}")

    tv = truth_sample(os.path.join("pop_configs", run["pop_config_file"]))
    cosmo = fast.FlatwCDMCosmology(tv["h"], tv["Om"], tv["w"], zmax=tv["zmax"])
    ld = fast.build_population_model(tv, use_low_bump=True, smooth_tail_edge=True)

    @jax.jit
    def log_pop(m1d, q, dl):
        z = cosmo.z_of_dL(dl)
        m1 = m1d / (1 + z)
        return (ld(m1, q, z) + jnp.log(cosmo.dVCdz(z))
                - 2 * jnp.log1p(z) - jnp.log(cosmo.ddL_dz(z)))

    # per-event population weights of the PE samples (batched over events)
    logw = np.empty_like(m1d)
    B = 1000
    for i in range(0, nobs, B):
        s = slice(i, i + B)
        logw[s] = np.asarray(log_pop(jnp.asarray(m1d[s]), jnp.asarray(q[s]),
                                     jnp.asarray(dl[s])))

    M = logw.max(axis=1, keepdims=True)
    w = np.exp(np.where(np.isfinite(logw), logw - M, -np.inf))
    wsum = w.sum(axis=1)
    dead = wsum == 0
    print(f"events with all-zero population weight: {dead.sum()}")
    neff = wsum**2 / (w**2).sum(axis=1)
    print(f"population-weighted PE n_eff per event: "
          f"min {neff.min():.1f}  median {np.median(neff):.1f}")

    print(f"\n{'coord':6s} {'mean F':>8s} {'sd F':>7s} "
          f"{'frac F<0.1':>11s} {'frac F>0.9':>11s}   (uniform: 0.500 0.289 0.100 0.100)")
    rng = np.random.default_rng(3)
    pits = {}
    for name, samp, true in (("m1d", m1d, true_m1d), ("q", q, true_q),
                             ("dl", dl, true_dl)):
        below = (w * (samp < true[:, None])).sum(axis=1)
        at = (w * (samp == true[:, None])).sum(axis=1)
        F = (below + rng.uniform(size=nobs) * at) / np.where(dead, 1.0, wsum)
        F = F[~dead]
        pits[name] = F
        print(f"{name:6s} {F.mean():8.4f} {F.std():7.4f} "
              f"{(F < 0.1).mean():11.4f} {(F > 0.9).mean():11.4f}")

    # deciles for the worst-behaved coordinate views
    print("\nPIT deciles (each should contain 0.100):")
    edges = np.linspace(0, 1, 11)
    hdr = "  ".join(f"{e:5.1f}" for e in edges[1:])
    print(f"{'coord':6s} {hdr}")
    for name, F in pits.items():
        h, _ = np.histogram(F, bins=edges)
        print(f"{name:6s} " + "  ".join(f"{x/len(F):5.3f}" for x in h))

    from scipy.stats import kstest
    print()
    for name, F in pits.items():
        ks = kstest(F, "uniform")
        print(f"KS vs uniform, {name:4s}: D={ks.statistic:.4f}  p={ks.pvalue:.3e}")


if __name__ == "__main__":
    main()

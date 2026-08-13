"""Does the n_z=30 PISN z-grid corrugate the potential enough to move the
posterior in mpisndot / mpisn?

Gradients being wrong hurts HMC efficiency but not its stationary
distribution -- the potential is what sets the posterior.  So scan the real
data's potential finely along mpisndot and mpisn at several n_z and compare
the *profiles*, not the gradients.

Run from scripts/.
"""
import argparse
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--config", default="run_configs/mock_O5_evo.ini")
p.add_argument("--prior", default="../runs/priors/gwtc5_evo.prior")
p.add_argument("--nobs", type=int, default=None)
p.add_argument("--n_z", type=int, nargs="+", default=[30, 60, 120])
args = p.parse_args()

sys.path.append("../src/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import numpyro.handlers as handlers
import intensity_models_fast as im
from bench_model import load_real_data
from utils import get_priors_from_file

data, cfg = load_real_data(args.config)
if args.nobs:
    for k in ("m1s_det", "qs", "dls", "log_pdraw"):
        data[k] = data[k][: args.nobs]
nobs = data["m1s_det"].shape[0]
prior = get_priors_from_file(args.prior)
print(f"nobs={nobs} nsel={len(data['m1s_det_sel'])} n_z={args.n_z}  {jax.devices()}")

model_args = (
    data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
    data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
    data["Ndraw"], prior,
)
TRUTH = dict(
    h=0.674, Om=0.315, w=-1.0, a=-0.9426, b=0.237, c=2.360,
    mpisn=33.29, dmbhmax=36.7345 - 33.29, sigma=0.0539,
    log_fpl=float(np.log(0.63909)), lam=4.814, dkappa=8.3659 - 4.814,
    zp=0.954, beta=-2.43, msigma_low=4.0, mp_low=9.121,
    log_flow=float(np.log(0.6025)), mpisndot=0.0,
)


def logpost(params, n_z):
    vals = {k: jnp.asarray(v) for k, v in params.items()
            if k in prior and not isinstance(prior[k], float)}
    with handlers.seed(rng_seed=0), handlers.substitute(data=vals):
        tr = handlers.trace(im.pop_cosmo_model).get_trace(
            *model_args, use_low_bump=True, n_z=n_z)
    lp = 0.0
    for name, site in tr.items():
        if site["type"] != "sample":
            continue
        fn = site.get("fn")
        if fn is not None and type(fn).__name__ == "Unit":
            lp += float(np.asarray(fn.log_factor))
        elif site.get("value") is not None:
            try:
                lp += float(np.sum(np.asarray(fn.log_prob(site["value"]))))
            except Exception:
                pass
    return lp


def scan(key, grid, ref):
    print(f"\n=== logpost vs {key} (others at truth), relative to {key}={ref} ===")
    hdr = "".join(f"{'n_z=' + str(n):>14s}" for n in args.n_z)
    print(f"{key:>9s}{hdr}")
    base = {n: logpost(dict(TRUTH, **{key: ref}), n) for n in args.n_z}
    out = {}
    for v in grid:
        row = {n: logpost(dict(TRUTH, **{key: v}), n) - base[n] for n in args.n_z}
        out[v] = row
        print(f"{v:9.3f}" + "".join(f"{row[n]:14.3f}" for n in args.n_z))
    return out


# mpisndot: the parameter that failed.  Fine enough to see ripples between the
# ~30 z nodes, wide enough to see the trend that drove chains to the floor.
scan("mpisndot", np.round(np.arange(-2.0, 1.01, 0.125), 4), 0.0)
# mpisn at fixed mpisndot=-1: the where(mco < mpisn(z_i)) kinks live here.
TRUTH_M = dict(TRUTH, mpisndot=-1.0)
globals()["TRUTH"] = TRUTH_M
scan("mpisn", np.round(33.29 + np.arange(-0.5, 0.51, 0.05), 4), 33.29)

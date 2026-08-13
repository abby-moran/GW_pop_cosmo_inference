"""Posterior-wide z-grid sensitivity via importance reweighting.

The n_z convergence checks in diag_nz_profile.py probe fixed-parameter
slices; this reweights an actual posterior.  For each draw theta of a run
sampled at n_z=30, compute

    dlp(theta) = log p(theta | n_z=n_hi) - log p(theta | n_z=30)

(prior terms cancel; what remains is the z-discretization difference of
loglike + selfactor + guards).  Importance weights w = exp(dlp - max) give

  * the reweighting ESS: (sum w)^2 / sum w^2 -- near n_draws means the two
    discretizations define statistically indistinguishable posteriors;
  * weighted vs unweighted quantiles for the physics parameters -- the
    actual posterior shift a finer z grid would cause.

Run from scripts/, e.g.:
    python diag_nz_reweight.py --nc ../runs/endO5_evo2/O5_evo2.nc \
        --config run_configs/mock_O5_evo2.ini --prior ../runs/priors/gwtc5_evo.prior
"""
import argparse
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--nc", default="../runs/endO5_evo2/O5_evo2.nc")
p.add_argument("--config", default="run_configs/mock_O5_evo2.ini")
p.add_argument("--prior", default="../runs/priors/gwtc5_evo.prior")
p.add_argument("--n_hi", type=int, nargs="+", default=[60, 120])
p.add_argument("--thin", type=int, default=1,
               help="use every thin-th draw (the n_z=120 leg is run 4x thinner "
                    "than this on top)")
args = p.parse_args()

sys.path.append("../src/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import arviz as az
import jax
import jax.numpy as jnp
from numpyro.infer.util import log_density

import intensity_models_fast as im
from bench_model import load_real_data
from utils import get_priors_from_file

print(f"x64={jax.config.jax_enable_x64}  {jax.devices()}")

data, cfg = load_real_data(args.config)
prior = get_priors_from_file(args.prior)
free = sorted(k for k, v in prior.items() if not isinstance(v, float))
model_args = (
    data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
    data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
    data["Ndraw"], prior,
)

idata = az.from_netcdf(args.nc)
post = idata.posterior
sites = free + ["R_unit"]
draws = {k: np.asarray(post[k]).reshape(-1)[:: args.thin] for k in sites}
n = len(draws[sites[0]])
print(f"run: {args.nc}  free sites: {free}")
print(f"{n} draws (thin={args.thin})")


def make_lp(n_z):
    kw = dict(use_low_bump=True, n_z=n_z)

    @jax.jit
    def lp(params):
        return log_density(im.pop_cosmo_model, model_args, kw, params)[0]

    return lp


def run_leg(lp, idx):
    out = np.empty(len(idx))
    t0 = time.perf_counter()
    for j, i in enumerate(idx):
        params = {k: jnp.asarray(draws[k][i]) for k in sites}
        out[j] = float(lp(params))
    print(f"    {len(idx)} evals in {time.perf_counter() - t0:.1f}s")
    return out


def weighted_quantiles(x, w, qs=(0.16, 0.5, 0.84)):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    cw /= cw[-1]
    return np.interp(qs, cw, x[i])


idx_all = np.arange(n)
print("\nreference leg: n_z=30")
lp30 = run_leg(make_lp(30), idx_all)

report_params = [k for k in ("mpisndot", "mpisn", "dmbhmax", "sigma", "a", "b",
                             "c", "beta", "zp", "log_fpl", "log_flow",
                             "mp_low", "msigma_low", "lam", "dkappa") if k in draws]

for n_hi in args.n_hi:
    # the finer (more expensive) legs can run thinned; ESS is what matters
    sub = idx_all if n_hi <= 60 else idx_all[::4]
    print(f"\n=== reweight n_z=30 -> n_z={n_hi}  ({len(sub)} draws) ===")
    lp_hi = run_leg(make_lp(n_hi), sub)
    dlp = lp_hi - lp30[sub]
    dlp64 = dlp - dlp.max()
    w = np.exp(dlp64)
    ess = w.sum() ** 2 / (w * w).sum()
    print(f"  dlp: std={dlp.std():.4f}  min..max=[{dlp.min():+.4f}, {dlp.max():+.4f}] "
          f"(constant shift irrelevant)")
    print(f"  reweighting ESS = {ess:.1f} / {len(sub)}  ({ess / len(sub):.1%})")
    print(f"  {'param':11s} {'unweighted 16/50/84':>28s} {'reweighted 16/50/84':>28s} "
          f"{'shift/sigma':>12s}")
    for k in report_params:
        x = draws[k][sub]
        uq = np.percentile(x, [16, 50, 84])
        wq = weighted_quantiles(x, w)
        sig = 0.5 * (uq[2] - uq[0])
        shift = (wq[1] - uq[1]) / sig if sig > 0 else np.nan
        print(f"  {k:11s} [{uq[0]:8.4f} {uq[1]:8.4f} {uq[2]:8.4f}] "
              f"[{wq[0]:8.4f} {wq[1]:8.4f} {wq[2]:8.4f}] {shift:+12.3f}")

"""
Test D of the recovery-bias investigation: refit the SAME detected events as
the main validation run, but hand the sampler each event's TRUE
(m1_det, q, dL) as a single delta-function "PE sample".  This is the exact
known-parameters hierarchical likelihood -- the mock-PE machinery is bypassed
entirely, while the population model, selection estimate and rate bookkeeping
are identical to the main run.

  - biases persist  -> generation/selection/likelihood inconsistency
  - truth recovered -> the mock-PE machinery is the culprit

Run via call_inf_true.sh (2 GPUs, 2 parallel chains).
"""
import os
import sys
sys.path.append('../src/')

import argparse
import configparser

import numpy as np
import pandas as pd

import numpyro

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="run_configs/mock_O5_noevo_val.ini")
parser.add_argument("--nobs", type=int, default=9000)
parser.add_argument("--nmcmc", type=int, default=1000)
parser.add_argument("--nchain", type=int, default=2)
parser.add_argument("--out", default=None)
args = parser.parse_args()

numpyro.set_host_device_count(args.nchain)

import jax
import jax.numpy as jnp
import arviz as az
from numpyro.infer import MCMC, NUTS, init_to_value

import intensity_models_fast as intensity_models
from utils import get_priors_from_file

cfg = configparser.ConfigParser()
cfg.read(args.config)
run = cfg["run"]
run_dir = os.path.join("../runs", run["run_dir"])
out = args.out or os.path.join(run_dir, f"O5_val_true_{args.nobs}.nc")

prior = get_priors_from_file(os.path.join("../runs/priors", run["prior"]))

sel_file = os.path.join(run_dir, run["output_sel_file"])
sel_all = pd.read_hdf(sel_file, key="true_parameters")

# Events: the first `nobs` detections -- the same events (in the same order)
# whose mock PE the main validation run fit.
evt = sel_all.iloc[:args.nobs]
m1s = evt["m1d"].to_numpy()[:, None]
qs = evt["q"].to_numpy()[:, None]
dls = evt["dl"].to_numpy()[:, None]
log_pdraw = np.zeros_like(m1s)

# Selection: first half of the sel set, exactly as run_inf.py uses it.
half = int(np.round(len(sel_all) / 2))
sel = sel_all.iloc[:half]
ndraw = sel["ndraw"].iloc[0] / 2

truth = dict(h=0.674, a=-0.9426, b=0.237, c=2.360, mpisn=33.29,
             dmbhmax=36.7345 - 33.29, sigma=0.0539,
             log_fpl=float(np.log(0.63909)), lam=4.814,
             dkappa=8.3659 - 4.814, zp=0.954, beta=-2.43,
             msigma_low=4.0, mp_low=9.121, log_flow=float(np.log(0.6025)),
             mpisndot=0.0)
init = {k: jnp.asarray(v) for k, v in truth.items()
        if k in prior and not isinstance(prior[k], float)}

print(f"events: {m1s.shape[0]} x {m1s.shape[1]} (true parameters, 1 sample each)")
print(f"selection: {len(sel)} samples, Ndraw = {ndraw:.3e}")
print(f"devices: {jax.devices()}")

kernel = NUTS(intensity_models.pop_cosmo_model,
              init_strategy=init_to_value(values=init), max_tree_depth=7)
mcmc = MCMC(kernel, num_warmup=args.nmcmc, num_samples=args.nmcmc,
            num_chains=args.nchain, chain_method="parallel", progress_bar=True)
# neff_penalty="none": with one sample per event, every per-event n_eff is 1,
# so both n_eff-based guards would be (meaninglessly) active.  The per-event
# term needs no MC guard here -- it is exact.
mcmc.run(jax.random.PRNGKey(1652819403), m1s, qs, dls, log_pdraw,
         sel["m1d"].to_numpy(), sel["q"].to_numpy(), sel["dl"].to_numpy(),
         sel["pdraw_sel"].to_numpy(), ndraw, prior,
         use_low_bump=True, neff_penalty="none")

idata = az.from_numpyro(mcmc, num_chains=args.nchain)
az.to_netcdf(idata, out)
print("saved", out)

post = idata.posterior
print(f"\n{'param':12s} {'truth':>9s} {'post mean':>10s} {'post sd':>9s} {'z':>6s}")
for p, t in truth.items():
    if p not in post:
        continue
    m, s = float(post[p].mean()), float(post[p].std())
    z = (m - t) / s if s > 0 else float("nan")
    print(f"{p:12s} {t:9.4f} {m:10.4f} {s:9.4f} {z:6.2f}")

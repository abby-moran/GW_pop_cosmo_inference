import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import sys
sys.path.append('../src/')
# Optimized drop-in for intensity_models (~25x faster and 5x less GPU memory;
# see scripts/test_fast_equivalence.py and scripts/bench_model.py).  Revert to
# the original by importing intensity_models.
# NOTE: by default this samples with smooth_tail_edge=True, a small change to
# the population model (the power-law tail is no longer hard-zeroed below
# m = mbhmax) that makes the density continuous and the NUTS gradients for h,
# mpisn, dmbhmax correct.  Pass smooth_tail_edge=False to pop_cosmo_model (via
# mcmc.run kwargs below) to reproduce the original model exactly.
# The Monte-Carlo accuracy guard also defaults to neff_penalty="mc_variance"
# (penalize sum_i 1/n_eff_i above mc_variance_budget=5, i.e. MC sigma of the
# total log likelihood above ~2.2 nats); pass neff_penalty="min_neff" to
# reproduce the original min-over-events n_eff >= nobs guard.
import intensity_models_fast as intensity_models
import numpy as np
import pandas as pd
from utils import get_priors_from_file
import arviz as az
import configparser
import argparse
from pathlib import Path
import h5py
import jax

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to run config file")
args = parser.parse_args()

cfg = configparser.ConfigParser()
cfg.read(args.config)
run = cfg["run"]
base_runs_dir = "../runs"
run_name = cfg["run"]["run_dir"]
run_dir = os.path.join(base_runs_dir, f"{run_name}")

outfile = os.path.join(run_dir, cfg["run"]["mcmc_out"])
prior_dir="../runs/priors"
prior = os.path.join(prior_dir, cfg["run"]["prior"])

nmcmc = run.getint("nmcmc")
nchain = run.getint("nchain")

evt_start = run.getint("evt_start")
evt_end= run.get("evt_end")
if evt_end is None or evt_end.lower() == "none":
    evt_end = None
else:
    evt_end = int(evt_end)

def load_true_vals(filename):
    tv = {}

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                tv[key.strip()] = float(val.strip())

    # transformed params expected by model
    tv['dkappa'] = tv['kappa'] - tv['lam']
    tv['log_fpl'] = np.log(tv['fpl'])
    tv['dmbhmax'] = tv['mbhmax'] - tv['mpisn']

    if 'flow' in tv:
        tv['log_flow'] = jnp.log(tv['flow'])
    else:
        tv['log_flow']=np.log(1e-5)
    tv['log_fpl'] = jnp.log(tv['fpl'])


    # remove unused raw params if desired
    del tv['kappa']
    del tv['fpl']
    del tv['mbhmax']

    return {k: jnp.array(v) for k, v in tv.items()}

pe_file = os.path.join(run_dir, cfg["run"]["output_file_PE"])
sel_file = os.path.join(run_dir, cfg["run"]["output_sel_file"])
ndevice=nchain
use_low_bump=run.getboolean("use_low_bump", fallback=True)

truth_params = {}
truth_file_name = cfg["run"].get("pop_config_file")

if truth_file_name is None or str(truth_file_name).lower() == "none":
    truth_file_name = None

config_dir = 'pop_configs'

import numpyro
numpyro.set_host_device_count(ndevice)
import jax
from numpyro.infer import MCMC, NUTS, SA
import jax.numpy as jnp
from numpyro.infer import init_to_value, init_to_sample
from numpyro.infer import DiscreteHMCGibbs, NUTS, MCMC


if truth_file_name is not None:
    truth_file_path = os.path.join(config_dir, truth_file_name)
    print("Loading truth parameters from: ", truth_file_path)
    truth_params=load_true_vals(truth_file_path)

if __name__ == "__main__":

    #nmcmc =  1800
    #nchain = 1
    random_seed = 1652819403
    print("loading in prior file: ", prior)
    prior = get_priors_from_file(prior)
    
    try:
        with h5py.File(pe_file, "r") as f:
            m1s = f["m1"][evt_start:evt_end]
            qs = f["q"][evt_start:evt_end]
            dls = f["dl"][evt_start:evt_end]
            pdraws = f["pdraw"][evt_start:evt_end]

        print(f"Loaded new-format HDF5 file {pe_file},  events {evt_start} to {evt_end}")

    except (KeyError, OSError):
        pe_samples_mock = pd.read_hdf(pe_file, key="samples").iloc[evt_start:evt_end]
        print(f"Loaded legacy-format HDF5 file {pe_file}, events {evt_start} to {evt_end}")

        m1s = np.asarray(pe_samples_mock["m1"].to_list())
        qs = np.asarray(pe_samples_mock["q"].to_list())
        dls = np.asarray(pe_samples_mock["dl"].to_list())
        pdraws = np.asarray(pe_samples_mock["pdraw"].to_list())

    pdraws = jnp.nan_to_num(pdraws, neginf=-1e30, posinf=1e30)

    print("array shapes (we want nevents, nsamples):",
        m1s.shape, qs.shape, dls.shape, pdraws.shape)
    
    sel_samples=pd.read_hdf(sel_file, key='true_parameters')#, start=0, stop=371545)
    len_sel=len(sel_samples)
    sel_samples=pd.read_hdf(sel_file, key='true_parameters', start=0, stop=int(np.round(len_sel/2)))

    ndraw=sel_samples['ndraw'].iloc[0]/2

    assert np.all(m1s > 0) 
    assert np.all(qs > 0) 
    #assert np.all(dls > 0) 
    assert np.all(qs<=1) 
    assert not np.any(np.isnan(pdraws)) 
    assert not np.any(np.isinf(pdraws))
    assert not np.any(np.isnan(sel_samples['pdraw_sel'])) 
    assert not np.any(np.isinf(sel_samples['pdraw_sel']))
    
    init_strategy = init_to_value(values=truth_params) if truth_params else init_to_sample
    print(truth_params)
    #kernel = DiscreteHMCGibbs(NUTS(intensity_models.pop_cosmo_model, init_strategy=init_strategy))

    model_args = (m1s, qs, dls, pdraws, sel_samples['m1d'].to_list(),
                  sel_samples['q'].to_list(), sel_samples['dl'].to_list(),
                  sel_samples['pdraw_sel'].to_list(), ndraw, prior)

    # Float32 recentering: evaluate the per-event log likelihoods and
    # log_mu_sel once at the init point and subtract them as constant
    # baselines inside the model's sums.  A constant shift of the potential is
    # invisible to MCMC, but it removes the dominant float32 roundoff term
    # (1 ulp of the ~16*nobs log-likelihood sum; 1.9e-2 nats at nobs=9000,
    # growing linearly with nobs).  The recorded 'lp' is shifted by the
    # printed offset.  See notes/2026-08-07-float32-recentering.md.
    # (Skipped automatically if the original intensity_models module is used,
    # which has no recentering support.)
    recenter_kwargs = {}
    baselines = None
    if hasattr(intensity_models, "recentering_baselines"):
        baselines = intensity_models.recentering_baselines(
            model_args, truth_params, use_low_bump=use_low_bump)
        recenter_kwargs = dict(loglike_ref=baselines['loglike_ref'],
                               log_mu_sel_ref=baselines['log_mu_sel_ref'])
        print(f"recentering baselines: log_mu_sel_ref = {baselines['log_mu_sel_ref']:.6f}, "
              f"dropped potential offset = {baselines['offset']:.6e} "
              f"(add to the centered 'loglike' factor for absolute values)")

    kernel = NUTS(intensity_models.pop_cosmo_model, init_strategy=init_strategy, max_tree_depth=7)#, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain,
                chain_method="parallel", progress_bar=True)
    # NUTS only collects ('z', 'diverging') by default, so a finished run used to
    # carry no record of sampler health -- energy errors, acceptance and tree
    # depth were all discarded and could not be checked after the fact.  These
    # are scalars per sample (a few kB per chain), and arviz maps them into
    # sample_stats as lp, energy, acceptance_rate, n_steps, tree_depth and
    # step_size.  See notes/2026-08-07-float32-accuracy-audit.md.
    mcmc.run(jax.random.PRNGKey(random_seed), *model_args, use_low_bump=use_low_bump,
        **recenter_kwargs,
        extra_fields=("potential_energy", "energy", "num_steps", "accept_prob",
                      "adapt_state.step_size"))
    #outfile="o3_c2_zm55_err5k.npz"
    samples = az.from_numpyro(mcmc, num_chains=nchain)
    # The centered run's lp/energy are shifted by a constant; keep the offset
    # with the output so absolute log-likelihood values remain recoverable.
    if baselines is not None:
        samples.posterior.attrs["recentering_offset"] = baselines["offset"]
    az.to_netcdf(samples, outfile)
    print("Saved samples to " + outfile)

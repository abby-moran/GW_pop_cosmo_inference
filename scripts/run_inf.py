import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import sys
sys.path.append('../src/')
import intensity_models
import numpy as np
import pandas as pd
from utils import get_priors_from_file
import arviz as az
import configparser
import argparse
from pathlib import Path

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

pe_file = os.path.join(run_dir, cfg["run"]["output_file_PE"])
sel_file = os.path.join(run_dir, cfg["run"]["output_sel_file"])
ndevice=nchain

truth_params = {}
truth_file_name = cfg["run"].get("pop_config_file", None)

import numpyro
numpyro.set_host_device_count(ndevice)
import jax
from numpyro.infer import MCMC, NUTS, SA
import jax.numpy as jnp

if truth_file_name:
    truth_file_path = os.path.join(run_dir, truth_file_name)
    print("Loading truth parameters from: ", truth_file_path)
    with open(truth_file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            truth_params[key.strip()] = jnp.array(float(val.strip()))

if __name__ == "__main__":

    #nmcmc =  1800
    #nchain = 1
    random_seed = 1652819403
    print("loading in prior file: ", prior)
    prior = get_priors_from_file(prior)
    pe_samples_mock = pd.read_hdf(pe_file, key='samples').iloc[evt_start:evt_end]# 1.5 to 2.5 k on the 1k tests
    print(f'loaded in {pe_file}, events {evt_start} to {evt_end}')
    m1s = np.asarray(pe_samples_mock['m1'].to_list())
    qs = np.asarray(pe_samples_mock['q'].to_list())#[:, :1000]
    dls =  np.asarray(pe_samples_mock['dl'].to_list())
    pdraws = np.asarray(pe_samples_mock['pdraw'].to_list())
    pdraws= jnp.nan_to_num(pdraws, neginf=-1e30, posinf=1e30)
    print("array shapes (we want nevents, nsamples): ", m1s.shape, qs.shape, dls.shape, pdraws.shape)

    sel_samples=pd.read_hdf(sel_file, key='true_parameters')
    ndraw=sel_samples['ndraw'].iloc[0]

    assert np.all(m1s > 0) 
    assert np.all(qs > 0) 
    #assert np.all(dls > 0) 
    assert np.all(qs<=1) 
    assert not np.any(np.isnan(pdraws)) 
    assert not np.any(np.isinf(pdraws))
    assert not np.any(np.isnan(sel_samples['pdraw_sel'])) 
    assert not np.any(np.isinf(sel_samples['pdraw_sel']))
    
    init_params = {k: jnp.stack([v] * nchain) for k, v in truth_params.items()} if truth_params else None

    kernel = NUTS(intensity_models.pop_cosmo_model)
    mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain,
                chain_method="parallel", progress_bar=True)
    mcmc.run(jax.random.PRNGKey(random_seed), m1s, qs, dls, pdraws, sel_samples['m1d'].to_list(), 
             sel_samples['q'].to_list(), sel_samples['dl'].to_list(), sel_samples['pdraw_sel'].to_list(),
        ndraw, prior, init_params=init_params)
    #outfile="o3_c2_zm55_err5k.npz"
    samples = az.from_numpyro(mcmc, num_chains=nchain)
    az.to_netcdf(samples, outfile)
    print("Saved samples to " + outfile)
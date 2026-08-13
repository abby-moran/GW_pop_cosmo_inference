"""Conditional scan of the model log-density over sigma (all else at truth).

Compares the evo7 (width15) dataset against the evo6 (val2) dataset to
locate where the sigma information was lost.  Records the decomposition:
per-event loglike sum, log_mu_sel, mc_var_loglike, min_neff, neff_sel.
"""
import os, sys, configparser
sys.path.append('../src/')
import numpy as np
import pandas as pd
import h5py
import jax
import jax.numpy as jnp
import numpyro
import numpyro.handlers as handlers
import intensity_models_fast as intensity_models
from utils import get_priors_from_file


def load_true_vals(filename):
    tv = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                tv[key.strip()] = float(val.strip())
    tv['dkappa'] = tv['kappa'] - tv['lam']
    tv['log_fpl'] = np.log(tv['fpl'])
    tv['dmbhmax'] = tv['mbhmax'] - tv['mpisn']
    tv['log_flow'] = np.log(tv['flow']) if 'flow' in tv else np.log(1e-5)
    if 'msigma_low' in tv:
        tv['log_fpeak'] = tv['log_flow'] - np.log(tv['msigma_low'])
    if 'Omh2' not in tv and 'Om' in tv and 'h' in tv:
        tv['Omh2'] = tv['Om'] * tv['h'] ** 2
    del tv['kappa']; del tv['fpl']; del tv['mbhmax']
    return {k: jnp.array(v) for k, v in tv.items()}


def load_dataset(ini_path):
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)
    run = cfg["run"]
    run_dir = os.path.join("../runs", run["run_dir"])
    evt_start = run.getint("evt_start")
    evt_end = run.getint("evt_end")
    pe_file = os.path.join(run_dir, run["output_file_PE"])
    sel_file = os.path.join(run_dir, run["output_sel_file"])
    prior = get_priors_from_file(os.path.join("../runs/priors", run["prior"]))
    truths = load_true_vals(os.path.join("pop_configs", run["pop_config_file"]))
    truths = intensity_models.map_truths_to_prior_coords(truths, prior)

    with h5py.File(pe_file, "r") as f:
        m1s = f["m1"][evt_start:evt_end]
        qs = f["q"][evt_start:evt_end]
        dls = f["dl"][evt_start:evt_end]
        pdraws = f["pdraw"][evt_start:evt_end]
    pdraws = jnp.nan_to_num(jnp.asarray(pdraws), neginf=-1e30, posinf=1e30)

    sel = pd.read_hdf(sel_file, key='true_parameters')
    sel = pd.read_hdf(sel_file, key='true_parameters',
                      start=0, stop=int(np.round(len(sel) / 2)))
    ndraw = sel['ndraw'].iloc[0] / 2
    model_args = (m1s, qs, dls, pdraws, sel['m1d'].to_list(), sel['q'].to_list(),
                  sel['dl'].to_list(), sel['pdraw_sel'].to_list(), ndraw, prior)
    return model_args, truths, evt_end - evt_start


def load_mixed(pe_ini, sel_ini):
    """PE + truths + prior from pe_ini, selection set from sel_ini."""
    args_pe, truths, nobs = load_dataset(pe_ini)
    args_sel, _, _ = load_dataset(sel_ini)
    model_args = args_pe[:4] + args_sel[4:9] + (args_pe[9],)
    return model_args, truths, nobs


def scan(name, ini_path, sigmas, sel_ini=None, **model_kwargs):
    if sel_ini is not None:
        model_args, truths, nobs = load_mixed(ini_path, sel_ini)
    else:
        model_args, truths, nobs = load_dataset(ini_path)
    rows = []
    for s in sigmas:
        pars = dict(truths)
        pars['sigma'] = jnp.asarray(s)
        pars = {k: jnp.asarray(v) for k, v in pars.items()}
        with handlers.seed(rng_seed=0), handlers.substitute(data=pars):
            tr = handlers.trace(intensity_models.pop_cosmo_model).get_trace(
                *model_args, store_per_event=True, **model_kwargs)
        ll = np.asarray(tr["loglik_array_dim"]["value"], dtype=np.float64)
        log_mu = float(np.asarray(tr["log_mu_sel"]["value"]))
        mc_var = float(np.asarray(tr["mc_var_loglike"]["value"]))
        min_neff = float(np.asarray(tr["min_neff"]["value"]))
        neff_sel = float(np.asarray(tr["neff_sel"]["value"]))
        # R-marginalized log-likelihood factor (up to sigma-independent consts)
        tot = ll.sum() - nobs * log_mu
        rows.append((s, tot, ll.sum(), log_mu, mc_var, min_neff, neff_sel))
        print(f"{name} sigma={s:.4f}: loglike_factor={tot:.2f} "
              f"sum_ll={ll.sum():.2f} log_mu_sel={log_mu:.6f} "
              f"mc_var={mc_var:.3f} min_neff={min_neff:.2f} neff_sel={neff_sel:.0f}",
              flush=True)
    df = pd.DataFrame(rows, columns=['sigma', 'loglike_factor', 'sum_ll',
                                     'log_mu_sel', 'mc_var', 'min_neff', 'neff_sel'])
    df.to_csv(f'/tmp/sigma_scan_{name}.csv', index=False)
    return df


if __name__ == "__main__":
    sigmas = [0.0505, 0.0539, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.15, 0.18, 0.22]
    which = sys.argv[1] if len(sys.argv) > 1 else 'tab'
    if which == 'tab':
        scan('width15', 'run_configs/mock_O5_fullcosmo_evo7.ini', sigmas)
        scan('val2', 'run_configs/mock_O5_fullcosmo_evo6.ini', sigmas)
    elif which == 'direct':
        scan('width15_direct', 'run_configs/mock_O5_fullcosmo_evo7.ini', sigmas,
             tabulate_mass_function=False, tabulate_selection=False)
        scan('val2_direct', 'run_configs/mock_O5_fullcosmo_evo6.ini', sigmas,
             tabulate_mass_function=False, tabulate_selection=False)
    elif which == 'mix':
        scan('w15pe_val2sel', 'run_configs/mock_O5_fullcosmo_evo7.ini', sigmas,
             sel_ini='run_configs/mock_O5_fullcosmo_evo6.ini')
        scan('val2pe_w15sel', 'run_configs/mock_O5_fullcosmo_evo6.ini', sigmas,
             sel_ini='run_configs/mock_O5_fullcosmo_evo7.ini')

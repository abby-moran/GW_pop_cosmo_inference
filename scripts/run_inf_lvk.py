"""LVK-control counterpart of run_inf.py: identical data handling, sampler
setup, and output format, but with the GWTC-5 Default "PowerLaw + 2 Peaks"
mass model (intensity_models_lvk) instead of the PISN model.  The PISN-only
options (use_low_bump, smooth_tail_edge, tail_anchor) do not exist here.
The optional ini key `pairing` ("lvk" default, or "mt") selects the pairing
convention: "mt" pairs the LVK mass function the way the PISN model does
(total-mass power law), for the mass-model vs pairing-function control."""
import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import sys
sys.path.append('../src/')
import intensity_models_lvk as intensity_models
import numpy as np
import pandas as pd
from utils import get_priors_from_file
import arviz as az
import configparser
import argparse
import h5py
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
from numpyro.infer import init_to_value, init_to_sample

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
prior_dir="../scripts/priors"
prior = os.path.join(prior_dir, cfg["run"]["prior"])

nmcmc = run.getint("nmcmc")
nchain = run.getint("nchain")
# Static pairing convention: "lvk" (q^beta, default) or "mt" (the PISN
# model's total-mass pairing of the same LVK mass function).
pairing = run.get("pairing", fallback="lvk")

# Optional float64 mode (`x64 = true` in [run], default false); same contract
# as run_inf.py (see notes/2026-09-01-h-divergences-float32.md).  Must run
# before any JAX array is created.  The LVK model has no tabulated/Pallas
# lookups, so no scatter_free_tables handling is needed here.
x64 = run.getboolean("x64", fallback=False)
if x64:
    jax.config.update("jax_enable_x64", True)
    print("x64 = true: jax_enable_x64 enabled")

evt_start = run.getint("evt_start")
evt_end= run.get("evt_end")
if evt_end is None or evt_end.lower() == "none":
    evt_end = None
else:
    evt_end = int(evt_end)
def load_true_vals_lvk(filename):
    """Truths for an LVK-model mock: pop configs store the physical
    parameters; derive the sampled coordinates the prior uses."""
    tv = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                tv[key.strip()] = float(val.strip())

    # Madau-Dickinson pop config -> the prior's dkappa coordinate.  Harmless
    # for a pure-power-law prior (lvk_gwtc5_plz.prior): dkappa/zp are then not
    # sample sites, and both init_to_value and the recentering `substitute`
    # ignore truth entries that name no site.
    if 'kappa' in tv and 'dkappa' not in tv:
        tv['dkappa'] = tv['kappa'] - tv['lam']
        del tv['kappa']
    if 'Omh2' not in tv and 'Om' in tv and 'h' in tv:
        tv['Omh2'] = tv['Om'] * tv['h'] ** 2
    # Flat mixture fractions -> stick-breaking coordinates.
    if 'frac_bpl' in tv and 'f_peaks' not in tv:
        tv['f_peaks'] = 1.0 - tv.pop('frac_bpl')
        if 'frac_p1' in tv and 'frac_p2' in tv:
            tv['f_p1'] = tv.pop('frac_p1') / max(tv['f_peaks'], 1e-30)
            del tv['frac_p2']
    return {k: jnp.array(v) for k, v in tv.items()}

pe_file = os.path.join(run_dir, cfg["run"]["output_file_PE"])
sel_file = os.path.join(run_dir, cfg["run"]["output_sel_file"])
ndevice=nchain
numpyro.set_host_device_count(ndevice)
max_tree_depth=run.getint("max_tree_depth", fallback=7)
dense_mass=run.getboolean("dense_mass", fallback=False)
target_accept_prob=run.getfloat("target_accept_prob", fallback=0.8)

truth_params = {}
truth_file_name = cfg["run"].get("pop_config_file")

if truth_file_name is None or str(truth_file_name).lower() == "none":
    truth_file_name = None
config_dir = 'pop_configs'

if truth_file_name is not None:
    truth_file_path = os.path.join(config_dir, truth_file_name)
    print("Loading truth parameters from: ", truth_file_path)
    truth_params=load_true_vals_lvk(truth_file_path)

if __name__ == "__main__":
    random_seed = 1652819403
    print("loading in prior file: ", prior)
    prior_file_name = prior
    prior = get_priors_from_file(prior)

    # map the truths into the prior's coordinates to start at the truth
    if truth_params and hasattr(intensity_models, "map_truths_to_prior_coords"):
        truth_params = intensity_models.map_truths_to_prior_coords(truth_params, prior)

    # Optional [ref_params] ini section: explicit values for the float32
    # recentering reference point (same convention as run_inf.py; names must
    # be SAMPLED sites of the prior).  Missing names still come from the
    # prior draw at seed 0 inside recentering_baselines.
    ref_params = dict(truth_params)
    _case_cfg = configparser.ConfigParser()
    _case_cfg.optionxform = str  # preserve parameter-name case (e.g. Omh2)
    _case_cfg.read(args.config)
    if _case_cfg.has_section("ref_params"):
        for name, val in _case_cfg["ref_params"].items():
            if name not in prior:
                raise ValueError(
                    f"[ref_params] {name}: not a parameter of {prior_file_name}")
            if isinstance(prior[name], float):
                raise ValueError(
                    f"[ref_params] {name}: fixed to {prior[name]} in "
                    f"{prior_file_name}, not a sampled site; remove it")
            ref_params[name] = jnp.asarray(float(val))
        print("recentering [ref_params] overrides:",
              {k: float(v) for k, v in ref_params.items()})

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

    if x64:
        # jnp.asarray preserves a float32 input dtype even under
        # jax_enable_x64, so upcast on the numpy side.
        m1s, qs, dls, pdraws = (np.asarray(a, dtype=np.float64)
                                for a in (m1s, qs, dls, pdraws))

    pdraws = jnp.nan_to_num(pdraws, neginf=-1e30, posinf=1e30)
    print("array shapes (we want nevents, nsamples):", m1s.shape, qs.shape, dls.shape, pdraws.shape)

    sel_samples = pd.read_hdf(sel_file, key='true_parameters')
    ndraw = sel_samples['ndraw'].iloc[0]
    print(f"selection set: {len(sel_samples)} rows, ndraw={ndraw}")

    assert np.all(m1s > 0)
    assert np.all(qs > 0)
    assert np.all(qs<=1)
    assert not np.any(np.isnan(pdraws))
    assert not np.any(np.isinf(pdraws))
    assert not np.any(np.isnan(sel_samples['pdraw_sel']))
    assert not np.any(np.isinf(sel_samples['pdraw_sel']))

    init_strategy = init_to_value(values=truth_params) if truth_params else init_to_sample
    print(truth_params)

    model_args = (m1s, qs, dls, pdraws, sel_samples['m1d'].to_list(),
                  sel_samples['q'].to_list(), sel_samples['dl'].to_list(),
                  sel_samples['pdraw_sel'].to_list(), ndraw, prior)

    # per-event log likelihoods at init point, subtracted inside the event sum
    # (same for the selection sum) to keep the float32 potential near 0;
    # R and log_mu_sel are corrected back to absolute
    recenter_kwargs = {}
    baselines = None
    if hasattr(intensity_models, "recentering_baselines"):
        baselines = intensity_models.recentering_baselines(model_args, ref_params,
                                                           pairing=pairing)
        recenter_kwargs = dict(
            loglike_ref=baselines['loglike_ref'],
            log_mu_sel_ref=baselines['log_mu_sel_ref'],
            log_pdraw_sel_scale=baselines['log_pdraw_sel_scale'],)

        print(f"recentering baselines: log_pdraw_sel_scale = "
              f"{baselines['log_pdraw_sel_scale']:.6f} "
              f"(physical log_mu_sel at ref; scaled ref = "
              f"{baselines['log_mu_sel_ref']:.1f}), "
              f"dropped potential offset = {baselines['offset']:.6e} "
              f"(add to the centered 'loglike' factor for absolute values)")

    print(f"NUTS: max_tree_depth={max_tree_depth} dense_mass={dense_mass} " f"target_accept_prob={target_accept_prob}")
    kernel = NUTS(intensity_models.pop_cosmo_model, init_strategy=init_strategy,
                  max_tree_depth=max_tree_depth, dense_mass=dense_mass, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain, chain_method="parallel", progress_bar=True)
    # make sure we get sample stats to check the sampler health post run
    mcmc.run(jax.random.PRNGKey(random_seed), *model_args, pairing=pairing,
        **recenter_kwargs, extra_fields=("potential_energy", "energy", "num_steps", "accept_prob","adapt_state.step_size"))
    samples = az.from_numpyro(mcmc, num_chains=nchain)

    # centered run lp/energy are shifted - record offset to recover later
    if baselines is not None:
        samples.posterior.attrs["recentering_offset"] = baselines["offset"]
        samples.posterior.attrs["log_pdraw_sel_scale"] = baselines["log_pdraw_sel_scale"]
    # mark the model family so post-processing scripts can dispatch
    samples.posterior.attrs["mass_model"] = "lvk_pl2p" if pairing == "lvk" else "lvk_pl2p_mt"
    # store the full run config in the output so runs are self-documenting
    # (raw ini strings; prefixed to avoid clashing with other attrs)
    samples.posterior.attrs["run_config_file"] = str(args.config)
    for key, value in cfg["run"].items():
        samples.posterior.attrs[f"run_config_{key}"] = value
    # also embed the full text of the prior / pop config so the .nc stays
    # self-contained even if those files are later edited, moved, or lost
    with open(prior_file_name) as f:
        samples.posterior.attrs["prior_file_contents"] = f.read()
    if truth_file_name is not None:
        with open(truth_file_path) as f:
            samples.posterior.attrs["pop_config_file_contents"] = f.read()
    az.to_netcdf(samples, outfile)
    print("Saved samples to " + outfile)

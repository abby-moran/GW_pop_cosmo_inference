import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import sys
sys.path.append('../src/')
import intensity_models_fast as intensity_models
import numpy as np
import pandas as pd
from utils import get_priors_from_file, warn_if_bump_too_broad, warn_log_flow_deprecated
import arviz as az
import configparser
import argparse
from pathlib import Path
import h5py
import jax
import numpyro
from numpyro.infer import MCMC, NUTS, SA
import jax.numpy as jnp
from numpyro.infer import init_to_value, init_to_sample
from numpyro.infer import NUTS, MCMC

# Revert to the original (5-25x slower) by importing intensity_models
# by default samples with smooth_tail_edge=True, pass smooth_tail_edge=False to revert
# defaults to neff_penalty="mc_variance", pass neff_penalty="min_neff" to revert

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

# Optional float64 mode (`x64 = true` in [run], default false): eliminates the
# float32-roughness divergences of h-free runs at ~2-3x wall cost (see
# notes/2026-09-01-h-divergences-float32.md).  jax_enable_x64 must be set
# BEFORE any JAX array is created; this point precedes all data loading, the
# truth/ref-param jnp conversions below, and model tracing (none of the
# imported src modules create JAX arrays at import time).  x64 also forces
# scatter_free_tables = False: the Pallas scatter-free VJP kernels are
# float32-only, so the tabulated lookups take the plain-gather gradient path.
x64 = run.getboolean("x64", fallback=False)
scatter_free_tables = run.getboolean("scatter_free_tables", fallback=None)
if x64:
    if scatter_free_tables:
        raise ValueError(
            "x64 = true is incompatible with scatter_free_tables = true: the "
            "Pallas scatter-free kernels are float32-only.  Remove the "
            "scatter_free_tables line from the ini (x64 forces it off).")
    scatter_free_tables = False
    jax.config.update("jax_enable_x64", True)
    print("x64 = true: jax_enable_x64 enabled, scatter_free_tables forced off")

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

    # Peak-height parametrization of the bump amplitude (log_fpeak priors).
    if 'msigma_low' in tv:
        tv['log_fpeak'] = tv['log_flow'] - jnp.log(tv['msigma_low'])
    # Default prior samples Omh2 = Om*h^2; pop_configs still store Om
    if 'Omh2' not in tv and 'Om' in tv and 'h' in tv:
        tv['Omh2'] = tv['Om'] * tv['h'] ** 2

    # remove unused raw params
    del tv['kappa']
    del tv['fpl']
    del tv['mbhmax']
    return {k: jnp.array(v) for k, v in tv.items()}

# max_tree_depth=7 saturates in 99% of its
# dense_mass lets NUTS learn the h / Omh2 / mpisn / mpisndot covariance 

pe_file = os.path.join(run_dir, cfg["run"]["output_file_PE"])
sel_file = os.path.join(run_dir, cfg["run"]["output_sel_file"])
ndevice=nchain
numpyro.set_host_device_count(ndevice)
use_low_bump=run.getboolean("use_low_bump", fallback=True)
smooth_tail_edge=run.getboolean("smooth_tail_edge", fallback=True)
# "simplex" (default), "ref_z", or "per_z"; the r-modes need a prior sampling log_r
tail_anchor=run.get("tail_anchor", fallback="simplex")
# "mt" (default, total-mass pairing) or "lvk" (normalized q^beta pairing)
pairing=run.get("pairing", fallback="mt")
# Two-scale h-pivot reparametrization (`h_pivot = true`, default false):
# samples flat u_mp_low / u_mpisn_ref and derives mp_low = u_mp_low * h**-g1,
# mpisn_ref = u_mpisn_ref * h**-g2 with a factor restoring the original prior;
# exact change of variables (posterior unchanged, ~2x ESS(h)); requires a
# prior that samples h.  See notes/2026-09-01-h-divergences-float32.md.
h_pivot=run.getboolean("h_pivot", fallback=False)
h_pivot_gammas={"mp_low": run.getfloat("h_pivot_gamma_mp_low", fallback=0.14),
                "mpisn_ref": run.getfloat("h_pivot_gamma_mpisn_ref", fallback=0.18)}
# only pass the kwargs when the pivot is on so the default path is untouched
hp_kwargs = (dict(h_pivot=True, h_pivot_gammas=h_pivot_gammas) if h_pivot
             else {})
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
    truth_params=load_true_vals(truth_file_path)

if __name__ == "__main__":
    random_seed = 1652819403
    print("loading in prior file: ", prior)
    prior_file_name = prior
    prior = get_priors_from_file(prior)

    # high msigma_low prior makes `a' unidentifiable - check
    _msl = prior.get("msigma_low")
    if _msl is not None:
        _msl_high = _msl if isinstance(_msl, float) else getattr(_msl, "high", None)
        if _msl_high is None:
            _msl_high = getattr(getattr(_msl, "support", None), "upper_bound", None)
        warn_if_bump_too_broad(_msl_high, context=f"prior upper bound, {prior_file_name}")

    # Sampling log_flow is deprecated in favor of log_fpeak.
    if "log_flow" in prior and "log_fpeak" not in prior:
        warn_log_flow_deprecated(context=prior_file_name)

    # map the truths into the prior's coordinates to start at the truth
    if truth_params and hasattr(intensity_models, "map_truths_to_prior_coords"):
        truth_params = intensity_models.map_truths_to_prior_coords(truth_params, prior)

    def _h_pivot_remap(params, context):
        """h_pivot on: pin the u-sites, not mp_low/mpisn_ref (init_to_value
        and recentering_baselines substitute SAMPLE sites, and the pivoted
        scales are deterministics).  Auto-remap physical values the user
        provided: u = m * h**gamma (inverse of m = u * h**-gamma)."""
        names = [n for n in h_pivot_gammas
                 if n in params and n in prior
                 and not isinstance(prior[n], float)]
        if not names:
            return params
        if "h" not in params:
            raise ValueError(
                f"h_pivot = true: remapping {names} in {context} to their "
                f"u-sites needs h; add h to {context} (or provide "
                f"u_{names[0]} etc. directly)")
        out = dict(params)
        for name in names:
            u = float(out.pop(name)) * float(out["h"]) ** h_pivot_gammas[name]
            out[f"u_{name}"] = jnp.asarray(u)
            print(f"h_pivot: remapped {context} {name} -> u_{name} = {u:.6g}")
        return out

    if h_pivot and truth_params:
        truth_params = _h_pivot_remap(truth_params, "truth/init params")

    # Optional [ref_params] ini section: explicit values for the float32
    # recentering reference point (name = float lines).  Merged ON TOP of
    # truth_params: provided names pin those sample sites, anything missing
    # still comes from the prior draw at seed 0 inside recentering_baselines.
    # Real-data runs have no truths, so without this section the reference is
    # a pure prior draw, which can land outside the posterior's support (see
    # notes/2026-08-31-recentering-reference-fix.md).  Names must be SAMPLED
    # sites of the prior: fixed floats and derived deterministics would be
    # silently overridden or inconsistent under numpyro.substitute.
    ref_params = dict(truth_params)
    _case_cfg = configparser.ConfigParser()
    _case_cfg.optionxform = str  # preserve parameter-name case (e.g. Omh2)
    _case_cfg.read(args.config)
    _hp_u_names = {f"u_{n}" for n in h_pivot_gammas} if h_pivot else set()
    if _case_cfg.has_section("ref_params"):
        for name, val in _case_cfg["ref_params"].items():
            if name not in prior and name not in _hp_u_names:
                raise ValueError(
                    f"[ref_params] {name}: not a parameter of {prior_file_name}")
            if name in prior and isinstance(prior[name], float):
                raise ValueError(
                    f"[ref_params] {name}: fixed to {prior[name]} in "
                    f"{prior_file_name}, not a sampled site; remove it")
            ref_params[name] = jnp.asarray(float(val))
        if h_pivot:
            # user-friendly: physical mp_low/mpisn_ref lines are remapped to
            # the u-sites; explicit u_mp_low/u_mpisn_ref lines pass through
            ref_params = _h_pivot_remap(ref_params, "[ref_params]")
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
        # Guarantee float64 event arrays: jnp.asarray preserves a float32
        # input dtype even under jax_enable_x64, so upcast on the numpy side
        # (the production HDF5 files are float64 already; this covers any
        # float32-stored mocks).  Selection columns go through .to_list()
        # (Python floats), which jnp.asarray turns into float64 under x64.
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

    # scatter_free_tables: only pass it when set (by an explicit ini key or by
    # x64 = true) so the default float32 path is bit-identical to before.
    sft_kwargs = ({} if scatter_free_tables is None
                  else dict(scatter_free_tables=scatter_free_tables))

    # per-event log likelihoods at init point and usbtract insid event sum
    # same for seelction sum (keep near 0, where fload32 is finer)
    # R and the log_mu_sel are corrected back to absolute
    recenter_kwargs = {}
    baselines = None
    if hasattr(intensity_models, "recentering_baselines"):
        baselines = intensity_models.recentering_baselines(
            model_args, ref_params, use_low_bump=use_low_bump,
            smooth_tail_edge=smooth_tail_edge, tail_anchor=tail_anchor,
            pairing=pairing, **hp_kwargs, **sft_kwargs)
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
    mcmc.run(jax.random.PRNGKey(random_seed), *model_args, use_low_bump=use_low_bump, smooth_tail_edge=smooth_tail_edge,
        tail_anchor=tail_anchor, pairing=pairing, **hp_kwargs, **sft_kwargs,
        **recenter_kwargs, extra_fields=("potential_energy", "energy", "num_steps", "accept_prob","adapt_state.step_size"))
    samples = az.from_numpyro(mcmc, num_chains=nchain)
    # Model-family stamp for downstream scripts (plot_ppd FAMILIES): an absent
    # attr already means the original PISN model, so "pisn" is stamped only
    # for uniformity and "pisn_lvkpair" marks the q^beta-paired variant.
    samples.posterior.attrs["mass_model"] = ("pisn" if pairing == "mt"
                                             else "pisn_lvkpair")
    
    # centered run lp/energy are shifted - record offset to recover later
    if baselines is not None:
        samples.posterior.attrs["recentering_offset"] = baselines["offset"]
        samples.posterior.attrs["log_pdraw_sel_scale"] = baselines["log_pdraw_sel_scale"]
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

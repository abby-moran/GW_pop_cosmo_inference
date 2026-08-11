ndevice = 4
import os
import numpyro
numpyro.set_host_device_count(ndevice)
import astropy.units as u
import sys
import jax
sys.path.append('../src/')
# Use the optimized module so the generated population matches the model
# sampled in run_inf.py; re-import the original `intensity_models` to
# reproduce the old behavior.
import intensity_models_fast as intensity_models
import numpy as np
import os.path as op
import pandas as pd
from weighting import get_pop_params
from utils import warn_if_bump_too_broad
import mock_observations
from scipy.stats import norm, truncnorm
import argparse
from functools import partial
import jax.scipy as jss
import shutil

import jax.numpy as jnp
from tqdm import tqdm
from inspect import getfullargspec
from scipy.interpolate import RegularGridInterpolator
import weighting
from weighting import dm1sz_dm1ddl
import configparser
import ast
from pathlib import Path
import tempfile
import h5py


#  read .ini file
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to run config file")
args = parser.parse_args()

cfg = configparser.ConfigParser()
cfg.read(args.config)

run = cfg["run"]

# helper to safely parse values 
def parse(val):
    try:
        return ast.literal_eval(val)
    except:
        return val

# load variables from ini
base_runs_dir = "../runs"
run_name = cfg["run"]["run_dir"]
run_dir = os.path.join(base_runs_dir, f"{run_name}")

inj_file = os.path.join(run_dir, cfg["run"]["output_file_inj"])
detection_threshold = run.getint("final_snr_threshold")
snr_grid_file = run.get("snr_grid")
grid=np.load(snr_grid_file)
nsamples = run.getint("n_PE")

obs_file = os.path.join(run_dir, cfg["run"]["obs_file"])
pe_file = os.path.join(run_dir, cfg["run"]["output_file_PE"])
sel_file = os.path.join(run_dir, cfg["run"]["output_sel_file"])
ndet =run.getint("ndet")
jitter = run.getboolean("jitter_SNR")

write_obs = run.getboolean("write_obs")
new_sel = run.getboolean("new_sel")
base_dir = Path("pop_configs")

pop_file=run.get("pop_config_file")
pop_config_file = base_dir / cfg["run"]["pop_config_file"]
pop_config_file = pop_config_file.resolve()

use_low_bump=run.getboolean("use_low_bump", fallback=True)
m_min=run.getfloat("m_min", fallback=0)
delta_m_sel = run.getfloat("delta_m_sel", fallback=1.0)
    
def get_mock_obs(df, out_file, cosmo, rho_fun, detection_threshold=8, 
                 jitter_SNR=True, ndet=1, append_tf=False, evt_offset=0,
                 detection_rng=None, mc_scale=None, q_scale=None, th_scale=None, m_min=5, delta_m=1):
    """
    Vectorized version: takes in an event dataframe df and generates a file which will
    store the values we'd actually observe from the events.
    """
    if detection_rng is None:
        detection_rng = np.random.default_rng()

    noise_rng = np.random.default_rng()

    #Add a little test to make sure cosmology and ndet are the same from injections to observations
    points_test = np.column_stack([df['m1d'].to_numpy(), df['q'].to_numpy()])[:30]
    test_rho0 = (np.exp(rho_fun(points_test)) * df['Theta'].to_numpy()[:30] * np.sqrt(ndet))* (1 / df['dL'].to_numpy()[:30])
    compare_points = df['SNR'].to_numpy()[:30]

    if not np.allclose(compare_points, test_rho0, rtol=1e-4):
        print('injections:', compare_points)
        print('observations:', test_rho0)
        raise RuntimeError("SNR calculations use either wrong cosmo, SNR interpolator, or ndet")


    if jitter_SNR:
        a_rho = (0.0 - df['SNR']) / np.sqrt(ndet)
        df['SNR_OBS'] = truncnorm.rvs(a_rho, np.inf, loc=df['SNR'], scale=np.sqrt(ndet), random_state=noise_rng)
    else:
        df['SNR_OBS'] = df['SNR']

    snr_mask = df['SNR_OBS'] > detection_threshold
    inj_det = df[snr_mask].copy()
    inj_det['mc'] = inj_det['m1'] * (inj_det['q']**(3/5) / ((1 + inj_det['q'])**(1/5)))
    inj_det['dl'] = cosmo.dL(np.array(inj_det['z'].tolist()))
    inj_det['mc_det'] = inj_det['mc'] * (1 + inj_det['z'])

    n = len(inj_det)
    if n == 0:
        inj_det['log_mc_obs'] = []
        inj_det['sigma_log_mc'] = []
        inj_det['q_obs'] = []
        inj_det['sigma_q'] = []
        inj_det['theta_obs'] = []
        inj_det['sigma_theta'] = []
        inj_det = inj_det.reset_index(drop=True)
        inj_det['evt'] = []
        inj_det.to_hdf(out_file, key="observations", mode="w" if not append_tf else "a", append=append_tf,
                format="table", min_itemsize={"evt": 20},)
        return pd.Index([]), np.array([])

    rho_obs = inj_det['SNR_OBS'].to_numpy()
    q_true = inj_det['q'].to_numpy()
    theta_true = inj_det['Theta'].to_numpy()
    mc_det = inj_det['mc_det'].to_numpy()

    uncert = mock_observations.Uncertainties.from_snr(rho_obs, mc_scale=mc_scale, q_scale=q_scale, th_scale=th_scale)
    slmc = np.asarray(uncert.sigma_log_mc)
    sq   = np.asarray(uncert.sigma_q)
    sth  = np.asarray(uncert.sigma_theta)

    # --- log_mc_obs ---
    log_mc_obs = norm.rvs(loc=np.log(mc_det), scale=slmc, random_state=noise_rng)
    mc_obs = np.exp(log_mc_obs)

    # --- q_obs (vectorized truncnorm) ---
    a_q = (0.0 - q_true) / sq
    b_q = (1.0 - q_true) / sq
    q_obs = truncnorm.rvs(a_q, b_q, loc=q_true, scale=sq, random_state=noise_rng)

    # --- theta_obs (vectorized truncnorm) ---
    a_th = (0.0 - theta_true) / sth
    b_th = (1.0 - theta_true) / sth
    theta_obs = truncnorm.rvs(a_th, b_th, loc=theta_true, scale=sth, random_state=noise_rng)

    # --- derived quantities ---
    m1_det = weighting.get_m1(mc_obs, q_obs)
    z_obs = weighting.get_z_obs_true(m1_det, q_obs, theta_obs, rho_obs, rho_fun, cosmo, ndet=ndet)
    m1_src = m1_det / (1 + z_obs)
    m2_src = m1_src * q_obs

    log_p_keep = (
        intensity_models.mmin_log_smooth_turnon(m1_src, delta_m, m_min)
        + intensity_models.mmin_log_smooth_turnon(m2_src, delta_m, m_min)
    )
    p_keep = np.asarray(jnp.exp(log_p_keep))

    # --- mass-cut acceptance (vectorized) ---
    u = noise_rng.random(n)
    keep_mask = u <= p_keep

    inj_det = inj_det[keep_mask].copy()
    inj_det['log_mc_obs']   = log_mc_obs[keep_mask]
    inj_det['sigma_log_mc'] = slmc[keep_mask]
    inj_det['q_obs']        = q_obs[keep_mask]
    inj_det['sigma_q']      = sq[keep_mask]
    inj_det['theta_obs']    = theta_obs[keep_mask]
    inj_det['sigma_theta']  = sth[keep_mask]
    #inj_det['pdraw_sel'] = inj_det['pdraw_sel'] * p_keep[keep_mask]

    detected_indices = inj_det.index

    inj_det = inj_det.reset_index(drop=True)
    inj_det['evt'] = [f'evt_{i+evt_offset:06d}' for i in inj_det.index]
    inj_det.to_hdf(out_file, key='observations',
                   mode='w' if not append_tf else 'a',
                   append=append_tf, format='table', min_itemsize={"evt": 20},)
    return detected_indices, np.array(inj_det['evt'].tolist())

def jax_interp_log_snr(m1_grid, q_grid, log_snr_grid, m1s, qs):
    ix = jnp.clip(jnp.searchsorted(m1_grid, m1s) - 1, 0, m1_grid.shape[0] - 2)
    iy = jnp.clip(jnp.searchsorted(q_grid,  qs)  - 1, 0, q_grid.shape[0]  - 2)
    tx = (m1s - m1_grid[ix]) / (m1_grid[ix+1] - m1_grid[ix])
    ty = (qs  - q_grid[iy])  / (q_grid[iy+1]  - q_grid[iy])
    return (log_snr_grid[ix,   iy  ] * (1-tx)*(1-ty)
          + log_snr_grid[ix+1, iy  ] * tx    *(1-ty)
          + log_snr_grid[ix,   iy+1] * (1-tx)* ty
          + log_snr_grid[ix+1, iy+1] * tx    * ty)

def _sample_truncnorm(key, loc, scale, a, b, shape):
    """Truncated normal via inverse CDF — JAX-native, vmappable."""
    u = jax.random.uniform(key, shape)
    cdf_a = jss.stats.norm.cdf(a)
    cdf_b = jss.stats.norm.cdf(b)
    return jss.stats.norm.ppf(cdf_a + u * (cdf_b - cdf_a)) * scale + loc

def draw_samples_single_event_jax(key, log_mc_obs, sigma_log_mc, q_obs, sigma_q,
                                   theta_obs, sigma_theta, rho_obs,
                                   m1_grid, q_grid, log_snr_grid,
                                   size_final, ndet, dl_fid=1.0, theta_fid=1.0, jitter=True):
    size  = 10 * size_final
    if jitter==False:
        scale=0
    else:
        scale = jnp.sqrt(jnp.maximum(ndet, 1.0))
    k = jax.random.split(key, 8)

    # ── q ────────────────────────────────────────────────────────────────
    a_q, b_q = -q_obs / sigma_q, (1 - q_obs) / sigma_q
    qs_raw   = _sample_truncnorm(k[0], q_obs, sigma_q, a_q, b_q, (2*size,))
    w_q = ((jss.stats.norm.cdf(b_q) - jss.stats.norm.cdf(a_q)) /
           (jss.stats.norm.cdf((1-qs_raw)/sigma_q) - jss.stats.norm.cdf(-qs_raw/sigma_q)))
    w_q = w_q / w_q.sum()
    cdf_q = jnp.cumsum(w_q)
    u_q = jax.random.uniform(k[1], (size,))
    idx_q = jnp.searchsorted(cdf_q, u_q)
    qs = qs_raw[idx_q]

    # ── log_mc ───────────────────────────────────────────────────────────
    log_mcs = jax.random.normal(k[2], (size,)) * sigma_log_mc + log_mc_obs
    mcs = jnp.exp(log_mcs)
    m1s = mcs / (qs**(3/5) / (1 + qs)**(1/5))

    # ── theta ────────────────────────────────────────────────────────────
    a_th, b_th = -theta_obs / sigma_theta, (1 - theta_obs) / sigma_theta
    thetas_raw = _sample_truncnorm(k[3], theta_obs, sigma_theta, a_th, b_th, (2*size,))
    w_th = ((jss.stats.norm.cdf(b_th) - jss.stats.norm.cdf(a_th)) /
            (jss.stats.norm.cdf((1-thetas_raw)/sigma_theta) - jss.stats.norm.cdf(-thetas_raw/sigma_theta)))
    w_th = w_th / w_th.sum()
    cdf_th = jnp.cumsum(w_th)
    u_th = jax.random.uniform(k[4], (size,))
    idx_th = jnp.searchsorted(cdf_th, u_th)
    thetas = thetas_raw[idx_th]

    # ── rho ──────────────────────────────────────────────────────────────
    a_rho = -rho_obs / scale
    rhos_raw = _sample_truncnorm(k[5], rho_obs, scale, a_rho, jnp.array(30.0), (2*size,))
    w_rho = jss.stats.norm.cdf(rho_obs/scale) / jss.stats.norm.cdf(rhos_raw/scale)
    w_rho = w_rho / w_rho.sum()
    cdf_rho = jnp.cumsum(w_rho)
    u_rho = jax.random.uniform(k[6], (size,))
    idx_rho = jnp.searchsorted(cdf_rho, u_rho)
    rhos  = rhos_raw[idx_rho]

    # ── dL + final reweight ───────────────────────────────────────────────
    snr_fid  = jnp.exp(jax_interp_log_snr(m1_grid, q_grid, log_snr_grid, m1s, qs))
    dls      = dl_fid * thetas / theta_fid * snr_fid * jnp.sqrt(jnp.maximum(ndet, 1.0)) / rhos
    reweight = dls / rhos * m1s * jss.stats.beta.pdf(thetas, 2, 4)
    reweight = jnp.nan_to_num(reweight, nan=0.0, posinf=0.0)
    reweight = reweight / reweight.sum()
    cdf_rw = jnp.cumsum(reweight)
    u_rw = jax.random.uniform(k[7], (size_final,))
    idx = jnp.searchsorted(cdf_rw, u_rw)
    return m1s[idx], qs[idx], dls[idx]

@partial(jax.jit, static_argnums=(11, 12, 13))
def batched_draw(keys, log_mc_obs, sigma_log_mc, q_obs, sigma_q,
                 theta_obs, sigma_theta, rho_obs,
                 m1_grid, q_grid, log_snr_grid, size_final, ndet, jitter=True):
    return jax.vmap(
        lambda key, lmc, slmc, q, sq, th, sth, rho:
            draw_samples_single_event_jax(key, lmc, slmc, q, sq, th, sth, rho,
                                          m1_grid, q_grid, log_snr_grid, size_final, ndet, jitter=jitter)
    )(keys, log_mc_obs, sigma_log_mc, q_obs, sigma_q, theta_obs, sigma_theta, rho_obs)

@jax.jit
def compute_log_w(m1, q, z, pdraw_cosmo):
    log_dN_vals = log_dN_func(m1, q, z)
    return (log_dN_vals
            + jnp.log(cosmo.dVCdz(z))
            - 2 * jnp.log1p(z)
            - jnp.log(pdraw_cosmo)
            - jnp.log(cosmo.ddL_dz(z)))

if __name__ == "__main__":
    rng = np.random.default_rng(251286134409181405721219170031242732711)

    chunk_size = run.getint("chunk_size", fallback=int(2e6))  # memory limit per read
    num_tot    = run.getint("num_tot",    fallback=int(5e7))   # target/cap on population draws
    n_total    = run.getint("n_total",    fallback=int(8e7))   # total injections to consider
    # "rejection" (default): accept each injection once with probability
    #   w/w_max -- yields exact i.i.d. draws from the population with no
    #   duplicates, at most ~Z draws (the pool's effective size); num_tot only
    #   caps the yield.  "multinomial": the old resampling-with-replacement,
    #   which silently replicates pool entries when num_tot exceeds the pool's
    #   effective size (a 12.6M-draw pool supports only ~2e5 draws; asking for
    #   2e7 gave events that were 55-fold copies of single injections and made
    #   hyperposteriors ~3.6x overconfident in truth-recovery tests).
    sampling_method = run.get("sampling_method", fallback="rejection").strip().lower()
    if sampling_method not in ("rejection", "multinomial"):
        raise ValueError(f"unknown sampling_method: {sampling_method!r}")
    
    with pd.HDFStore(inj_file, mode='r') as store:
        n_rows = store.get_storer('true_parameters').nrows
    if n_total > n_rows:
        print(f"  Warning: n_total={n_total:,} exceeds file size {n_rows:,}, clamping.")
        n_total = n_rows
    elif n_total < n_rows:
        # Leaving rows unread is safe with rejection sampling (no silent
        # duplicates), but it still wastes pool size that could raise the
        # acceptance yield -- the usual reason a production .ini with
        # n_total=3e7 under-uses a num_loops=150 injection file.
        print(f"  Warning: n_total={n_total:,} uses only {100 * n_total / n_rows:.1f}% "
              f"of the injection file ({n_rows:,} rows); "
              f"{n_rows - n_total:,} draws will be left unread.  "
              f"Raise n_total (and chunk_size) to use the full pool.")
    population_parameters, cosmo = get_pop_params(pop_config_file)
    if use_low_bump:
        warn_if_bump_too_broad(population_parameters.get("msigma_low"),
                               context=f"true population, {pop_file}")

    m1_grid  = grid["m1_grid"]
    q_grid   = grid["q_grid"]
    snr_grid = grid["snr_grid"]
    dL_fid   = float(grid["dL_fid"])
    log_snr_interp = RegularGridInterpolator(
        (m1_grid, q_grid), np.log(snr_grid), bounds_error=False, fill_value=-np.inf)

    log_dN_obj  = intensity_models.LogDNDMDQDV
    pop_params  = {key: population_parameters[key]
                   for key in getfullargspec(log_dN_obj)[0][1:]
                   if key in population_parameters.keys()}
    # smooth_tail_edge=True matches the run_inf.py / pop_cosmo_model default;
    # set smooth_tail_edge=False for the old hard-edged behavior.
    log_dN_func = intensity_models.build_population_model(pop_params, use_low_bump=use_low_bump,
                                                          smooth_tail_edge=True)

    if write_obs:
        print("Pass 1: computing weights, caching to disk...")
        log_w_max = -np.inf
        cache_dir = tempfile.mkdtemp(prefix="logw_cache_")
        chunk_files = []

        for start in tqdm(range(0, n_total, chunk_size)):
            chunk = pd.read_hdf(inj_file, key='true_parameters',
                                start=start, stop=start + chunk_size)
            chunk['dm1sz_dm1ddl2'] = dm1sz_dm1ddl(chunk['z'].to_numpy(), cosmology=cosmo)
            chunk['pdraw_cosmo']   = chunk['pdraw_mqz'] * chunk['dm1sz_dm1ddl2']
            #log_dN_vals = log_dN_func(chunk['m1'].values, chunk['q'].values, chunk['z'].values)
            log_w = compute_log_w(
                jnp.asarray(chunk['m1'].values),
                jnp.asarray(chunk['q'].values),
                jnp.asarray(chunk['z'].values),
                jnp.asarray(chunk['pdraw_cosmo'].values),)
            log_w = np.asarray(log_w)

            fpath = op.join(cache_dir, f"logw_{start}.npy")
            np.save(fpath, log_w)
            chunk_files.append(fpath)

            chunk_max = float(np.nanmax(log_w))
            if chunk_max > log_w_max:
                log_w_max = chunk_max

        print(f"  log_w_max = {log_w_max:.4f}")

        print("Pass 2: computing global normalizer (from cache, no recompute)...")
        Z = 0.0
        for fpath in chunk_files:
            log_w = np.load(fpath)
            Z += float(np.nansum(np.exp(log_w - log_w_max)))

        print(f"  Z = {Z:.4e}")

        if sampling_method == "rejection":
            # Decide every acceptance up front (cheap: only the cached log_w
            # is touched) so the total number of population draws -- the
            # 'ndraw' the selection-function estimate is normalized by -- is
            # known before any chunk is written.
            print("Pass 2b: rejection sampling from the cached weights...")
            kept_idx = []
            for fpath in chunk_files:
                log_w = np.load(fpath).astype(np.float64)
                p_acc = np.nan_to_num(np.exp(log_w - log_w_max), nan=0.0)
                kept_idx.append(np.nonzero(rng.random(p_acc.size) < p_acc)[0])
            n_pop = int(sum(k.size for k in kept_idx))
            print(f"  accepted {n_pop:,} of {n_total:,} pool draws "
                  f"(efficiency {n_pop / n_total:.3%}) -- exact i.i.d. "
                  f"population draws, no duplicates")
            if n_pop > num_tot:
                frac = num_tot / n_pop
                kept_idx = [k[rng.random(k.size) < frac] for k in kept_idx]
                n_pop = int(sum(k.size for k in kept_idx))
                print(f"  thinned to {n_pop:,} (num_tot = {num_tot:,})")
            elif n_pop < num_tot:
                print(f"  WARNING: pool supports only {n_pop:,} population draws "
                      f"but num_tot = {num_tot:,} were requested.\n"
                      f"           Proceeding with {n_pop:,}; to reach num_tot, "
                      f"grow the injection file / n_total to ~"
                      f"{int(1.2 * num_tot * n_total / max(n_pop, 1)):,} draws.")
        else:
            n_pop = num_tot

        # With pdraw_sel properly normalized (see below), the rate the
        # inference should recover from this mock is exactly n_pop / C.
        log_C = np.log(Z) + log_w_max - np.log(n_total)
        print(f"  population normalization log_C = {log_C:.4f}; "
              f"true rate of this mock dataset: R = n_pop/C = {n_pop * np.exp(-log_C):.4f}")

        print("Pass 3: sampling and processing...")
        first_chunk = True
        evt_offset  = 0
        mc_scale = population_parameters.get('mc_scale', None)
        q_scale  = population_parameters.get('q_scale',  None)
        th_scale = population_parameters.get('th_scale', None)

        for idx, start in enumerate(tqdm(range(0, n_total, chunk_size))):
            chunk = pd.read_hdf(inj_file, key='true_parameters',
                                start=start, stop=start + chunk_size)
            chunk['dm1sz_dm1ddl2'] = dm1sz_dm1ddl(chunk['z'].to_numpy(), cosmology=cosmo)
            chunk['pdraw_cosmo']   = chunk['pdraw_mqz'] * chunk['dm1sz_dm1ddl2']

            # load cached log_w instead of recomputing log_dN_func, dVCdz, ddL_dz, etc.
            log_w = np.load(chunk_files[idx]).astype(np.float64)

            w_chunk  = np.nan_to_num(np.exp(log_w - log_w_max), nan=0.0)

            if sampling_method == "rejection":
                chunk_sampled = kept_idx[idx]
            else:
                p_chunk  = w_chunk / Z
                n_chunk  = int(np.round(p_chunk.sum() * num_tot))
                chunk_sampled = np.sort(np.random.choice(len(chunk), p=p_chunk / p_chunk.sum(), size=n_chunk))

            df_det_chunk = chunk.iloc[chunk_sampled].copy()
            # The draws follow the *normalized* population density
            #   p(theta) = pop(theta) / C,  C = int pop ~ Z*exp(log_w_max)/n_total,
            # and the selection estimator mu_sel = (1/ndraw) sum pop(Lambda)/pdraw_sel
            # needs pdraw_sel to be exactly that normalized density.
            # w*pdraw_cosmo alone is pop*exp(-log_w_max); the n_total/Z factor
            # completes the normalization.  (Before this fix the recovered rate
            # R = nobs/mu_sel was deflated by the constant Z/n_total; population
            # -shape posteriors were unaffected.)
            df_det_chunk['pdraw_sel'] = (w_chunk[chunk_sampled]
                                        * chunk['pdraw_cosmo'].values[chunk_sampled]
                                        * (n_total / Z))
            df_det_chunk['dl']    = cosmo.dL(df_det_chunk['z'].to_numpy())
            df_det_chunk['m1d']   = df_det_chunk['m1'] * (1 + df_det_chunk['z'])
            df_det_chunk['ndraw'] = n_pop
            df_det_chunk = df_det_chunk.reset_index(drop=True)

            detected_indices, evt_names = get_mock_obs(
                df_det_chunk, obs_file, cosmo, log_snr_interp, ndet=ndet,
                jitter_SNR=jitter, detection_threshold=detection_threshold,
                append_tf=not first_chunk, evt_offset=evt_offset,
                mc_scale=mc_scale, q_scale=q_scale, th_scale=th_scale, m_min=m_min, delta_m=delta_m_sel
            )

            det_mask  = df_det_chunk.index.isin(detected_indices)
            sel_chunk = df_det_chunk[det_mask].copy()
            sel_chunk['ndraw'] = n_pop
            sel_chunk['evt']   = evt_names
            sel_chunk.to_hdf(sel_file, key='true_parameters',
                            mode='w' if first_chunk else 'a',
                            append=not first_chunk, format='table', min_itemsize={"evt": 20})

            evt_offset += det_mask.sum()
            first_chunk  = False

        R_all = n_pop * np.exp(-log_C)
        print(f"  total detections: {evt_offset:,} of {n_pop:,} population draws")
        print(f"  true rate: R = {R_all:.4f} if ALL {evt_offset:,} detections are analyzed; "
              f"an analysis of the first nobs detections should recover "
              f"R = {R_all:.4f} * nobs/{evt_offset:,} "
              f"(e.g. nobs=9000 -> R = {R_all * 9000 / max(int(evt_offset), 1):.4f})")

        # cleanup cache
        shutil.rmtree(cache_dir)

    else:
        print("Using old obs file")

    print("Generating mock PE...")
    pe_samples_full = pd.read_hdf(obs_file, 'observations', start=0, stop =10000)
    #pe_samples_full=pe_samples_full[0:10000]
    evt_df = pe_samples_full.drop_duplicates('evt').sort_values('evt')
    log_mc_obs_arr  = jnp.array(evt_df['log_mc_obs'].values)
    sigma_log_mc_arr = jnp.array(evt_df['sigma_log_mc'].values)
    q_obs_arr       = jnp.array(evt_df['q_obs'].values)
    sigma_q_arr     = jnp.array(evt_df['sigma_q'].values)
    theta_obs_arr   = jnp.array(evt_df['theta_obs'].values)
    sigma_theta_arr = jnp.array(evt_df['sigma_theta'].values)
    rho_obs_arr     = jnp.array(evt_df['SNR_OBS'].values)

    n_events = len(evt_df)
    keys = jax.random.split(jax.random.PRNGKey(0), n_events)

    m1_grid_jax    = jnp.array(m1_grid)
    q_grid_jax     = jnp.array(q_grid)
    log_snr_grid_jax = jnp.array(np.log(snr_grid))

    batch_size = 32

    all_m1s = []
    all_qs = []
    all_dls = []

    for i in range(0, n_events, batch_size):

        sl = slice(i, i + batch_size)

        out = batched_draw(
            keys[sl],
            log_mc_obs_arr[sl],
            sigma_log_mc_arr[sl],
            q_obs_arr[sl],
            sigma_q_arr[sl],
            theta_obs_arr[sl],
            sigma_theta_arr[sl],
            rho_obs_arr[sl],
            m1_grid_jax,
            q_grid_jax,
            log_snr_grid_jax,
            nsamples,
            ndet,
            jitter=jitter
        )

        all_m1s.append(np.array(out[0]))
        all_qs.append(np.array(out[1]))
        all_dls.append(np.array(out[2]))

    m1s = np.concatenate(all_m1s, axis=0)
    qs  = np.concatenate(all_qs, axis=0)
    dls = np.concatenate(all_dls, axis=0)
    pdraws=np.zeros_like(m1s)

    df_samples = pd.DataFrame({'m1': list(m1s),       # each element is an array of nsamples
                                    'q': list(qs), 'dl': list(dls), 'pdraw': list(pdraws)})
    with h5py.File(pe_file, "w") as f:
        f["m1"] = m1s
        f["q"] = qs
        f["dl"] = dls
        f["pdraw"] = pdraws

    print("array shapes (we want nevents, nsamples): ",
        m1s.shape, qs.shape, dls.shape)
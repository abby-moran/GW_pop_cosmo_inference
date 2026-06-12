ndevice = 4
import os
import numpyro
numpyro.set_host_device_count(ndevice)
import astropy.units as u
import sys
import jax
sys.path.append('../src/')
import intensity_models
import numpy as np
import os.path as op
import pandas as pd
from weighting import get_pop_params
import mock_observations
from scipy.stats import norm, truncnorm
import argparse
from functools import partial
import jax.scipy as jss


import jax.numpy as jnp
from tqdm import tqdm
from inspect import getfullargspec
from scipy.interpolate import RegularGridInterpolator
import weighting
from weighting import dm1sz_dm1ddl
import configparser
import ast
from pathlib import Path

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

m_min=run.getint("m_min", fallback=0)
    
def get_mock_obs(df, out_file, cosmo, rho_fun, detection_threshold=8, 
                 jitter_SNR=True, ndet=1, append_tf=False, evt_offset=0,
                 detection_rng=None, mc_scale=None, q_scale=None, th_scale=None, m_min=5):
    """
    takes in an event dataframe df and generates a file which will store the values we'd actulaly observe from the events

    inputs
        df: events 
        out_file: where to store the observed values
        cosmo: cosmology model we're using, usually FlatwCDMCosmology with population paramters
        detection_threshold: what SNR we cut on, 8 by default
        jitter_SNR: False if detection is perfect, otherwise should be on
        ndet: number of mock detectors, controls how much we jitter the SNR by
    
    outputs a file with observed log chirp mass, log q, and Theta (finn-chernoff parameter) with corresponding uncertainties calculted from 
    SNR scalings
    """
    if detection_rng is None:
        detection_rng = np.random.default_rng()
    
    # unseeded rng for observation noise (different realization each run)
    noise_rng = np.random.default_rng()

    #jitter = detection_rng.normal(loc=0, scale=np.sqrt(ndet), size=len(df))    
    if jitter_SNR:
        a_rho=(0.0 - df['SNR']) / np.sqrt(ndet)
        df['SNR_OBS'] = truncnorm.rvs(a_rho, np.inf, loc=df['SNR'], scale= np.sqrt(ndet),  random_state=noise_rng)
    else:
        df['SNR_OBS'] = df['SNR'] 
    
    # Store which original df indices passed the SNR cut
    snr_mask = df['SNR_OBS'] > detection_threshold
    detected_indices = df.index[snr_mask]        
    
    inj_det = df[snr_mask].copy()
    #inj_det = df[df['SNR_OBS'] > detection_threshold].copy()
    inj_det['mc'] = inj_det['m1'] * (inj_det['q']**(3/5) / ((1 + inj_det['q'])**(1/5)))
    inj_det['dl'] = cosmo.dL(np.array(inj_det['z'].tolist()))
    inj_det['mc_det'] = inj_det['mc'] * (1 + inj_det['z'])
    log_mc_obs = []
    sigma_log_mc = []
    q_obs=[]
    sigma_q=[]
    theta_obs=[]
    sigma_theta=[]
    passed_indices = []  # track original df indices that survive mass cut
    for i, row in tqdm(inj_det.iterrows()):
        uncert = mock_observations.Uncertainties.from_snr(row['SNR_OBS'],
                                        mc_scale=mc_scale, q_scale=q_scale, th_scale=th_scale)
        slmc = uncert.sigma_log_mc
        log_mc_obs_i = norm.rvs(loc=np.log(row['mc_det']), scale=slmc, random_state=noise_rng)
        mc_obs_i = np.exp(log_mc_obs_i)

        sq = uncert.sigma_q
        a = (0.0 - row['q']) / sq
        b = (1 - row['q']) / sq
        q_obs_i = truncnorm.rvs(a, b, loc=row['q'], scale=sq, random_state=noise_rng)

        m1_det=weighting.get_m1(mc_obs_i, q_obs_i)

        sth = uncert.sigma_theta  
        a = (0.0 - row['Theta']) / sth  
        b = (1 - row['Theta']) / sth
        theta_obs_i = truncnorm.rvs(a, b, loc=row['Theta'], scale=sth, random_state=noise_rng)
        
        z_obs = weighting.get_z_obs_true(m1_det, q_obs_i, theta_obs_i, row['SNR_OBS'], rho_fun, cosmo, ndet=ndet)
        m1_src = m1_det / (1 + z_obs)
        m2_src = m1_src * q_obs_i
        if m1_src < m_min or m2_src < m_min:
            continue  # skip this event

        passed_indices.append(i)  # survived mass cut — record original index
        log_mc_obs.append(log_mc_obs_i)
        sigma_log_mc.append(slmc)
        q_obs.append(q_obs_i)
        sigma_q.append(sq)
        sigma_theta.append(sth)
        theta_obs.append(theta_obs_i)

    # Trim inj_det to only events that passed the mass cut
    inj_det = inj_det.loc[passed_indices].copy()
    detected_indices = pd.Index(passed_indices)
    
    inj_det['log_mc_obs'] = log_mc_obs
    inj_det['sigma_log_mc'] = sigma_log_mc
    inj_det['q_obs'] = q_obs
    inj_det['sigma_q'] = sigma_q
    inj_det['theta_obs'] = theta_obs
    inj_det['sigma_theta'] = sigma_theta
    
    inj_det['z'] = inj_det['z']
    inj_det = inj_det.reset_index(drop=True)
    inj_det['evt'] = [f'evt_{i+evt_offset:06d}' for i in inj_det.index]  # offset here
    inj_det.to_hdf(out_file, key='observations', 
               mode='w' if not append_tf else 'a', 
               append=append_tf, format='table')
    return detected_indices, np.array(inj_det['evt'].tolist())

def gen_mock_PE(obs_file, log_SNR_fun, population_parameters, cosmo, nsamples=200, 
                outfile=None, ndet=1, append_tf=False, new_sel=True, jitter=True):
    """
    takes a file which constains observed values, and samples assuming Gaussians and outputs that in a file. 
    Need an SNR scaling relation log_SNR_fun

    Output file has keys m1, q, dl, praw, evt. Each of shape nevents x nsamples
    """
    
    pe_samples_full = pd.read_hdf(obs_file, 'observations')
    pe_samples_full=pe_samples_full[0:10000]
    
    m1s, qs, dls, pdraws , evts= [], [], [], [], []
    mass_obs_evt, q_err, dl_=[], [], []
    for i, (n, e) in enumerate(pe_samples_full.groupby('evt')):
        # preallocate arrays for this event
        m1_event = []
        q_event = []
        dl_event = []
        pdraw_event = []
        evt=[]
        for num in range(1):
            samples = weighting.draw_mock_samples_mine(
                e['log_mc_obs'].iloc[0],  # detector frame
                e['sigma_log_mc'].iloc[0],
                #e['log_q_obs'].iloc[0], 
                #e['sigma_log_q'].iloc[0],
                e['q_obs'].iloc[0], 
                e['sigma_q'].iloc[0],
                
                e['theta_obs'].iloc[0],
                e['sigma_theta'].iloc[0],
                e['SNR_OBS'].iloc[0], 
                log_SNR_fun, cosmo, ndet=ndet, size_final=nsamples, jitter_SNR=jitter)
            
            m1_event.append(samples[0])
            q_event.append(samples[1])
            dl_event.append(samples[2])
            pdraw_event.append(samples[3])
            evt.append(e['evt'])
    
        # store all samples for this event
        m1s.append(np.array(m1_event).flatten())
        qs.append(np.array(q_event).flatten())
        dls.append(np.array(dl_event).flatten())
        pdraws.append(np.array(pdraw_event).flatten())
        evts.append(np.array(evt).flatten())
    
    # convert to arrays (shape = nevents × nsamples)
    m1s = np.array(m1s)
    qs = np.array(qs)
    dls = np.array(dls)
    pdraws = np.array(pdraws)
    evts = np.array(evts)

    if outfile is not None:
        df_samples = pd.DataFrame({'m1': list(m1s),       # each element is an array of nsamples
                                    'q': list(qs), 'dl': list(dls), 'pdraw': list(pdraws), 'evt': list(evts[:,0])})
        df_samples.to_hdf(outfile, key="samples", mode="w")

    return(m1s, qs, dls, pdraws, evts)
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

if __name__ == "__main__":
    rng = np.random.default_rng(251286134409181405721219170031242732711)

    #if ndet > 0:
    #    jitter = True
    #else:
    #    jitter = False

    # load from config 
    chunk_size = run.getint("chunk_size", fallback=int(2e6))  # memory limit per read
    num_tot    = run.getint("num_tot",    fallback=int(5e7))   # how many to reweight
    n_total    = run.getint("n_total",    fallback=int(8e7))   # total injections to consider
    
    with pd.HDFStore(inj_file, mode='r') as store:
        n_rows = store.get_storer('true_parameters').nrows   
    if n_total > n_rows:
        print(f"  Warning: n_total={n_total} exceeds file size {n_rows}, clamping.")
        n_total = n_rows
    population_parameters, cosmo = get_pop_params(pop_config_file)

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
    log_dN_func = log_dN_obj(**pop_params)

    if write_obs:
        print("Pass 1: finding global log_w_max...")
        log_w_max = -np.inf
        for start in tqdm(range(0, n_total, chunk_size)):
            chunk = pd.read_hdf(inj_file, key='true_parameters',
                                start=start, stop=start + chunk_size)
            chunk['dm1sz_dm1ddl2'] = dm1sz_dm1ddl(chunk['z'].to_numpy(), cosmology=cosmo)
            chunk['pdraw_cosmo']   = chunk['pdraw_mqz'] * chunk['dm1sz_dm1ddl2']
            log_dN_vals = log_dN_func(chunk['m1'].values, chunk['q'].values, chunk['z'].values)
            log_w = (log_dN_vals
                    + jnp.log(cosmo.dVCdz(chunk['z'].values))
                    - 2 * jnp.log1p(chunk['z'].values)
                    - jnp.log(chunk['pdraw_cosmo'].values)
                    - jnp.log(cosmo.ddL_dz(chunk['z'].values)))
            chunk_max = float(np.nanmax(log_w))
            if chunk_max > log_w_max:
                log_w_max = chunk_max

        print(f"  log_w_max = {log_w_max:.4f}")

        print("Pass 2: computing global normalizer...")
        Z = 0.0
        for start in tqdm(range(0, n_total, chunk_size)):
            chunk = pd.read_hdf(inj_file, key='true_parameters',
                                start=start, stop=start + chunk_size)
            chunk['dm1sz_dm1ddl2'] = dm1sz_dm1ddl(chunk['z'].to_numpy(), cosmology=cosmo)
            chunk['pdraw_cosmo']   = chunk['pdraw_mqz'] * chunk['dm1sz_dm1ddl2']
            log_dN_vals = log_dN_func(chunk['m1'].values, chunk['q'].values, chunk['z'].values)
            log_w = (log_dN_vals
                    + jnp.log(cosmo.dVCdz(chunk['z'].values))
                    - 2 * jnp.log1p(chunk['z'].values)
                    - jnp.log(chunk['pdraw_cosmo'].values)
                    - jnp.log(cosmo.ddL_dz(chunk['z'].values)))
            Z += float(np.nansum(np.exp(np.array(log_w) - log_w_max)))

        print(f"  Z = {Z:.4e}")

        print("Pass 3: sampling and processing...")
        first_chunk = True
        evt_offset  = 0
        mc_scale = population_parameters.get('mc_scale', None)
        q_scale  = population_parameters.get('q_scale',  None)
        th_scale = population_parameters.get('th_scale', None)

        for start in tqdm(range(0, n_total, chunk_size)):
            chunk = pd.read_hdf(inj_file, key='true_parameters',
                                start=start, stop=start + chunk_size)
            chunk['dm1sz_dm1ddl2'] = dm1sz_dm1ddl(chunk['z'].to_numpy(), cosmology=cosmo)
            chunk['pdraw_cosmo']   = chunk['pdraw_mqz'] * chunk['dm1sz_dm1ddl2']
            log_dN_vals = log_dN_func(chunk['m1'].values, chunk['q'].values, chunk['z'].values)
            log_w = (log_dN_vals
                    + jnp.log(cosmo.dVCdz(chunk['z'].values))
                    - 2 * jnp.log1p(chunk['z'].values)
                    - jnp.log(chunk['pdraw_cosmo'].values)
                    - jnp.log(cosmo.ddL_dz(chunk['z'].values)))

            # each chunk gets a globally-consistent fraction of the total samples
            w_chunk  = np.nan_to_num(np.exp(np.array(log_w) - log_w_max), nan=0.0)
            p_chunk  = w_chunk / Z                          # probabilities within this chunk, sum across all chunks = 1
            n_chunk  = int(np.round(p_chunk.sum() * num_tot))  # expected samples from this chunk
            #if n_chunk == 0:
            #    continue

            chunk_sampled = np.sort(
                np.random.choice(len(chunk), p=p_chunk / p_chunk.sum(), size=n_chunk)
            )  # sample within chunk using local renorm — valid because n_chunk already encodes the global share

            df_det_chunk = chunk.iloc[chunk_sampled].copy()
            df_det_chunk['pdraw_sel'] = (w_chunk[chunk_sampled]
                                        * chunk['pdraw_cosmo'].values[chunk_sampled])
            df_det_chunk['dl']    = cosmo.dL(df_det_chunk['z'].to_numpy())
            df_det_chunk['m1d']   = df_det_chunk['m1'] * (1 + df_det_chunk['z'])
            df_det_chunk['ndraw'] = num_tot
            df_det_chunk = df_det_chunk.reset_index(drop=True)

            detected_indices, evt_names = get_mock_obs(
                df_det_chunk, obs_file, cosmo, log_snr_interp, ndet=ndet,
                jitter_SNR=jitter, detection_threshold=detection_threshold,
                append_tf=not first_chunk, evt_offset=evt_offset,
                mc_scale=mc_scale, q_scale=q_scale, th_scale=th_scale, m_min=m_min
            )

            det_mask  = df_det_chunk.index.isin(detected_indices)
            sel_chunk = df_det_chunk[det_mask].copy()
            sel_chunk['ndraw'] = num_tot
            sel_chunk['evt']   = evt_names
            sel_chunk.to_hdf(sel_file, key='true_parameters',
                            mode='w' if first_chunk else 'a',
                            append=not first_chunk, format='table')

            evt_offset += det_mask.sum()
            first_chunk  = False

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
    df_samples.to_hdf(pe_file, key="samples", mode="w")

    print("array shapes (we want nevents, nsamples): ",
        m1s.shape, qs.shape, dls.shape)
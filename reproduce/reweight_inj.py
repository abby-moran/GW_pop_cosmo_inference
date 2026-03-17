ndevice = 4
import os
import numpyro
numpyro.set_host_device_count(ndevice)
from astropy.cosmology import Planck18
import astropy.units as u
import sys
import jax
sys.path.append('../src/')
import intensity_models
import numpy as np
import os.path as op
import pandas as pd
import paths
from utils import get_priors_from_file
from intensity_models import coords
import mock_observations
from scipy.stats import norm, truncnorm

import jax.numpy as jnp
from tqdm import tqdm
from scipy.special import logsumexp
from inspect import getfullargspec
from scipy.interpolate import RegularGridInterpolator
import weighting



def dm1sz_dm1ddl(z, cosmology=None):
    if not cosmology:
        #return (1+z) / (Planck18.comoving_distance(z).to(u.Gpc).value + (1+z)*Planck18.hubble_distance.to(u.Gpc).value / Planck18.efunc(z))
        dm1s_dm1d = (1+z)**-1
        ddl_dz = (Planck18.comoving_distance(z).to(u.Gpc).value + (1 + z) * Planck18.hubble_distance.to(u.Gpc).value / Planck18.efunc(z))
        return dm1s_dm1d * (ddl_dz)**-1
    else:
        dm1s_dm1d = (1+z)**-1
        ddl_dz = cosmology.ddL_dz((z))
        return  dm1s_dm1d * (ddl_dz)**-1

def get_pop_params(config_file):
    population_parameters = dict()
    population_parameters = dict()
    with open(config_file) as param_file:
        for line in param_file:
            (key, val) = line.split('=')
            population_parameters[key.strip()] = val.strip()
            try:
                population_parameters[key.strip()] = float(val.strip())
            except ValueError:
                pass
    cosmo = intensity_models.FlatwCDMCosmology(population_parameters['h'], population_parameters['Om'],
                                           population_parameters['w'], population_parameters['zmax'])

    return population_parameters, cosmo


def get_mock_obs(df, out_file, cosmo, mult=1, detection_threshold=8, 
                 jitter_SNR=True, ndet=1, append_tf=False, evt_offset=0,
                 detection_rng=None):
    """
    takes in an event dataframe df and generates a file which will store the values we'd actulaly observe from the events

    inputs
        df: events 
        out_file: where to store the observed values
        cosmo: cosmology model we're using, usually FlatwCDMCosmology with population paramters
        mult: multiplier on the paramter uncertatintes, so higher numbers scale all the uncertainties up (except SNR)
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

    jitter = detection_rng.normal(loc=0, scale=np.sqrt(ndet), size=len(df))
    
    if jitter_SNR:
        df['SNR_OBS'] = df['SNR'] + jitter
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
    log_q_obs=[]
    sigma_log_q=[]
    theta_obs=[]
    sigma_theta=[]
    for i, row in tqdm(inj_det.iterrows()):
        uncert = mock_observations.Uncertainties.from_snr(row['SNR_OBS'])
        log_mc_obs.append(np.log(row['mc_det']) + mult*uncert.sigma_log_mc*noise_rng.normal())
        sigma_log_mc.append(mult*uncert.sigma_log_mc)
        
        slq =mult*uncert.sigma_q / row['q']   # error propagation
        b = (0.0 - np.log(row['q'])) / slq
        log_q_obs.append(truncnorm.rvs(-np.inf, b, loc=np.log(row['q']), scale=slq,  random_state=noise_rng))
        sigma_log_q.append(slq)

        sigma_theta.append(mult*uncert.sigma_theta)
        # Θ_obs ~ N[0, 0.25](Θ_true, σΘ),
        # modeled after DOI 10.3847/2041-8213/ab77c9
        a = (0.0 - row['Theta']) / sigma_theta[-1]
        b = (1 - row['Theta']) / sigma_theta[-1]
        theta_obs.append(truncnorm.rvs(a, b, loc=row['Theta'], scale= sigma_theta[-1],  random_state=noise_rng))
    
    inj_det['log_mc_obs'] = log_mc_obs
    inj_det['sigma_log_mc'] = sigma_log_mc
    
    inj_det['log_q_obs'] = log_q_obs
    inj_det['sigma_log_q'] = sigma_log_q
    inj_det['theta_obs'] = theta_obs
    inj_det['sigma_theta'] = sigma_theta
    
    inj_det['z'] = inj_det['z']
    inj_det = inj_det.reset_index(drop=True)
    inj_det['evt'] = [f'evt_{i+evt_offset:06d}' for i in inj_det.index]  # offset here
    inj_det.to_hdf(out_file, key='observations', 
               mode='w' if not append_tf else 'a', 
               append=append_tf, format='table')
    return detected_indices, np.array(inj_det['evt'].tolist())

def gen_mock_PE(obs_file, log_SNR_fun, population_parameters, cosmo, nsamples=200, outfile=None, ndet=1, append_tf=False):
    """
    takes a file which constains observed values, and samples assuming Gaussians and outputs that in a file. 
    Need an SNR scaling relation log_SNR_fun

    Output file has keys m1, q, dl, praw, evt. Each of shape nevents x nsamples
    """
    
    pe_samples_full = pd.read_hdf(obs_file, 'observations')
    pe_samples_full=pe_samples_full
    
    m1s, qs, dls, pdraws , evts= [], [], [], [], []
    mass_obs_evt, q_err, dl_=[], [], []
    for n, e in pe_samples_full.groupby('evt'):
        # preallocate arrays for this event
        m1_event = []
        q_event = []
        dl_event = []
        pdraw_event = []
        evt=[]
        for num in range(1):
            samples = weighting.draw_mock_samples(
                e['log_mc_obs'].iloc[0],  # detector frame
                e['sigma_log_mc'].iloc[0],
                e['log_q_obs'].iloc[0], 
                e['sigma_log_q'].iloc[0],
                e['dl'].iloc[0],
                
                e['theta_obs'].iloc[0],
                e['sigma_theta'].iloc[0],
                e['SNR_OBS'].iloc[0], 
                log_SNR_fun, ndet=ndet, size=nsamples)
            
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

    # compute median dL per event
    dl_cutoff = cosmo.dL(population_parameters['zmax'])
    ind_det = np.median(dls, axis=1) < dl_cutoff

    m1s = m1s[ind_det]
    qs = qs[ind_det]
    dls = dls[ind_det]
    pdraws = pdraws[ind_det]
    evts = evts[ind_det]
    if outfile is not None:
        df_samples = pd.DataFrame({'m1': list(m1s),       # each element is an array of nsamples
                                    'q': list(qs), 'dl': list(dls), 'pdraw': list(pdraws), 'evt': list(evts[:,0])})
        df_samples.to_hdf(outfile, key="samples", mode="w")

    return(m1s, qs, dls, pdraws, evts)

if __name__ == "__main__":
    rng = np.random.default_rng(251286134409181405721219170031242732711)

    mult = 1
    inj_file='../src/c2_zp5_snr0.h5'
    obs_file = '../src/data/obsc2_zm55_err.h5'
    sel_file = '../src/sel_c2_zm55_err.h5'
    pe_file='../src/pe_c2_zm55_err.h5'
    ndet=1
    new_sel=True
    if ndet>0:
        jitter=True
    else:
        jitter=False

    detection_threshold = 8
    chunk_size = int(2e6) # memory limit
    num_tot = int(5e7) #how many to reweight, should be les than n_total
    n_total=int(8e7) # how many of our total injectinos to consider, right now its too long so only use some
    #with h5py.File('../src/c2_zp5_snr0.h5', 'r') as f:
    #    n_total = f['true_parameters']['m1'].shape[0][0:int(1e8)]
    population_parameters, cosmo = get_pop_params('../reproduce/configs/c2_zp5.txt')

    grid = np.load("../src/snr_grid_m1det_ext.npz")
    m1_grid  = grid["m1_grid"]
    q_grid   = grid["q_grid"]
    snr_grid = grid["snr_grid"]
    dL_fid   = float(grid["dL_fid"])
    log_snr_interp = RegularGridInterpolator((m1_grid, q_grid), np.log(snr_grid), bounds_error=False, fill_value=-np.inf)

    prior = get_priors_from_file("priors/high_zmax.prior")

    log_dN_obj = intensity_models.LogDNDMDQDV
    pop_params = {key: population_parameters[key] for key in getfullargspec(log_dN_obj)[0][1:] if key in population_parameters.keys()}
    log_dN_func = log_dN_obj(**pop_params)

    # compute weights, one chunk at a time
    print('Pass 1: computing weights...')
    log_w_chunks = []

    for start in tqdm(range(0, n_total, chunk_size)):
        chunk = pd.read_hdf(inj_file, key='true_parameters',
                            start=start, stop=start + chunk_size)
        chunk['dm1sz_dm1ddl2'] = dm1sz_dm1ddl(chunk['z'].to_numpy(), cosmology=cosmo)
        chunk['pdraw_cosmo'] = chunk['pdraw_mqz'] * chunk['dm1sz_dm1ddl2']

        log_dN_vals = log_dN_func(chunk['m1'].values, chunk['q'].values, chunk['z'].values)
        log_w = (log_dN_vals
                 + jnp.log(cosmo.dVCdz(chunk['z'].values))
                 - 2 * jnp.log1p(chunk['z'].values)
                 - jnp.log(chunk['pdraw_cosmo'].values)
                 - jnp.log(cosmo.ddL_dz(chunk['z'].values)))
        log_w_chunks.append(np.array(log_w))

    log_w_all = np.concatenate(log_w_chunks)
    del log_w_chunks  # free memory
    
    log_w_max = np.nanmax(log_w_all)
    accept_prob = np.exp(log_w_all - log_w_max)
    accept_prob = np.nan_to_num(accept_prob, nan=0.0)
    accept_prob /= accept_prob.sum()

    print('Sampling indices...')
    sampled_indices = np.sort(np.random.choice(len(log_w_all), p=accept_prob, size=num_tot))
    # pdraw_sel per sampled index (needed later for sel_samples)
    pdraw_sel_all = np.exp(log_w_all + np.log(np.exp(log_w_all - log_w_max) * 0))  # recompute below per chunk
    del log_w_all, accept_prob
    
    # process the events we want (resampled to our target dist) in chunks
    print('Pass 2: processing selected rows...')
    all_detected_indices_global = []  # global positions in df_det
    evt_offset = 0
    df_det_global_offset = 0  # tracks position in df_det across chunks
    first_chunk = True

    for start in tqdm(range(0, n_total, chunk_size)):
        stop = start + chunk_size

        # which of the sampled_indices fall in this chunk
        mask = (sampled_indices >= start) & (sampled_indices < stop)
        chunk_sampled = sampled_indices[mask] - start  # local indices within chunk

        if len(chunk_sampled) == 0:
            continue

        chunk = pd.read_hdf(inj_file, key='true_parameters',
                            start=start, stop=stop)
        chunk['dm1sz_dm1ddl2'] = dm1sz_dm1ddl(chunk['z'].to_numpy(), cosmology=cosmo)
        chunk['pdraw_cosmo'] = chunk['pdraw_mqz'] * chunk['dm1sz_dm1ddl2']

        log_dN_vals = log_dN_func(chunk['m1'].values, chunk['q'].values, chunk['z'].values)
        log_w = (log_dN_vals
                 + jnp.log(cosmo.dVCdz(chunk['z'].values))
                 - 2 * jnp.log1p(chunk['z'].values)
                 - jnp.log(chunk['pdraw_cosmo'].values)
                 - jnp.log(cosmo.ddL_dz(chunk['z'].values)))

        df_det_chunk = chunk.iloc[chunk_sampled].copy()
        df_det_chunk['pdraw_sel'] = np.exp(np.array(log_w)[chunk_sampled]
                                           + np.log(np.array(chunk['pdraw_cosmo'].values)[chunk_sampled]))
        df_det_chunk['dl'] = cosmo.dL(df_det_chunk['z'].to_numpy())
        df_det_chunk['m1d'] = df_det_chunk['m1'] * (1 + df_det_chunk['z']) 
        df_det_chunk['ndraw'] = num_tot
        df_det_chunk = df_det_chunk.reset_index(drop=True)


        detected_indices, evt_names = get_mock_obs(df_det_chunk, obs_file, cosmo,
            mult=mult, jitter_SNR=jitter, detection_threshold=detection_threshold,
            append_tf=not first_chunk, evt_offset=evt_offset)

        # sel_samples: detected rows with true values
        det_mask = df_det_chunk.index.isin(detected_indices)
        sel_chunk = df_det_chunk[det_mask].copy()
        sel_chunk['ndraw'] = num_tot
        sel_chunk['evt'] = evt_names
        sel_chunk.to_hdf(sel_file, key='true_parameters',
                         mode='w' if first_chunk else 'a',
                         append=not first_chunk, format='table')

        evt_offset += det_mask.sum()
        first_chunk = False

    print('Generating mock PE...')
    m1s, qs, dls, pdraws, evts = gen_mock_PE(obs_file, log_snr_interp, population_parameters, cosmo,
        outfile=pe_file, ndet=ndet, nsamples=300)
    
    if new_sel:
        surviving_evts = set(evts[:, 0])  # evt names that made it through
        sel_samples = pd.read_hdf(sel_file, key='true_parameters')
        sel_samples = sel_samples[sel_samples['evt'].isin(surviving_evts)]
        sel_samples.to_hdf(sel_file, key='true_parameters', format='table', mode='w')
    print("array shapes (we want nevents, nsamples): ", m1s.shape, qs.shape, dls.shape, pdraws.shape)
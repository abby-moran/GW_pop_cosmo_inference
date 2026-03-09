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
from numpyro.infer import MCMC, NUTS, SA
import os.path as op
import pandas as pd
import paths
from utils import get_priors_from_file
from intensity_models import coords
import jax.numpy as jnp
import tqdm
from scipy.special import logsumexp
from inspect import getfullargspec

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
population_parameters = dict()
config_file = '../reproduce/configs/c2_zp5.txt'

population_parameters = dict()
with open(config_file) as param_file:
    for line in param_file:
        (key, val) = line.split('=')
        population_parameters[key.strip()] = val.strip()
        try:
            population_parameters[key.strip()] = float(val.strip())
        except ValueError:
            pass

if __name__ == "__main__":

    cosmo = intensity_models.FlatwCDMCosmology(population_parameters['h'], population_parameters['Om'],
                                            population_parameters['w'], population_parameters['zmax'])

    nmcmc =  1200
    nchain = 1
    random_seed = 1652819403

    prior = get_priors_from_file("priors/high_zmax.prior")
    inj=pd.read_hdf('../src/c2_zp5_snr0.h5', key='true_parameters')[0:int(1e6)]
    #inj=pd.read_hdf('../src/data/obsc2_zm55.h5', key='observations')
    inj=inj[inj['z']<prior['zmax']]

    inj['dm1sz_dm1ddl2']=dm1sz_dm1ddl(inj['z'].to_numpy(), cosmology=cosmo)
    inj['pdraw_cosmo']= inj['pdraw_mqz'] * inj['dm1sz_dm1ddl2']
    inj['m1d'] = inj['m1']*(1+inj['z'])
    ndraw =len(inj)#np.max(inj['ndraw'])

    print('Reweighting injections to target distribution')
    log_dN_obj = intensity_models.LogDNDMDQDV
    pop_params = {key: population_parameters[key] for key in getfullargspec(log_dN_obj)[0][1:] if key in population_parameters.keys()}
    log_dN_func=log_dN_obj(**pop_params)
    log_dN_vals = log_dN_func(inj['m1'].values, inj['q'].values, inj['z'].values)
    log_w = log_dN_vals +jnp.log(cosmo.dVCdz(inj['z'].values)) -2*jnp.log1p(inj['z'].values)- jnp.log(inj['pdraw_cosmo'].values)- jnp.log(cosmo.ddL_dz(inj['z'].values)) 
    log_w_max = np.nanmax(log_w)
    accept_prob = np.exp(log_w - log_w_max)
    indices = np.random.choice(len(inj), p=accept_prob/np.sum(accept_prob), size=int(5e4))
    #indices = np.random.choice(len(inj), p=accept_prob/np.sum(accept_prob),size=int(1e5))

    inj['pdraw_sel']=np.exp(log_w+jnp.log(inj['pdraw_cosmo'].values))
    df_det = inj.iloc[indices].copy()
    df_det['dl'] = cosmo.dL(df_det['z'].to_numpy())
    print('Loading in events and selection samples from the reweighted injections')

    #pe_samples_mock = pd.read_hdf('../src/pe_con2_2k.hdf5', 'samples')
    pe_samples_mock =df_det[0:3000]# pd.read_hdf('../src/pe_con2tw2.hdf5', 'samples') 
    m1s = np.asarray(pe_samples_mock['m1d'].to_list())[:, None]
    qs = np.asarray(pe_samples_mock['q'].to_list())[:, None]
    dls =  np.asarray(pe_samples_mock['dl'].to_list())[:, None]#np.asarray(pe_samples_mock['dl'].to_list()) 
    pdraws =np.ones_like(m1s)# np.asarray(pe_samples_mock['pdraw'].to_list())[:, None]
    print("array shapes (we want nevents, nsamples): ", m1s.shape, qs.shape, dls.shape, pdraws.shape)

    sel_samples=df_det[0:20000]
    ndraw=len(sel_samples)
    assert np.all(m1s > 0) 
    assert np.all(qs > 0) 
    assert np.all(dls >= 0) 
    assert np.all(qs<=1) 
    assert np.all(pdraws>0) 
    assert not np.any(np.isnan(pdraws)) 
    assert not np.any(np.isinf(pdraws))
   
    assert np.all(sel_samples['pdraw_sel']>0) 
    assert not np.any(np.isnan(sel_samples['pdraw_sel'])) 
    assert not np.any(np.isinf(sel_samples['pdraw_sel']))

    kernel = NUTS(intensity_models.pop_cosmo_model)#, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain, progress_bar=True)
    #ndraw=len(df_det)
    mcmc.run(jax.random.PRNGKey(random_seed), m1s, qs, dls, pdraws,
            sel_samples['m1d'].to_list(), sel_samples['q'].to_list(), sel_samples['dl'].to_list(), sel_samples['pdraw_sel'].to_list(), ndraw, prior)
    samples = mcmc.get_samples(group_by_chain=True)
    np.savez("o3_c2zmax55_cut.npz", **samples)
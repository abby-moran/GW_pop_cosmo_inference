ndevice = 1
import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

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
import numpyro.infer.util as util

if __name__ == "__main__":

    nmcmc =  1200
    nchain = 1
    random_seed = 1652819403

    prior = get_priors_from_file("priors/high_zmax.prior")
    pe_samples_mock = pd.read_hdf('../src/pe_c2_zm55_cut.h5', key='samples').iloc[0:2000]   
    m1s = np.asarray(pe_samples_mock['m1'].to_list())
    qs = np.asarray(pe_samples_mock['q'].to_list())
    dls =  np.asarray(pe_samples_mock['dl'].to_list()) +1e-40
    pdraws = np.asarray(pe_samples_mock['pdraw'].to_list())
    print("array shapes (we want nevents, nsamples): ", m1s.shape, qs.shape, dls.shape, pdraws.shape)

    sel_samples=pd.read_hdf('../src/sel_c2_zm55_cut.h5', key='true_parameters')
    ndraw=sel_samples['ndraw'].iloc[0]

    assert np.all(m1s > 0) 
    assert np.all(qs > 0) 
    assert np.all(dls > 0) 
    assert np.all(qs<=1) 
    assert np.all(pdraws>0) 
    assert not np.any(np.isnan(pdraws)) 
    assert not np.any(np.isinf(pdraws))
   
    assert np.all(sel_samples['pdraw_sel']>0) 
    assert not np.any(np.isnan(sel_samples['pdraw_sel'])) 
    assert not np.any(np.isinf(sel_samples['pdraw_sel']))

    kernel = NUTS(intensity_models.pop_cosmo_model)#, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(random_seed), m1s, qs, dls, pdraws,
            sel_samples['m1d'].to_list(), sel_samples['q'].to_list(), sel_samples['dl'].to_list(), sel_samples['pdraw_sel'].to_list(), ndraw, prior)
    samples = mcmc.get_samples(group_by_chain=True)
    np.savez("o3_c2zmax55_cut.npz", **samples)


    
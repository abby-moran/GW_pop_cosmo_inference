import astropy.cosmology as cosmo
from astropy.cosmology import Planck18
import astropy.units as u
import dataclasses
from dataclasses import dataclass
import h5py
import intensity_models
import jax.numpy as jnp
import numpy as np
import intensity_models
from inspect import getfullargspec
from utils import chi_effective_prior_from_isotropic_spins
import pandas as pd
from fisher_snrs import compute_snrs
from scipy.stats import norm
import fisher_snrs
import jax
jax.config.update("jax_enable_x64", True)


COSMO_PARAMS = ['h', 'w', 'Om']
@dataclass
class ModelParameters(object):
    a: object = 1.8
    b: object  = -0.71
    c: object = 2.9
    mpisn: object = 31.0
    mbhmax: object = 36.0
    sigma: object = 2.3
    fpl: object = 0.21
    beta: object = -2.2
    lam: object = 4.7
    kappa:object = 7.0
    zp: object = 3.0
    R: object = 2.3

default_parameters = ModelParameters()

#default_log_dNdmdqdV = intensity_models.LogDNDMDQDV(default_parameters.a, default_parameters.b, default_parameters.c, default_parameters.mpisn, default_parameters.mbhmax, default_parameters.sigma, default_parameters.fpl, default_parameters.beta, default_parameters.lam, default_parameters.kappa, default_parameters.zp)
#default_log_dNdmdqdV.__doc__ = r"""
#Default mass-redshift distribution, more-or-less a reasonable fit to O3a.
#"""

def default_pop_wt(m1, q, z):
    """Weights in `(m1,q,z)` corresponding to the :func:`default_log_dNdmdqdV`."""
    log_dN = default_log_dNdmdqdV(m1, q, z)
    return 4*np.pi*np.exp(log_dN)*Planck18.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value/(1+z)

def pop_wt(m1, q, z, default=True, **kwargs):
    if default and (not kwargs):
        h, Om, w = Planck18.h, Planck18.Om0, -1
        log_dN_func = default_log_dNdmdqdV
    else:
        h, Om, w = kwargs.pop('h'), kwargs.pop('Om'), kwargs.pop('w')
        log_dN_obj = intensity_models.LogDNDMDQDV
        pop_params = {key: kwargs[key] for key in getfullargspec(log_dN_obj)[0][1:] if key in kwargs.keys()}
        log_dN_func = log_dN_obj(**pop_params)
    if "cosmo" not in kwargs.keys():
        cosmo = intensity_models.FlatwCDMCosmology(h, Om, w, zmax = kwargs.get("zmax", 20))
    else:
        cosmo = kwargs.get("cosmo")
    log_dN = log_dN_func(m1, q, z)
    return np.exp(log_dN) * cosmo.dVCdz(z) / (1+z)

def li_prior_wt(m1, q, z, cosmology_weighted=False):
    """Returns LALInference/Bilby prior over `m1`, `q`, and `z`.
    
    `cosmology_weighted` controls whether to use the default prior or one that
    uses the uniform-merger-rate-in-the-comoving-frame redshift weighting."""
    if cosmology_weighted:
        return 4*np.pi*np.square(1+z)*m1*Planck18.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value/(1+z)
    else:
        return np.square(1+z)*m1*np.square(Planck18.luminosity_distance(z).to(u.Gpc).value)*(Planck18.comoving_distance(z).to(u.Gpc).value + (1+z)*Planck18.hubble_distance.to(u.Gpc).value/Planck18.efunc(z))
    
def extract_posterior_samples(file, nsamp, desired_pop_wt=None, rng=None):
    """Returns posterior samples over `m1`, `q`, `z` extracted from `file`.

    The returned samples will be drawn from a density proportional to
    `desired_pop_wt` (or, if none is given, the default LALInference/Bilby
    prior).

    :param file: The file (HDF5) containing posterior samples.

    :param nsamp: The number of samples desired.  The code will raise a
        `ValueError` if too few samples exist in the file.

    :param desired_pop_wt: A function over `(m1, q, z)` giving the desired
        population weight for the returned samples.  If none given, this will be
        the default (non-cosmologically-weighted) LALInference/Bilby prior.

    :param rng: A random number generator used for the draws; if `None`, one
        will be initialized randomly.

    :return: Arrays `(m1, q, z, pop_wt)` giving the samples and (unnormalized)
        weights according to the extracted population.    
    """
    if rng is None:
        rng = np.random.default_rng()

    with h5py.File(file, 'r') as f:
        if 'PublicationSamples' in f.keys():
            # O3a files
            samples = np.array(f['PublicationSamples/posterior_samples'])
        elif 'C01:Mixed' in f.keys():
            # O3b files
            samples = np.array(f['C01:Mixed/posterior_samples'])
        else:
            raise ValueError(f'could not read samples from file {file}')
        
        m1 = np.array(samples['mass_1_source'])
        q = np.array(samples['mass_ratio'])
        z = np.array(samples['redshift'])
        
        m2 = q*m1
        if np.median(m2) < intensity_models.mbh_min:
            raise ValueError(f'rejecting {file} because median m2 < {intensity_models.mbh_min} MSun')

        if desired_pop_wt is None:
            pop_wt = li_prior_wt(m1, q, z)
        else:
            pop_wt = desired_pop_wt(m1, q, z)
        wt = pop_wt / li_prior_wt(m1, q, z)
        wt = wt / np.sum(wt)

        ns = 1/np.sum(wt*wt)
        if ns < 2*nsamp:
            raise ValueError('could not read samples from {:s} due to too few samples: {:.1f}'.format(file, ns))

        inds = rng.choice(np.arange(len(samples)), nsamp, p=wt)
        return (m1[inds], q[inds], z[inds], pop_wt[inds])

def extract_selection_samples(file, nsamp, desired_pop_wt=None, far_threshold=1, rng=None):
    """Return `(m1, q, z, pdraw, nsel)` to estimate selection effects.
    
    :param file: The injection file.

    :param nsamp: The number of samples to be returned.

    :param desired_pop_wt: Function giving a weight in `(m1, q, z)` from which
        the population of injections should be drawn.  If none is given, the
        reference distribution for the actual injections will be used; otherwise
        the distribution of injections will be re-weighted to achieve the
        desired poplation.

    :param far_threshold: The threshold on the FAR (per year) at which an
        injection is considered detected.

    :param rng: A random number generator for the draws; if `None`, one will be
        initialized randomly.

    :return: A tuple `(m1, q, z, pdraw, nsel)`, giving a draw of detected
        injections from the desired population.  `pdraw` is properly normalized
        for estimating detectability as in, e.g., [Farr
        (2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract).
    """
    if rng is None:
        rng = np.random.default_rng()

    with h5py.File(file, 'r') as f:
        m1s_sel = np.array(f['injections/mass1_source'])
        qs_sel = np.array(f['injections/mass2_source'])/m1s_sel
        zs_sel = np.array(f['injections/redshift'])
        a1s_sel = np.sqrt(sum([np.array(f[f'injections/spin1{ii}'])**2 for ii in ['x', 'y', 'z']]))
        a2s_sel = np.sqrt(sum([np.array(f[f'injections/spin2{ii}'])**2 for ii in ['x', 'y', 'z']]))
        costilt1s_sel  = (
            np.array(f[f'injections/spin1z']) / a1s_sel
        )
        costilt2s_sel  = (
            np.array(f[f'injections/spin2z']) / a2s_sel
        )


        pdraw_sel = np.array(f['injections/mass1_source_mass2_source_sampling_pdf'])*np.array(f['injections/redshift_sampling_pdf'])*m1s_sel

        #pdraw_sel *= (np.array(f['injections/spin1x_spin1y_spin1z_sampling_pdf']) * np.array(f['injections/spin2x_spin2y_spin2z_sampling_pdf']) * (2 * np.pi * a1s_sel**2 * 2 * np.pi * a2s_sel**2))
        pycbc_far = np.array(f['injections/far_pycbc_hyperbank'])
        pycbc_bbh_far = np.array(f['injections/far_pycbc_bbh'])
        gstlal_far = np.array(f['injections/far_gstlal'])
        mbta_far = np.array(f['injections/far_mbta'])

        detected = (pycbc_far < far_threshold) | (pycbc_bbh_far < far_threshold) | (gstlal_far < far_threshold) | (mbta_far < far_threshold) 
        ndraw = f.attrs['n_accepted'] + f.attrs['n_rejected']

        T = (f.attrs['analysis_time_s'])/(3600.0*24.0*365.25)

        pdraw_sel /= T

        m1s_sel = m1s_sel[detected]
        qs_sel = qs_sel[detected]
        zs_sel = zs_sel[detected]
        a1s_sel = a1s_sel[detected]
        a2s_sel = a2s_sel[detected]
        costilt1s_sel = costilt1s_sel[detected]
        costilt2s_sel = costilt2s_sel[detected]
        pdraw_sel = pdraw_sel[detected]

        if desired_pop_wt is None:
            pop_wt = pdraw_sel
        else:
            pop_wt = desired_pop_wt(m1s_sel, qs_sel, zs_sel)

        unnorm_wt = pop_wt/pdraw_sel
        sum_norm_wt = unnorm_wt / np.sum(unnorm_wt)
        pdraw_wt = pop_wt / (np.sum(unnorm_wt) / ndraw)

        if nsamp is not None:
            inds = rng.choice(len(m1s_sel), size=nsamp, p=sum_norm_wt)
        else:
            inds = np.arange(len(m1s_sel))
        m1s_sel_cut = m1s_sel[inds]
        qs_sel_cut = qs_sel[inds]
        zs_sel_cut = zs_sel[inds]
        a1s_sel_cut = a1s_sel[inds]
        a2s_sel_cut = a2s_sel[inds]
        costilt1s_sel_cut = costilt1s_sel[inds]
        costilt2s_sel_cut = costilt2s_sel[inds]
        pdraw_sel_cut = pdraw_wt[inds]
        ndraw_cut = len(m1s_sel)

        return m1s_sel_cut, qs_sel_cut, zs_sel_cut, a1s_sel_cut, a2s_sel_cut, costilt1s_sel_cut, costilt2s_sel_cut, pdraw_sel_cut, ndraw_cut
    
def dm1sz_dm1ddl(z, cosmo=None):
    if not cosmo:
        #return (1+z) / (Planck18.comoving_distance(z).to(u.Gpc).value + (1+z)*Planck18.hubble_distance.to(u.Gpc).value / Planck18.efunc(z))
        dm1s_dm1d = (1+z)**-1
        ddl_dz = (Planck18.comoving_distance(z).to(u.Gpc).value + (1 + z) * Planck18.hubble_distance.to(u.Gpc).value / Planck18.efunc(z))

        return dm1s_dm1d * (ddl_dz)**-1
    else:
        return (1+z)**-1 / (cosmo.dC(z) + (1+z)*cosmo.dH / cosmo.E(z))

def draw_mock_samples(log_mc_obs, sigma_log_mc, q_obs, sigma_q, log_dl_obs, sigma_log_dl, size=1, output_source_frame=False, rng=None):
    """
    All inputs in detector frame 
    """
    if rng is None:
        rng = np.random.default_rng()

    log_mcs = rng.normal(loc=log_mc_obs, scale=sigma_log_mc, size=size)

    qs = rng.normal(loc=q_obs, scale=sigma_q, size=size)
    while np.any(qs < 0) or np.any(qs > 1):
        s = (qs < 0) | (qs > 1)
        qs[s] = rng.normal(loc=q_obs, scale=sigma_q, size=np.sum(s))

    log_dls = rng.normal(loc=log_dl_obs, scale=sigma_log_dl, size=size)

    mcs = np.exp(log_mcs)
    m1s = mcs / (qs**(3/5)/((1+qs)**(1/5)))

    dls = np.exp(log_dls)
    
    
    if output_source_frame:
        zs = np.expm1(np.linspace(np.log(1), np.log(1+10), 1024))
        ds = Planck18.luminosity_distance(zs).to(u.Gpc).value
        z = np.interp(dls, ds, zs)
        m1_source = m1s / (1 + z)

        # Flat in log(Mc), q, log(d), so prior is the product of three terms:
        # d log(Mc) / d m1_source = 1/m1_source
        # 1
        # d log(dl) / dz = 1/dl (dC + (1+z)*dH/E(z))
        prior_wt = 1/m1_source/dls*(Planck18.comoving_distance(z).to(u.Gpc).value + (1+z)*Planck18.hubble_distance.to(u.Gpc).value/Planck18.efunc(z))

        return m1_source, qs, z, prior_wt
    else:

        prior_wt =  1/m1s/dls
        #prior_wt = dls**2 * m1s      # same form as get_samples_from_event
    return m1s, qs, dls, prior_wt


class PowerLawPDF(object):
    def __init__(self, alpha, a, b):
        assert alpha > 1

        self.alpha = alpha
        self.a = a
        self.b = b

        self.norm = (self.a - (self.a/self.b)**self.alpha*self.b)/(self.a*(self.alpha-1))

    def __call__(self, x):
        return (self.a/x)**self.alpha/self.a/self.norm
    
    def icdf(self, c):
        return ((self.a**self.alpha*self.b*c + self.a*self.b**self.alpha*(1-c))/(self.a*self.b)**self.alpha)**(1/(1-self.alpha))


def load_jax64(arr):
    """Force array into jax float64, even if it's originally float32 or int."""
    return np.array(arr, dtype=np.float64)

    
def sel_samples_mock(file, nsamp=None, desired_pop_wt=None, SNR=1, rng=None, detectors=['H1','L1'], sensitivity='aligo', batch_num=400, 
                     SNR_load=False, SNR_file='LIGO_SNR.txt', SNR_write=True):
    """Return `(m1, q, z, pdraw, nsel)` to estimate selection effects. Can choose our detector and SNR threshold here to provide more flexbility to generate selection samples for a variety of mock cataglogues
    
    :param file: The injection file.

    :param nsamp: The number of samples to be returned.

    :param desired_pop_wt: Function giving a weight in `(m1, q, z)` from which
        the population of injections should be drawn.  If none is given, the
        reference distribution for the actual injections will be used; otherwise
        the distribution of injections will be re-weighted to achieve the
        desired poplation.

    :param SNR: threshold of SNR. we compute SNR here for a diven detector and then cut as we cut our mock observed population 

    :param rng: A random number generator for the draws; if `None`, one will be
        initialized randomly.

    :return: A tuple `(m1, q, z, pdraw, nsel)`, giving a draw of detected
        injections from the desired population.  `pdraw` is properly normalized
        for estimating detectability as in, e.g., [Farr
        (2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract).
    """

    #first part stays the same
    
    if rng is None:
        rng = np.random.default_rng()

    with h5py.File(file, 'r') as f:
        m1s_sel = load_jax64(f['injections/mass1_source'])
        m2s_sel = load_jax64(f['injections/mass2_source'])
        mts_sel = m1s_sel + m2s_sel
        qs_sel  = load_jax64(f['injections/mass2_source']) / m1s_sel
        zs_sel  = load_jax64(f['injections/redshift'])
        a1s_sel = jnp.sqrt(sum([load_jax64(f[f'injections/spin1{ii}'])**2 for ii in ['x','y','z']]))
        a2s_sel = jnp.sqrt(sum([load_jax64(f[f'injections/spin2{ii}'])**2 for ii in ['x','y','z']]))
        costilt1s_sel = load_jax64(f['injections/spin1z']) / a1s_sel
        costilt2s_sel = load_jax64(f['injections/spin2z']) / a2s_sel
    
        pdraw_sel = (load_jax64(f['injections/mass1_source_mass2_source_sampling_pdf'])* load_jax64(f['injections/redshift_sampling_pdf'])* m1s_sel)
        T = (f.attrs['analysis_time_s'])/(3600.0*24.0*365.25)
        ndraw = f.attrs['n_accepted'] + f.attrs['n_rejected']
        pdraw_sel /= T
    

        #now we need to change how we cut since these FARs aren't necessarily for this detector (can I do this?)
        df = {
            'm1': m1s_sel,
            'q': qs_sel,
            'z': zs_sel,
            'dL': load_jax64(f['injections/distance']) / 1e3,
            'm1d': load_jax64(f['injections/mass1']),
            'ra':  load_jax64(f['injections/right_ascension']), #ra in rad here
            'dec': load_jax64(f['injections/declination']),
            'psi': load_jax64(f['injections/polarization']),
            'gmst': load_jax64(f['injections/gps_time']),
            's1x': load_jax64(f['injections/spin1x']),
            's1y': load_jax64(f['injections/spin1y']),
            's1z': load_jax64(f['injections/spin1z']),
            's2x': load_jax64(f['injections/spin2x']),
            's2y': load_jax64(f['injections/spin2y']),
            's2z': load_jax64(f['injections/spin2z']),
            'iota': load_jax64(f['injections/inclination']),
            'pdraw_mqz': pdraw_sel}
        #snr_list = []    
        
        #SNR_comp=jnp.asarray(fisher_snrs.compute_snrs_batch(df.iloc[0:100], detectors=detectors, sensitivity=sensitivity))
        #return SNR_comp, f['injections/optimal_snr_h'][0:100]
        
        n_events = len(m1s_sel)
        SNR_comp = np.zeros(n_events)
        num_left = n_events
        tot_num = n_events
        num_loops=int(np.trunc(n_events/batch_num)+1)
        if SNR_load:
            snr_net = np.loadtxt(SNR_file) #np.sqrt(np.sum(SNR_comp**2, axis=0))  # sum across detectors

        else:
            for i in range(num_loops):
                start=int(tot_num - num_left)
                stop=int(tot_num - num_left + batch_num)
                if num_left > batch_num:
                    df_here = {k: v[start:stop] for k, v in df.items()}
                else:
                    df_here = {k: v[start:] for k, v in df.items()}
                SNR_batch = fisher_snrs.compute_snrs_batch(df_here, detectors=detectors, sensitivity=sensitivity)
                #SNR_batch = jax.device_get(SNR_batch)  # move to CPU to free GPU
                SNR_batch.block_until_ready()
    
                SNR_comp[start:stop]= SNR_batch
                del df_here, SNR_batch
                #jax.devices()[0].synchronize_all_streams()
                if i% 10 ==0:
                    print("batch done, num left: ", num_left)
                num_left -= batch_num
            snr_net=np.array(SNR_comp)
        
        if SNR_write:
            np.savetxt(SNR_file, SNR_comp)
            snr_net=np.array(SNR_comp)
        detected = (np.array(snr_net) > SNR)#[-1] 

        m1s_sel = m1s_sel[detected]
        qs_sel = qs_sel[detected]
        zs_sel = zs_sel[detected]
        a1s_sel = a1s_sel[detected]
        a2s_sel = a2s_sel[detected]
        costilt1s_sel = costilt1s_sel[detected]
        costilt2s_sel = costilt2s_sel[detected]
        pdraw_sel = pdraw_sel[detected]

        if desired_pop_wt is None:
            pop_wt = pdraw_sel
        else:
            pop_wt = desired_pop_wt(m1s_sel, qs_sel, zs_sel)

        unnorm_wt = pop_wt/pdraw_sel
        sum_norm_wt = unnorm_wt / np.sum(unnorm_wt)
        pdraw_wt = pop_wt / (np.sum(unnorm_wt) / ndraw)

        if nsamp is not None:
            inds = rng.choice(len(m1s_sel), size=nsamp, p=sum_norm_wt)
        else:
            inds = np.arange(len(m1s_sel))
        m1s_sel_cut = m1s_sel[inds]
        qs_sel_cut = qs_sel[inds]
        zs_sel_cut = zs_sel[inds]
        a1s_sel_cut = a1s_sel[inds]
        a2s_sel_cut = a2s_sel[inds]
        costilt1s_sel_cut = costilt1s_sel[inds]
        costilt2s_sel_cut = costilt2s_sel[inds]
        pdraw_sel_cut = pdraw_wt[inds]
        ndraw_cut = len(m1s_sel)

        return m1s_sel_cut, qs_sel_cut, zs_sel_cut, a1s_sel_cut, a2s_sel_cut, costilt1s_sel_cut, costilt2s_sel_cut, pdraw_sel_cut, ndraw_cut
    
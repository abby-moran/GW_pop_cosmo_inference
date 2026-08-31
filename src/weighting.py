from astropy.cosmology import Planck18
import astropy.units as u
import dataclasses
from dataclasses import dataclass
import h5py
import intensity_models
import jax.numpy as jnp
import numpy as np
import os
import re
import intensity_models
from inspect import getfullargspec
from utils import chi_effective_prior_from_isotropic_spins
import pandas as pd
import mock_observations
#import fisher_snrs #import compute_snrs
from scipy.stats import norm, truncnorm
#import fisher_snrs
import jax.scipy.stats as jsst
import jax
jax.config.update("jax_enable_x64", True)
from pycbc.detector import Detector
det = Detector("H1")
from scipy.stats import truncnorm


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
    # Keep the dVCdz/(1+z) to convert rate density to probability density in z
    return np.exp(log_dN) * cosmo.dVCdz(z) / (1+z)

def li_prior_wt(m1, q, z, cosmology_weighted=False):
    """Returns LALInference/Bilby prior over `m1`, `q`, and `z`.
    
    `cosmology_weighted` controls whether to use the default prior or one that
    uses the uniform-merger-rate-in-the-comoving-frame redshift weighting."""
    if cosmology_weighted:
        return 4*np.pi*np.square(1+z)*m1*Planck18.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value/(1+z)
    else:
        return np.square(1+z)*m1*np.square(Planck18.luminosity_distance(z).to(u.Gpc).value)*(Planck18.comoving_distance(z).to(u.Gpc).value + (1+z)*Planck18.hubble_distance.to(u.Gpc).value/Planck18.efunc(z))
    
def get_samples_from_event(file, desired_pop_weight=None, far_threshold=1, zmax = 3, 
                           special_events={ 'GW150914_095045','GW200129_065458','GW190521_074359','GW190521_030229',},
                           normal_preference=['C00:NRSur7dq4','C01:Mixed','C00:Mixed','C00:IMRPhenomXPHM-SpinTaylor',
                                                'C01:IMRPhenomXPHM-SpinTaylor','C01:IMRPhenomXPHM','C01:IMRPhenomPv2_NRTidal:HighSpin',],
                        special_preference = ['C00:NRSur7dq4','C01:IMRPhenomXPHM',
                                'C00:IMRPhenomXPHM-SpinTaylor','C01:IMRPhenomXPHM-SpinTaylor','C01:IMRPhenomPv2_NRTidal:HighSpin',
                                'C01:Mixed','C00:Mixed',] ):    
    match = re.search(r"(GW\d{6}(?:_\d{6})?)",os.path.basename(file) )
    if not match:
        print('unable to find event name for file', file)
        return None
    event = match.group(1)
    with h5py.File(file, 'r') as f:
        # Choose the appropriate waveform preference
        preference = (special_preference if event in special_events else normal_preference)

        # Pick the first available waveform
        for waveform in preference:
            if waveform in f:
                samples = np.array(f[waveform]['posterior_samples'])
                break
        else:
            print(f"Available keys in file {event}: {list(f.keys())}")
            return None

    # Apply redshift cut
    zs=samples['redshift'] [()]
    mask = samples['redshift'] < zmax
    m1_det = samples['mass_1'][()][mask]
    qs = samples['mass_ratio'][()][mask]
    dLs = samples['luminosity_distance'][()][mask] / 1e3

    filename = os.path.basename(file)
    parts = re.split("_|-", filename)
    data_release=parts[1]

    if data_release == 'GWTC4p0' or data_release == 'GWTC5p0':
        # https://arxiv.org/pdf/2605.27225: uniform in redshifted masses, spin mag, 
        # # isotropic spins, sky location
        # # distance prior is uniform merger rate in comoving volume and time
        dvcdz = Planck18.differential_comoving_volume(zs[mask]).to(u.Gpc**3 / u.sr).value #comoving volume
        ddl_dz = (Planck18.comoving_distance(zs[mask]).to(u.Gpc).value + 
                  (1 + zs[mask]) * Planck18.hubble_distance.to(u.Gpc).value / Planck18.efunc(zs[mask]))
        # pdraw(m,q,z)/(dm1_det dm2_det dz) = dvc/dz * dm1m2/dm1q *dz/dl =pdraw/(dm1_det, dq ddL)

        prior =dvcdz * m1_det / ddl_dz 
    else:
        prior = dLs**2 * m1_det
    
    return m1_det, qs, dLs, prior

def finn_chernoff_theta(ra, dec, iota, gps_time):   
    Fp, Fx = det.antenna_pattern(ra, dec,0.0,gps_time)
    theta = 2*np.sqrt(Fp**2 * ((1 + np.cos(iota)**2)/2)**2+ Fx**2 * np.cos(iota)**2)

    return theta

def log_p_spin_magnitude(s, sigma=0.5):
    """log p(s) for truncated Gaussian on [0, 1], std=sigma (Eq. 25) of https://arxiv.org/pdf/2508.10638."""
    a, b = (0.0 - 0.0) / sigma, (1.0 - 0.0) / sigma
    return truncnorm.logpdf(s, a, b, loc=0.0, scale=sigma)

def p_cos_tilt(cos_tau):
    """p(cos tau) mixture, Eq. 26. of https://arxiv.org/pdf/2508.10638"""
    aligned = (1 + cos_tau)**3 / 4.0
    isotropic = 0.5
    return 0.3 * aligned + 0.7 * isotropic

def log_p_spin_vector(sx, sy, sz, sigma=0.5):
    """
    log p(s_x, s_y, s_z) for a single component spin, under the GWTC-5
    injection prior (Eqs. 25-26):https://arxiv.org/pdf/2508.10638 
    iid magnitude (truncated Gaussian),
    tilt mixture, uniform azimuth -- with the spherical->Cartesian Jacobian.
    """
    s = np.sqrt(sx**2 + sy**2 + sz**2)
    cos_tau = np.divide(sz, s, out=np.zeros_like(sz), where=(s > 0))

    log_p_s = log_p_spin_magnitude(s, sigma=sigma)
    log_p_costau = np.log(p_cos_tilt(cos_tau))
    log_p_phi = -np.log(2 * np.pi)
    log_jacobian = -2 * np.log(s)   # d^3s = s^2 ds dcos(tau) dphi

    return log_p_s + log_p_costau + log_p_phi + log_jacobian

def extract_selection_samples(file, nsamp, desired_pop_wt=None, far_threshold=1, rng=None, mass_sel=2.5):
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
    try:
        with h5py.File(file, 'r') as f:
            if 'injections' in f:
                mode = 'hdf5_old'
            else:
                mode = 'unknown'
    except Exception:
        mode = 'pandas'

    # --- load data ---
    if mode == 'hdf5_old':
        with h5py.File(file, 'r') as f:
            m1s_sel = np.array(f['injections/mass1_source'])
            qs_sel = np.array(f['injections/mass2_source'])/m1s_sel
            zs_sel = np.array(f['injections/redshift'])
            a1s_sel = np.sqrt(sum([np.array(f[f'injections/spin1{ii}'])**2 for ii in ['x', 'y', 'z']]))
            a2s_sel = np.sqrt(sum([np.array(f[f'injections/spin2{ii}'])**2 for ii in ['x', 'y', 'z']]))
            costilt1s_sel  = (
                np.array(f[f'injections/spin1z']) / a1s_sel)
            costilt2s_sel  = (
                np.array(f[f'injections/spin2z']) / a2s_sel)


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
            ndraw_cut = np.zeros(len(a2s_sel_cut))+ndraw
    else: #for pd format
        f = pd.read_hdf(file, key='events')
        m1s_sel = np.array(f['mass1_source'])
        qs_sel = np.array(f['mass2_source'])/m1s_sel
        m2s_sel=np.array(f['mass2_source'])
        

        theta_true = np.random.beta(2, 4, len(m1s_sel))  # default fallback

        if 'right_ascension' in f.columns:
            ras = np.array(f['right_ascension'])
            decs = np.array(f['declination'])
            iotas = np.array(f['inclination'])
            gps_time = np.array(f['time_geocenter'])

            good = np.isfinite(ras) & np.isfinite(decs)

            theta_true[good] = finn_chernoff_theta(
                ras[good],
                decs[good],
                iotas[good],
                gps_time[good]
            )
            print(len(gps_time[good]))

        try:
            zs_sel = np.array(f['z'])
        except:
            zs_sel = np.array(f['redshift'])
        m1s_det=m1s_sel*(1+zs_sel)
        a1s_sel = np.sqrt(sum([np.array(f[f'spin1{ii}'])**2 for ii in ['x', 'y', 'z']]))
        a2s_sel = np.sqrt(sum([np.array(f[f'spin2{ii}'])**2 for ii in ['x', 'y', 'z']]))
        costilt1s_sel  = (
            np.array(f[f'spin1z']) / a1s_sel)
        costilt2s_sel  = (
            np.array(f[f'spin2z']) / a2s_sel)

        try:
            pdraw_sel = (np.exp(np.array(f['lnpdraw_mass1_source']))*
                     np.exp(np.array(f['lnpdraw_mass2_source_GIVEN_mass1_source']))*
                    np.exp(np.array(f['lnpdraw_z']))*m1s_sel/np.array(f['weights'])
                )
        except:
            log_pdraw_joint = np.array(f['lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z'])
            log_p_s1 = log_p_spin_vector(np.array(f['spin1x']), np.array(f['spin1y']), np.array(f['spin1z']))
            log_p_s2 = log_p_spin_vector(np.array(f['spin2x']), np.array(f['spin2y']), np.array(f['spin2z']))
            pdraw_sel = np.exp(log_pdraw_joint - log_p_s1 - log_p_s2) * m1s_sel / np.array(f['weights'])
            #dm1_src, dq dz currently
            
        #pdraw_sel *= (np.array(f['spin1x_spin1y_spin1z_sampling_pdf']) * np.array(f['spin2x_spin2y_spin2z_sampling_pdf']) * (2 * np.pi * a1s_sel**2 * 2 * np.pi * a2s_sel**2))
        def get_first(f, keys, example=pdraw_sel):
            for k in keys:
                try:
                    key_out = np.array(f[k])

                    # replace invalid FARs with large value
                    key_out[np.isinf(key_out)] = 100
                    key_out[np.isnan(key_out)] = 100

                    return key_out

                except Exception:
                    pass
            raise KeyError(f"None of these keys found: {keys}")
            #print(f"None of these keys found: {keys}")
            #return np.ones_like(example)

        def combine_far_rowwise(f, keys):
            """Row-wise minimum FAR across all available pipeline/run columns.
            NaN/inf in a given column for a given row means 'this pipeline/run
            didn't report on this injection' -> treat as +inf, not as 100."""
            cols = []
            for k in keys:
                try:
                    arr = np.array(f[k], dtype=float)
                    cols.append(arr)
                except Exception:
                    continue
            if not cols:
                raise KeyError(f"None of these keys found: {keys}")
            stacked = np.vstack(cols)
            stacked[np.isnan(stacked)] = np.inf
            return np.min(stacked, axis=0)

        pycbc_far   = combine_far_rowwise(f, ['pycbc_far', 'o4b_pycbc_far', 'o4a_pycbc_far', 'o3_pycbc_bbh_far'])
        cwb_bbh_far = combine_far_rowwise(f, ['cwb-bbh_far', 'o4b_cwb-bbh_far', 'o4a_cwb-bbh_far', 'o3_cwb_far'])
        gstlal_far  = combine_far_rowwise(f, ['gstlal_far', 'o4b_gstlal_far', 'o4a_gstlal_far', 'o3_gstlal_far'])
        mbta_far    = combine_far_rowwise(f, ['mbta_far', 'o4b_mbta_far', 'o4a_mbta_far', 'o3_mbta_far'])

        # --- semi-analytic path for O1/O2 (no pipeline FAR exists for these rows) ---

        is_semianalytic = f['semianalytic_observed_phase_maximized_snr_net'] > 0
        semianalytic_detected = is_semianalytic & (f['semianalytic_observed_phase_maximized_snr_net'] > 10)

        far_detected = (pycbc_far < far_threshold) | (cwb_bbh_far < far_threshold) | \
                    (gstlal_far < far_threshold) | (mbta_far < far_threshold)

        #is_semianalytic = ~np.isfinite(np.minimum.reduce([pycbc_far, cwb_bbh_far, gstlal_far, mbta_far]))
        #semianalytic_detected = is_semianalytic & (SNR_net > 10)   # LVK uses 9-11; 10 is their stated default

        #pycbc_far = get_first(f, ['pycbc_far', 'o4b_pycbc_far', 'o4a_pycbc_far', 'o3_pycbc_bbh_far'])
        #cwb_bbh_far = get_first(f, ['cwb-bbh_far', 'o4b_cwb-bbh_far', 'o4a_cwb-bbh_far', 'o3_cwb_far'])
        #gstlal_far = get_first(f, ['gstlal_far', 'o4b_gstlal_far', 'o4a_gstlal_far', 'o3_gstlal_far'])
        #mbta_far = get_first(f, ['mbta_far', 'o4b_mbta_far', 'o4a_mbta_far', 'o3_mbta_far'])
        
        
        SNR=np.array(f['estimated_optimal_snr_net'])
        a_rho = (0.0 - SNR) / np.sqrt(3)
        SNR_obs = truncnorm.rvs(a_rho, np.inf, loc=SNR, scale=np.sqrt(3))

        uncert = mock_observations.Uncertainties.from_snr(SNR_obs, mc_scale=.1, q_scale=1.7, th_scale=1.1)
        slmc = np.asarray(uncert.sigma_log_mc)
        sq   = np.asarray(uncert.sigma_q)
        sth  = np.asarray(uncert.sigma_theta)

        # --- log_mc_obs ---
        mc_det=get_mc(m1s_det, qs_sel)
        log_mc_obs = norm.rvs(loc=np.log(mc_det), scale=slmc)
        mc_obs = np.exp(log_mc_obs)

        # --- q_obs (vectorized truncnorm) ---
        a_q = (0.0 - qs_sel) / sq
        b_q = (1.0 - qs_sel) / sq
        q_obs = truncnorm.rvs(a_q, b_q, loc=qs_sel, scale=sq)

        # --- theta_obs (vectorized truncnorm) ---
        a_th = (0.0 - theta_true) / sth
        b_th = (1.0 - theta_true) / sth
        theta_obs = truncnorm.rvs(a_th, b_th, loc=theta_true, scale=sth)

        # --- derived quantities ---
        m1_det_obs = get_m1(mc_obs, q_obs)
        z_obs= theta_obs/theta_true * SNR/SNR_obs
        m1_src_obs = m1_det_obs / (1 + z_obs)
        m2_src_obs = m1_src_obs * q_obs

        #m2s_det=m2s_sel*(1+zs_sel)
        #m1_src = m1_det / (1 + z_obs)
        #m2_src = m1_src * q_obs
        detected = (far_detected | semianalytic_detected) & (m2_src_obs > mass_sel)
        #detected = ((pycbc_far < far_threshold) | (cwb_bbh_far < far_threshold) | (gstlal_far < far_threshold) | (mbta_far < far_threshold)) & (m2_src_obs>mass_sel)
        with h5py.File(file, 'r') as obj:
            ndraw = obj.attrs['total_generated']
            T=np.array(obj.attrs['total_analysis_time'])/(3600.0*24.0*365.25)

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

        #if nsamp is not None:
        #    inds = rng.choice(len(m1s_sel), size=nsamp, p=sum_norm_wt)
        #else:
        #    inds = np.arange(len(m1s_sel))

        if nsamp is not None:
            inds = rng.choice(len(m1s_sel), size=nsamp, p=sum_norm_wt)
            pdraw_final = pdraw_wt      # resampling changes the effective draw density -> use reweighted version
        else:
            inds = np.arange(len(m1s_sel))
            pdraw_final = pdraw_sel 
        m1s_sel_cut = m1s_sel[inds]
        qs_sel_cut = qs_sel[inds]
        zs_sel_cut = zs_sel[inds]
        a1s_sel_cut = a1s_sel[inds]
        a2s_sel_cut = a2s_sel[inds]
        costilt1s_sel_cut = costilt1s_sel[inds]
        costilt2s_sel_cut = costilt2s_sel[inds]
        pdraw_sel_cut = pdraw_final[inds]
        ndraw_cut = np.zeros(len(a2s_sel_cut))+ndraw
        
    return m1s_sel_cut, qs_sel_cut, zs_sel_cut, a1s_sel_cut, a2s_sel_cut, costilt1s_sel_cut, costilt2s_sel_cut, pdraw_sel_cut, ndraw_cut


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

def get_z_obs_true(m1, q_obs, theta_obs, rho_obs, rho_fun, cosmo, ndet=3, dl_fid=1, theta_fid=1):
        
    m1_obs=m1#np.exp(log_mc_obs)
    
    points = np.column_stack([m1_obs, q_obs])
    snr_fid = np.exp(rho_fun(points))
    if ndet==0: #for when we're doing the zero uncertainty case
        ndet=1
    dls = dl_fid*theta_obs/theta_fid * snr_fid*np.sqrt(ndet)/rho_obs
    return cosmo.z_of_dL(dls)


def get_mc(m1, q):
    return m1* (q**(3/5) / (1 + q)**(1/5))

def get_m1(mc, q):
    return mc/(q**(3/5) / (1 + q)**(1/5))
    
def draw_mock_samples_mine(log_mc_obs, sigma_log_mc, q_obs, sigma_q, #log_dl_obs, sigma_log_dl, 
                           theta_obs, sigma_theta, rho_obs, rho_fun, cosmo,
                           size_final=1, detection_threshold=8, rng=None, dl_fid=1, theta_fid=1, ndet=1
                           ,sample_params=True):#, m_min=5.0):
    """
    All inputs in detector frame 
    """
    if sample_params==False:
        log_mcs=np.zeros(size_final)+log_mc_obs
        qs=np.zeros_like(log_mcs)+q_obs
        thetas=np.zeros_like(log_mcs)+theta_obs
        mcs = np.exp(log_mcs)
        m1s = mcs / (qs**(3/5) / (1 + qs)**(1/5))
        rhos_0 = np.zeros_like(log_mcs)+rho_obs
        points = np.column_stack([m1s, qs])
        snr_fid = np.exp(rho_fun(points))
        if ndet==0:
            ndet=1
        dls = dl_fid*thetas/theta_fid * snr_fid*np.sqrt(ndet)/rhos_0
        log_prior_wt = np.zeros(size_final)
        return m1s, qs, dls, log_prior_wt

    if rng is None:
        rng = np.random.default_rng()
    size=10*size_final

    a_q = (0.0 - q_obs) / sigma_q
    b_q = (1 - q_obs) / sigma_q
    qs = truncnorm.rvs(a_q, b_q, loc=q_obs, scale=sigma_q, size=2*size, random_state=rng)
    # compute weights: 1 / Phi(-x / sigma)
    weights = (norm.cdf((1 - q_obs) / sigma_q) - norm.cdf(-q_obs / sigma_q)) / \
          (norm.cdf((1 - qs) / sigma_q) - norm.cdf(-qs / sigma_q))
    
    #  https://arxiv.org/pdf/2411.02494
    weights=np.array(weights)
    weights /= np.sum(weights) #normalize
    ess = 1.0 / np.sum(weights**2)
    if ess < size:
        print(f"Warning: Effective sample size ({ess:.1f}) < requested size ({size})")
    # resample 
    qs = rng.choice(qs, size=size, p=weights)

    if sigma_log_mc==0:
        log_mcs=np.zeros(size)+log_mc_obs
    else:
        log_mcs = norm.rvs(loc=log_mc_obs, scale=sigma_log_mc, size=size, random_state=rng)
    mcs = np.exp(log_mcs)
    m1s = mcs / (qs**(3/5) / (1 + qs)**(1/5))

    #Θ ~ N[0, 1](Θ_obs, σΘ) / [Φ((1 – Θ) / σΘ) – Φ(–Θ/ σΘ)]
    a_th = (0.0 - theta_obs) / sigma_theta
    b_th = (1 - theta_obs) / sigma_theta
    thetas = truncnorm.rvs(a_th, b_th, loc=theta_obs, scale=sigma_theta, size=2*size, random_state=rng)
    # compute weights: 1 / Phi(-x / sigma)
    weights = (norm.cdf((1 - theta_obs) / sigma_theta) - norm.cdf(-theta_obs / sigma_theta)) / \
          (norm.cdf((1 - thetas) / sigma_theta) - norm.cdf(-thetas / sigma_theta))
    weights /= np.sum(weights) #normalize
    ess = 1.0 / np.sum(weights**2)
    if ess < size:
        print(f"Warning: Effective sample size ({ess:.1f}) < requested size ({size})")
    thetas_final= rng.choice(thetas, size=size, p=weights)

    if ndet==0:
        scale = np.sqrt(1)
    else:
        scale = np.sqrt(ndet)
    a_rho = (0.0 - rho_obs) / scale
    rhos_0 = truncnorm.rvs(a_rho, np.inf, loc=rho_obs, scale=scale, size=2*size, random_state=rng)
    weights =  norm.cdf(rho_obs / scale) / norm.cdf(rhos_0 / scale)
    weights /= np.sum(weights) #normalize
    ess = 1.0 / np.sum(weights**2)
    if ess < size:
        print(f"Warning: Effective sample size on rho ({ess:.1f}) < requested size ({size})")
    rhos= rng.choice(rhos_0, size=size, p=weights)

    #dL = dL_fid x (Θ / Θ_fid) x ρ_fid (M, q, dL_fid, Θ_fid)  / ρ
    points = np.column_stack([m1s, qs])
    snr_fid = np.exp(rho_fun(points))
    dls = dl_fid*thetas_final/theta_fid * snr_fid*scale/rhos
    
    eps=1e-30
    
    reweight_fact=dls/rhos *m1s * jsst.beta.pdf(thetas_final, 2, 4) #*qs
    reweight_fact=jnp.nan_to_num(reweight_fact, nan=0, neginf=-1e40, posinf=1e40)
    reweight_fact=reweight_fact/np.sum(reweight_fact)
    ess = 1.0 / np.sum(reweight_fact**2)
    if ess < size_final:
        print(f"Warning: Effective sample size ({ess:.1f}) < requested size ({size_final})")
   
    indicies=np.random.choice(range(size), size=size_final, p=reweight_fact, replace=True)
    m1s=m1s[indicies]
    qs=qs[indicies]
    dls=dls[indicies]
    snr_fid=snr_fid[indicies]

    log_prior_wt = np.zeros(size_final)
    return m1s, qs, dls, log_prior_wt
    

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
                     SNR_load=False, SNR_file='LIGO_SNR.txt', SNR_write=True, z_max=1.9):
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
                SNR_batch.block_until_ready()
                
                SNR_comp[start:stop]= SNR_batch
                if i==0:
                    m1_test=np.array(df_here['m1'])
                    print(m1_test[-20:])
                    print(SNR_batch[-20:])

                del df_here, SNR_batch
                #jax.devices()[0].synchronize_all_streams()
                if i% 10 ==0:
                    print("batch done, num left: ", num_left)
                num_left -= batch_num
            snr_net=np.array(SNR_comp)
        
        if SNR_write:
            np.savetxt(SNR_file, SNR_comp)
            snr_net=np.array(SNR_comp)
        detected = (np.array(snr_net) > SNR) #& (np.array(zs_sel) < z_max)
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
    
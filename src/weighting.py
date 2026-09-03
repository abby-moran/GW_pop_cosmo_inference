from astropy.cosmology import Planck18, FlatLambdaCDM
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
    
# Cosmology assumed by bilby's UniformSourceFrame distance prior in the
# GWTC-4.x/5.x PE releases: 'Planck15_LAL' = flat LambdaCDM with
# H0 = 67.90 km/s/Mpc, Om0 = 0.3065 (Ode0 = 0.6935).  NOT astropy's Planck15
# (H0 = 67.74) and NOT Planck18.  Used to map the released luminosity
# distances back to the redshift the prior was actually defined in.
PLANCK15_LAL = FlatLambdaCDM(H0=67.90, Om0=0.3065)
_P15LAL_ZGRID = np.linspace(0.0, 10.0, 20000)
_P15LAL_DLGRID_GPC = PLANCK15_LAL.luminosity_distance(_P15LAL_ZGRID).to(u.Gpc).value


def _gwtc45_pe_prior(m1_det, dL_Gpc):
    """PE sampling-prior density in (m1_det, q, dL) for GWTC-4/5 release files.

    The declared bilby priors are
      luminosity_distance = UniformSourceFrame(cosmology='Planck15_LAL')
                          -> p(z) propto dVc/dz / (1+z),
                             p(dL) = p(z) / (ddL/dz),  ddL/dz = D_C + (1+z) D_H / E(z)
      chirp_mass = UniformInComponentsChirpMass, mass_ratio = UniformInComponentsMassRatio
                          -> uniform in (m1_det, m2_det) -> p(m1_det, q) propto m1_det
    so  prior = m1_det * dVc/dz / ((1+z) * ddL/dz),  z = z(dL | Planck15_LAL).
    Per-event constants (units, 4pi sr, bounds, normalization) are irrelevant
    to the hierarchical likelihood.  See notes/2026-09-03-gwtc45-pe-prior-fix.md.
    """
    z = np.interp(dL_Gpc, _P15LAL_DLGRID_GPC, _P15LAL_ZGRID)
    dvcdz = PLANCK15_LAL.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value
    ddl_dz = (PLANCK15_LAL.comoving_distance(z).to(u.Gpc).value
              + (1.0 + z) * PLANCK15_LAL.hubble_distance.to(u.Gpc).value
              / PLANCK15_LAL.efunc(z))
    return m1_det * dvcdz / ((1.0 + z) * ddl_dz)


def _decode_h5_string(x):
    x = np.asarray(x).ravel()[0]
    return x.decode() if isinstance(x, bytes) else str(x)


def _check_gwtc45_analytic_prior(file, group_used):
    """Guard: the GWTC-4/5 prior formula above assumes UniformSourceFrame x
    uniform-in-components.  Verify against the file's declared analytic
    priors when present; otherwise warn that the assumption is unchecked."""
    with h5py.File(file, 'r') as f:
        g = f[group_used]
        if 'priors' in g and 'analytic' in g['priors']:
            a = g['priors']['analytic']
            ld = _decode_h5_string(a['luminosity_distance'][()])
            mc = _decode_h5_string(a['chirp_mass'][()])
            if ('UniformSourceFrame' not in ld
                    or 'UniformInComponentsChirpMass' not in mc):
                raise ValueError(
                    f"Unexpected PE prior in {file} group {group_used!r}: "
                    f"luminosity_distance = {ld!r}, chirp_mass = {mc!r}. "
                    "get_samples_from_event assumes UniformSourceFrame x "
                    "UniformInComponentsChirpMass for GWTC-4/5 files.")
        else:
            print(f"WARNING: {os.path.basename(file)} group {group_used!r} "
                  "has no priors/analytic block; ASSUMING the bilby "
                  "UniformSourceFrame(Planck15_LAL) x uniform-in-components "
                  "PE prior.")


def get_samples_from_event(file, desired_pop_weight=None, far_threshold=1, zmax = 3,
                           group=None):
    """Load (m1_det, q, dl_Gpc, prior) PE samples from a single-event release file.

    :param group: If given, read ``f[group]['posterior_samples']`` instead of
        the default fixed-priority key search.  Use this to select the exact
        analysis the LVK population papers used per event (e.g. the
        ``event_sample_IDs`` attribute of a popsummary file).  Returns None if
        the requested group is absent.
    """
    # Fixed-priority group search (used when `group` is None).
    _default_groups = ('PublicationSamples',          # O3a
                       'C01:Mixed',                   # O3b
                       'PrecessingSpinIMRHM',         # GWTC-2.1
                       'C00:Mixed',                   # O4
                       'C00:NRSur7dq4',               # other bit of O4
                       'C01:IMRPhenomXPHM-SpinTaylor',  # GWTC-5
                       'C00:IMRPhenomXPHM-SpinTaylor')  # GWTC-5
    with h5py.File(file, 'r') as f:
        if group is not None:
            if group not in f.keys():
                print(f"Requested group {group!r} not in file {file}; "
                      f"available keys: {list(f.keys())}")
                return None
            group_used = group
        else:
            group_used = next((g for g in _default_groups if g in f.keys()), None)
            if group_used is None:
                print(f"Available keys in file {file}: {list(f.keys())}")
                return None
        samples = np.array(f[group_used]['posterior_samples'])

    zs=samples['redshift'] [()]
    mask = samples['redshift'] < zmax
    m1_det = samples['mass_1'][()][mask]
    qs = samples['mass_ratio'][()][mask]
    dLs = samples['luminosity_distance'][()][mask] / 1e3

    filename = os.path.basename(file)
    parts = re.split("_|-", filename)
    data_release=parts[1]

    if data_release in ('GWTC4p0', 'GWTC4p1', 'GWTC5p0', 'GWTC5p1'):
        # bilby UniformSourceFrame(Planck15_LAL) x uniform-in-components:
        # prior = m1_det * dVc/dz / ((1+z) * ddL/dz), z from dL under
        # Planck15_LAL (the file's `redshift` column uses a different
        # cosmology).  See _gwtc45_pe_prior and
        # notes/2026-09-03-gwtc45-pe-prior-fix.md.
        _check_gwtc45_analytic_prior(file, group_used)
        prior = _gwtc45_pe_prior(m1_det, dLs)
    else:
        print(filename)
        prior = dLs**2 * m1_det
    
    return m1_det, qs, dLs, prior

def finn_chernoff_theta(ra, dec, iota, gps_time):
        
    Fp, Fx = det.antenna_pattern(
        ra,
        dec,
        0.0,
        gps_time
    )

    theta = 2*np.sqrt(
        Fp**2 * ((1 + np.cos(iota)**2)/2)**2
        + Fx**2 * np.cos(iota)**2
    )

    return theta

def extract_selection_samples(file, nsamp, desired_pop_wt=None, far_threshold=1, rng=None, mass_sel=2.5):
    """Return `(m1, q, z, pdraw, nsel)` to estimate selection effects.

    :param file: The injection file.

    :param nsamp: The number of samples to be returned.  If `None` (the usual
        case), ALL detected injections are returned with their true draw
        density `pdraw` and `ndraw` = total number of proposals; no
        resampling or rescaling of `pdraw` is performed.

    :param desired_pop_wt: Function giving a weight in `(m1, q, z)` from which
        the population of injections should be drawn.  Only used when `nsamp`
        is not `None`: the detected injections are then resampled to the
        desired population (Farr 2019 convention) and the returned `ndraw`
        equals `nsamp`.

    :param far_threshold: The threshold on the FAR (per year) at which an
        injection is considered detected.

    :param rng: A random number generator for the draws; if `None`, one will be
        initialized randomly.

    :param mass_sel: Cut on the TRUE source-frame secondary mass (Msun);
        injections with `mass2_source <= mass_sel` are discarded.

    :return: A tuple `(m1, q, z, a1, a2, costilt1, costilt2, pdraw, ndraw)`,
        giving the detected injections.  `pdraw` is properly normalized for
        estimating detectability as in, e.g., [Farr
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

            # NOTE: this legacy branch is for the old O3 injection-file format
            # (an 'injections' group with per-parameter sampling_pdf datasets
            # and n_accepted/n_rejected attrs); it lacks the GWTC-5 mixture
            # columns (searches attr, joint lnpdraw, semianalytic SNR), so the
            # detection mask and pdraw construction above are left as-is.
            if nsamp is None:
                # No subsampling: return every detected injection with its
                # true draw density and the total number of proposals.
                # (Rescaling pdraw by pop_wt/(sum(wt)/ndraw) here, as an old
                # version did, is only valid in the resampling convention
                # below and biases mu_sel by ~ndraw/nsel otherwise.)
                inds = np.arange(len(m1s_sel))
                pdraw_wt = pdraw_sel
                ndraw_out = ndraw
            else:
                # Farr (2019) resampling: draw nsamp injections with weight
                # pop_wt/pdraw; the resampled set is distributed as pop_wt
                # (restricted to detections), so the consistent draw density
                # is pop_wt/(sum(wt)/ndraw) with ndraw_out = nsamp.
                if desired_pop_wt is None:
                    pop_wt = pdraw_sel
                else:
                    pop_wt = desired_pop_wt(m1s_sel, qs_sel, zs_sel)
                unnorm_wt = pop_wt/pdraw_sel
                sum_norm_wt = unnorm_wt / np.sum(unnorm_wt)
                pdraw_wt = pop_wt / (np.sum(unnorm_wt) / ndraw)
                inds = rng.choice(len(m1s_sel), size=nsamp, p=sum_norm_wt)
                ndraw_out = nsamp
            m1s_sel_cut = m1s_sel[inds]
            qs_sel_cut = qs_sel[inds]
            zs_sel_cut = zs_sel[inds]
            a1s_sel_cut = a1s_sel[inds]
            a2s_sel_cut = a2s_sel[inds]
            costilt1s_sel_cut = costilt1s_sel[inds]
            costilt2s_sel_cut = costilt2s_sel[inds]
            pdraw_sel_cut = pdraw_wt[inds]
            ndraw_cut = np.zeros(len(a2s_sel_cut))+ndraw_out
    else: #for pd format
        # GWTC-5 sensitivity mixture file (Zenodo 19500052): a pandas 'events'
        # table of found+hopeless injections spanning the semianalytic O1/O2
        # epochs and the real O3/O4a/O4b injection campaigns.
        f = pd.read_hdf(file, key='events')
        with h5py.File(file, 'r') as obj:
            searches = [s.decode() if isinstance(s, bytes) else s
                        for s in obj.attrs['searches']]
            ndraw = obj.attrs['total_generated']
            T = np.array(obj.attrs['total_analysis_time'])/(3600.0*24.0*365.25)

        m1s_sel = np.array(f['mass1_source'])
        m2s_sel = np.array(f['mass2_source'])
        qs_sel = m2s_sel/m1s_sel

        try:
            zs_sel = np.array(f['z'])
        except KeyError:
            zs_sel = np.array(f['redshift'])
        a1s_sel = np.sqrt(sum([np.array(f[f'spin1{ii}'])**2 for ii in ['x', 'y', 'z']]))
        a2s_sel = np.sqrt(sum([np.array(f[f'spin2{ii}'])**2 for ii in ['x', 'y', 'z']]))
        costilt1s_sel  = (
            np.array(f[f'spin1z']) / a1s_sel)
        costilt2s_sel  = (
            np.array(f[f'spin2z']) / a2s_sel)

        # Draw density over (m1_source, q, z) per detector-frame year.  The
        # stored lnpdraw is the joint density over (m1_source, m2_source, z,
        # cartesian spin components); multiplying by m1 converts m2 -> q, and
        # multiplying by (4 pi a^2) per spin marginalizes the cartesian spin
        # density against the isotropic, uniform-magnitude (a_max = 1) spin
        # population.  (An old commented-out version used 2 pi -- wrong by a
        # factor of 4 per injection pair.)
        try:
            # older mock format: factorized mass/z draw pdfs with no spin term
            pdraw_sel = (np.exp(np.array(f['lnpdraw_mass1_source']))*
                     np.exp(np.array(f['lnpdraw_mass2_source_GIVEN_mass1_source']))*
                    np.exp(np.array(f['lnpdraw_z']))*m1s_sel/np.array(f['weights'])
                )
        except KeyError:
            pdraw_sel = (np.exp(np.array(f['lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z']))*
                        m1s_sel/np.array(f['weights'])*
                        (4*np.pi*a1s_sel**2)*(4*np.pi*a2s_sel**2))

        # Detection: an injection is found if its semianalytic (O1/O2 epochs)
        # observed SNR exceeds 10, or if ANY of the searches listed in the
        # file's 'searches' attribute recovered it below the FAR threshold.
        # Missing FARs are stored as +inf, so a plain elementwise min is safe.
        far_cols = np.vstack([np.array(f[f'{s}_far']) for s in searches])
        if np.isnan(far_cols).any():
            raise ValueError("NaNs found in FAR columns; expected missing FARs to be +inf")
        far = far_cols.min(axis=0)
        snr_net = np.array(f['semianalytic_observed_phase_maximized_snr_net'])

        # Cut on the TRUE source-frame secondary mass; scattering the cut
        # through mock-observed masses is inconsistent with the true pdraw.
        detected = ((snr_net > 10) | (far < far_threshold)) & (m2s_sel > mass_sel)

        pdraw_sel /= T

        m1s_sel = m1s_sel[detected]
        qs_sel = qs_sel[detected]
        zs_sel = zs_sel[detected]
        a1s_sel = a1s_sel[detected]
        a2s_sel = a2s_sel[detected]
        costilt1s_sel = costilt1s_sel[detected]
        costilt2s_sel = costilt2s_sel[detected]
        pdraw_sel = pdraw_sel[detected]

        if nsamp is None:
            # No subsampling: return every detected injection with its true
            # draw density and ndraw = total number of proposals.  (Rescaling
            # pdraw by pop_wt/(sum(wt)/ndraw) here, as an old version did, is
            # only valid in the resampling convention below; applied without
            # resampling it biases mu_sel by ~ndraw/nsel.)
            inds = np.arange(len(m1s_sel))
            pdraw_wt = pdraw_sel
            ndraw_out = ndraw
        else:
            # Farr (2019) resampling: draw nsamp injections with weight
            # pop_wt/pdraw; the resampled set is distributed as pop_wt
            # (restricted to detections), so the consistent draw density is
            # pop_wt/(sum(wt)/ndraw) with ndraw_out = nsamp.
            if desired_pop_wt is None:
                pop_wt = pdraw_sel
            else:
                pop_wt = desired_pop_wt(m1s_sel, qs_sel, zs_sel)
            unnorm_wt = pop_wt/pdraw_sel
            sum_norm_wt = unnorm_wt / np.sum(unnorm_wt)
            pdraw_wt = pop_wt / (np.sum(unnorm_wt) / ndraw)
            inds = rng.choice(len(m1s_sel), size=nsamp, p=sum_norm_wt)
            ndraw_out = nsamp
        m1s_sel_cut = m1s_sel[inds]
        qs_sel_cut = qs_sel[inds]
        zs_sel_cut = zs_sel[inds]
        a1s_sel_cut = a1s_sel[inds]
        a2s_sel_cut = a2s_sel[inds]
        costilt1s_sel_cut = costilt1s_sel[inds]
        costilt2s_sel_cut = costilt2s_sel[inds]
        pdraw_sel_cut = pdraw_wt[inds]
        ndraw_cut = np.zeros(len(a2s_sel_cut))+ndraw_out

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
    
from astropy.cosmology import Planck18
import astropy.units as u
import lal
import lalsimulation as lalsim
import numpy as np
import os.path as op
import sys
import pandas as pd
import paths
from tqdm import tqdm, trange
import weighting
import scipy.integrate as sint
import intensity_models
import jimFisher
from jimFisher.Fisher import FisherSamples
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from bilby.gw.conversion import component_masses_to_chirp_mass
import ripple as rp
from ripplegw import ms_to_Mc_eta
from ripplegw.waveforms import IMRPhenomD
import lalsimulation as lalsim
import os

from fisher_utils import convert_to_ripple_params, IFO_GEOMETRIC_KEYS, BOUNDS
from scipy.interpolate import interp1d

sensitivity_path = f"{os.path.dirname(__file__)}/sensitivity_files"
#jimGW_detectors = jimgw.single_event.detector.detector_preset

ASD_FILES = {"CE": f"{sensitivity_path}/LIGO-P1600143-v18-CE-ASD.txt",
            "aplus": f"{sensitivity_path}/AplusDesign.txt",
            "aligo": f"{sensitivity_path}/aligo_O4high.txt",
            'o3_PSD': f"{sensitivity_path}/H1_o3_PSD.txt"}


#SENSITIVITIES = {'aligo': lalsim.SimNoisePSDaLIGODesignSensitivityP1200087,
#                'aplus': lalsim.SimNoisePSDaLIGOAPlusDesignSensitivityT1800042,
#                'CE': lalsim.SimNoisePSDCosmicExplorerP1600143}

def next_pow_2(x):
    np2 = 1
    while np2 < x:
        np2 = np2 << 1
    return np2

def compute_snrs(d, detectors=['H1'], sensitivity='aligo', fmin=20.0, fmax=2048.0, psdstart=20.0, f_ref=20.0):
    """
    Compute SNRs for a set of sources described in dataframe d
    d: pandas.DataFrame
        Contains parameters
        d in Gpc
    detectors: list of str
    sensitivity : str, PSD choice for ripple
    fmin, fmax: float
        freq band (Hz) - don't mess with this
    psdstart: float, starting freq 
 
    Returns network SNRs for each row in d
    borrowed some stuff from https://gist.github.com/maxisi/e3bb4af28edd892b38448340a3e90a75
    """
    
    snrs = []
    #fs = jnp.linspace(fmin, fmax, int(2*fmax))
    freqs, sens = np.loadtxt(ASD_FILES[sensitivity], unpack=True)
    if sensitivity=='o3_PSD':
        psd=sens
    else:
        psd = sens**2 #assuming ASD here
        
    for _, r in tqdm(d.iterrows(), total=len(d)):
        # unpack source params
        m2s = r.m1*r.q
        m1d = r.m1*(1+r.z)
        m2d = m2s*(1+r.z)

        a1 = np.sqrt(r.s1x*r.s1x + r.s1y*r.s1y + r.s1z*r.s1z)
        a2 = np.sqrt(r.s2x*r.s2x + r.s2y*r.s2y + r.s2z*r.s2z)
        
        params_here = {"mass_1": m1d, "mass_2": m2d, "s1_z": r.s1z, "s2_z": r.s2z, 
              "luminosity_distance": r.dL * 1e3, "phase_c": 0., "cos_iota": np.cos(r.iota), "ra": r.ra, #Gpc to Mpc 
              "sin_dec": np.sin(r.dec), 'psi': r.psi, "t_c": 0., "s1_x": r.s1x, "s1_y": r.s1y, "s2_x": r.s2x, "s2_y": r.s2y}
        params_here['chirp_mass'], params_here['eta'] = ms_to_Mc_eta(jnp.asarray((params_here['mass_1'], params_here['mass_2'])))
        params_here['mass_ratio'] = params_here['mass_2'] / params_here['mass_1']
        params_here_ripple=convert_to_ripple_params(params_here)
        
        theta_ripple= jnp.array([params_here_ripple['M_c'], params_here_ripple['eta'], 0.0, 0.0, params_here_ripple['d_L'], #chi1=chi2=0, dist in Mpc
                                 params_here_ripple['t_c'], params_here_ripple['psi'], params_here_ripple['iota']]) 
        #theta_ripple = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])
        
        T = max(4, next_pow_2(lalsim.SimInspiralChirpTimeBound(fmin, m1d*lal.MSUN_SI, m2d*lal.MSUN_SI, a1, a2)))
        Nfft = int(T*2*fmax)
        freq_arr_i = np.linspace(0, fmax, Nfft, endpoint=False)
        freq_arr_i = jnp.array(freq_arr_i[(freq_arr_i >= fmin) & (freq_arr_i <= fmax)])
        
        sn_det = []
        for det in detectors:
            det_obj = lal.cached_detector_by_prefix[det]
            gmst = lal.GreenwichMeanSiderealTime(params_here_ripple['t_c'])
            Fp, Fc = lal.ComputeDetAMResponse(det_obj.response, float(params_here_ripple['ra']),  float(params_here_ripple['dec']), 
                                              float(params_here_ripple['psi']), float(gmst))
            dt = lal.TimeDelayFromEarthCenter(det_obj.location, float(params_here_ripple['ra']),  float(params_here_ripple['dec']), float(params_here_ripple['t_c']))
            
            hp, hc = IMRPhenomD.gen_IMRPhenomD_hphc(freq_arr_i, theta_ripple, f_ref)

            # detector projection
            h_fd = (Fp * hp + Fc * hc) * jnp.exp(2j * jnp.pi * freq_arr_i * dt)

            psd_interped = interp1d(freqs, psd, fill_value=(psd[0], psd[-1]))(freq_arr_i)
            psd_vals = jnp.array(psd_interped)# psd_series.data.data)
            
            # SNR
            snr_here = compute_optimal_SNR(h_fd, psd_vals, freq_arr_i)
            sn_det.append(snr_here)
        
        snr = jnp.sqrt(jnp.sum(jnp.array(sn_det)**2))  # quadrature sum across detectors
        snrs.append(snr)

    return jnp.array(snrs)
    
def compute_optimal_SNR(h, psd, fs):
    """
    df element normalizes by 1/T
    """
    df = fs[1] - fs[0]  # frequency step
    psd=psd
    integrand = (jnp.abs(h) ** 2) / psd
    snr_sq = 4 * jnp.sum(integrand) * df
    return jnp.sqrt(snr_sq)



# JAX BATCHED VERSION?
def compute_snrs_batch(df, detectors=['H1'], sensitivity='aligo', fmin=20.0, fmax=2048.0, deltaf=.25, f_ref=20.0):
    """
    Fully JAX-native SNR computation for a batch of GW events.
    df: pandas.DataFrame with columns m1,q,z,s1x,s1y,s1z,s2x,s2y,s2z,dL,iota,psi,ra,dec
    mass in source frame
    dL in Gpc
    """

    # Convert parameters to JAX arrays
    m1 = jnp.array(df["m1"])
    q = jnp.array(df["q"])
    z = jnp.array(df["z"])
    s1x = jnp.array(df["s1x"])
    s1y = jnp.array(df["s1y"])
    s1z = jnp.array(df["s1z"])
    s2x = jnp.array(df["s2x"])
    s2y = jnp.array(df["s2y"])
    s2z = jnp.array(df["s2z"])
    dL = jnp.array(df["dL"]) * 1e3  # Gpc => Mpc
    iota = jnp.array(df["iota"])
    psi = jnp.array(df["psi"])
    ra = jnp.array(df["ra"])
    dec = jnp.array(df["dec"])
    gps_time = jnp.array(df["gps_time"])

    # Derived quantities
    m2 = m1 * q
    m1d = m1 * (1 + z)
    m2d = m2 * (1 + z)
    a1 = jnp.sqrt(s1x**2 + s1y**2 + s1z**2)
    a2 = jnp.sqrt(s2x**2 + s2y**2 + s2z**2)
    nevents=len(np.array(df["m1"]))

    df_val = float(deltaf)
    fs = np.arange(0, fmax + df_val, df_val)
    valid = (fs >= fmin) & (fs <= fmax)
    freq_common = jnp.array(fs[valid])
    df_jax = freq_common[1] - freq_common[0]

    # Load PSD and create interpolation
    freqs_psd, sens = np.loadtxt(ASD_FILES[sensitivity], unpack=True)
    if sensitivity == 'o3_PSD':
        psd = sens
    else:
        psd = sens**2  # assuming ASD if not labeled PSD
    psd_interp = interp1d(freqs_psd, psd, bounds_error=False, fill_value="extrapolate")


    # Precompute detector responses
    def precompute_detectors():
        dets = {}
        for det in detectors:
            det_obj = lal.cached_detector_by_prefix[det]
            Fp_list, Fc_list, dt_list = [], [], []
            for i in range(nevents):
                gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(gps_time[i])))%(2*np.pi)
                Fp, Fc = lal.ComputeDetAMResponse(det_obj.response, float(ra[i]), float(dec[i]), float(psi[i]), float(gmst))
                dt = lal.TimeDelayFromEarthCenter(det_obj.location, float(ra[i]), float(dec[i]), float(gps_time[i]))
                Fp_list.append(Fp)
                Fc_list.append(Fc)
                dt_list.append(dt)
            dets[det] = (jnp.array(Fp_list), jnp.array(Fc_list), jnp.array(dt_list))
        return dets

    det_objs = precompute_detectors()

    # Convert to ripple parameters
    def convert_single(i):
        p = {"mass_1": m1d[i], "mass_2": m2d[i], "s1_z": s1z[i], "s2_z": s2z[i], "luminosity_distance": dL[i], "iota": iota[i], 
             "psi": psi[i], "t_c": 0.0, "s1_x": s1x[i], "s1_y": s1y[i], "s2_x": s2x[i], "s2_y": s2y[i]} 
        p['chirp_mass'], eta = ms_to_Mc_eta(jnp.asarray((p['mass_1'], p['mass_2']))) #detector frame 
        eta = jnp.where(eta >= 0.249, 0.249, eta)
        p['eta']=eta
        p['mass_ratio'] = p['mass_2'] / p['mass_1']
        theta = jnp.array([p['chirp_mass'], p['eta'], p['s1_z'],p['s2_z'], p['luminosity_distance'], p['t_c'], p['psi'], p['iota']]) #spins dont matter here 
        # just take the z component, or the total magnitude 
        return theta

    theta_ripple = jax.vmap(convert_single)(np.arange(nevents))
    
    # SNR computation using mask multiplication (no boolean indexing)
    def snr_single(theta, event_idx):
        hp, hc = IMRPhenomD.gen_IMRPhenomD_hphc(freq_common, theta, f_ref)
        sn_det = []
        psd_vals = jnp.array(psd_interp(np.array(freq_common)))
        psd_safe = jnp.where(psd_vals > 0.0, psd_vals, jnp.inf)

        for det in detectors:
            Fp_arr, Fc_arr, dt_arr = det_objs[det]
            Fp_i, Fc_i, dt_i = Fp_arr[event_idx], Fc_arr[event_idx], dt_arr[event_idx]
            # Detector projection and phase shift
            h_fd = (Fp_i * hp + Fc_i * hc) * jnp.exp(2j * jnp.pi * freq_common * dt_i)
            integrand = (jnp.abs(h_fd)**2) / psd_safe 
            #df = freq_common[1] - freq_common[0] 
            sn_det.append(jnp.sqrt(4 * jnp.sum(integrand) * df_jax)) 
        return jnp.sqrt(jnp.sum(jnp.array(sn_det)**2))

    snr_fn = jax.vmap(snr_single, in_axes=(0, 0))
    snrs = snr_fn(theta_ripple, jnp.arange(nevents))

    return snrs

#def compute_Nfft(m1, m2, a1, a2, fmin, fmax):
#    # Use LAL to get chirp time bound
#    T = max(4, next_pow_2(lalsim.SimInspiralChirpTimeBound(float(fmin), float(m1*lal.MSUN_SI), float(m2*lal.MSUN_SI), float(a1), float(a2))))
#    return int(T * 2 * fmax)


def compute_snrs_old(d, detectors = ['H1', 'L1'], sensitivity = 'aligo', fmin = 20, fmax = 2048, psdstart = 20):
    psdstop = 0.95*fmax
    fref = fmin

    fishers = dict()
    snrs = []
    for dur in [4, 8, 16, 32, 64, 128, 256, 512]:
        for det in detectors:
            name=f"fisher_{det}_{dur}s"
            fishers[name] = FisherSamples(name=name, fmin = 20, fmax = 2048, sensitivity=sensitivity, location=det, duration=dur,trigger_time=0,waveform="IMRPhenomD", f_ref=fref)

    for _, r in tqdm(d.iterrows(), total=len(d)):
        m2s = r.m1*r.q
        m1d = r.m1*(1+r.z)
        m2d = m2s*(1+r.z)

        a1 = np.sqrt(r.s1x*r.s1x + r.s1y*r.s1y + r.s1z*r.s1z)
        a2 = np.sqrt(r.s2x*r.s2x + r.s2y*r.s2y + r.s2z*r.s2z)


        T = max(4, next_pow_2(lalsim.SimInspiralChirpTimeBound(fmin, m1d*lal.MSUN_SI, m2d*lal.MSUN_SI, a1, a2)))

        sn = []
        # distance goes Gpc to Mpc here
        # jim GW takes distances in MPC by default: https://colab.research.google.com/drive/1ah_mwVpn3A32jhctA6BTj-Nqk7SGf9Dj?usp=sharing
        # for an event 440 Mpc away
        for det in detectors:
            fisher = fishers[f"fisher_{det}_{T}s"]
            params_here = {"mass_1": m1d, "mass_2": m2d, "s1_z": r.s1z, "s2_z": r.s2z, 
              "luminosity_distance": r.dL * 1e3, "phase_c": 0., "cos_iota": np.cos(r.iota), "ra": r.ra, #Gpc to Mpc 
              "sin_dec": np.sin(r.dec), 'psi': r.psi, "t_c": 0., "s1_x": r.s1x, "s1_y": r.s1y, "s2_x": r.s2x, "s2_y": r.s2y}
            params_here['chirp_mass'] = component_masses_to_chirp_mass(params_here['mass_1'], params_here['mass_2'])
            params_here['mass_ratio'] = params_here['mass_2'] / params_here['mass_1']
            
            snr = fisher.get_snr(params_here)
            sn.append(snr)
        sn = jnp.array(sn)
        snrs.append(jnp.sqrt(jnp.sum(jnp.square(sn))))

    return snrs

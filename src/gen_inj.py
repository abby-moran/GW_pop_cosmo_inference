from astropy.cosmology import Planck18
import astropy.units as u
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
import lalsimulation as lalsim
import numpy as np
import os.path as op
import sys
import pandas as pd
import paths
import jax.numpy as jnp
from tqdm import tqdm, trange
import weighting
import scipy.integrate as sint
import intensity_models
from inspect import getfullargspec
import scipy
import fisher_snrs
from fisher_snrs import compute_snrs
#import mock_injections
#from mock_injections import *
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


import jax
import h5py
import jax.scipy.special as jss
jax.config.update("jax_enable_x64", True)

SENSITIVITIES = {'aligo': lalsim.SimNoisePSDaLIGODesignSensitivityP1200087,
                'aplus': lalsim.SimNoisePSDaLIGOAPlusDesignSensitivityT1800042,
                'CE': lalsim.SimNoisePSDCosmicExplorerP1600143}

population_parameters = dict()
config_file = '../reproduce/configs/config3.txt'
#outfile = 'new_mock_inj_cut.h5'#'mock_injections_o3_zp1.h5'

population_parameters = dict()
with open(config_file) as param_file:
    for line in param_file:
        (key, val) = line.split('=')
        population_parameters[key.strip()] = val.strip()
        try:
            population_parameters[key.strip()] = float(val.strip())
        except ValueError:
            pass
snr_threshold = 1
sensitivity='o3_PSD'
detectors = population_parameters.pop('detectors', 'H1').split(',')
custom_cosmo = intensity_models.FlatwCDMCosmology(population_parameters['h'], population_parameters['Om'], population_parameters['w'], population_parameters['zmax'])
population_parameters['cosmo'] = custom_cosmo
print("Using the following custom population_parameters: " + str(population_parameters))

class ZPDF(object):
    def __init__(self, lam, kappa, zp, zmax, cosmo):
        self.lam = lam
        self.kappa = kappa
        self.zp = zp

        self.zmax = zmax
        self.cosmo = cosmo

        self.zinterp = np.expm1(np.linspace(np.log(1), np.log(1+self.zmax), 1024))
        self.norm = 1

        unnorm_pdf = self(self.zinterp)
        
        self.norm = 1/np.trapz(unnorm_pdf, self.zinterp)
        self.pdfinterp = unnorm_pdf * self.norm

        self.cdfinterp = sint.cumtrapz(self.pdfinterp, self.zinterp, initial=0)

    def __call__(self, z):
        if self.cosmo == 'default':
            return self.norm*(1+z)**self.lam / (1 + ((1+z)/(1+self.zp))**self.kappa) * Planck18.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value / (1+z)
        else:
            return self.norm*(1+z)**self.lam / (1 + ((1+z)/(1+self.zp))**self.kappa) * (self.cosmo.dVCdz(z)) / (1+z)

    def icdf(self, c):
        return np.interp(c, self.cdfinterp, self.zinterp)
    

class InterpolatedPDF(object):
    def __init__(self, xs, cdfs):
        self.xs = xs
        self.cdfs = cdfs / cdfs[-1]
        self.pdfs = np.diff(cdfs) / np.diff(xs)

    def __call__(self, x):
        x = np.atleast_1d(x)
        i = np.searchsorted(self.xs, x)-1

        return self.pdfs[i]
    
    def icdf(self, c):
        return np.interp(c, self.cdfs, self.xs)

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


if __name__ == "__main__":
    grid = np.load("snr_grid_m1det_ext.npz")
    
    m1_grid  = grid["m1_grid"]
    q_grid   = grid["q_grid"]
    snr_grid = grid["snr_grid"]
    dL_fid   = float(grid["dL_fid"])
    
    log_snr_interp = RegularGridInterpolator((m1_grid, q_grid), np.log(snr_grid), bounds_error=False, fill_value=-np.inf)

    for i in range(20):
        ndraw=int(1e7)
        a=(0-population_parameters["zp"])/(population_parameters["zp"])
        b=(population_parameters["zmax"]-population_parameters["zp"])/(population_parameters["zp"])
        zpdf = scipy.stats.truncnorm(a, b, loc=population_parameters["zp"], scale=(population_parameters["zp"]))
        
        a=(population_parameters["mbh_min"]-population_parameters["mpisn"])/(2*population_parameters["mpisn"])
        mpdf = scipy.stats.truncnorm(a, np.inf, loc=population_parameters["mpisn"], scale=(2*population_parameters["mpisn"]))
        
        rng = np.random.default_rng()
        z = zpdf.ppf(rng.uniform(low=0, high=1, size=ndraw))
        m = mpdf.ppf(rng.uniform(low=0, high=1, size=ndraw))
        
        offset=population_parameters['mbh_min']/m
        qpdf = scipy.stats.uniform(loc=0+offset, scale=1-offset) #goes from loc to loc+scale
        q = qpdf.ppf(rng.uniform(0, 1, size=ndraw))  
        
        
        #mtpdf = PowerLawPDF(2, m+population_parameters['mbh_min'], 2 * m)
        
        #mt = mtpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))
        
        #m2 = mt - m
        #q = m2/m
        
        #print("calculating pdraws")
        #pdraw = mpdf.pdf(m)*zpdf.pdf(z)*(mtpdf(mt)*m)
        
        # for truncated uniform q
        # then
        
        mt=m+q*m
        m2 = mt - m
        print("calculating pdraws")
        #pdraw = mpdf.pdf(m)*zpdf.pdf(z)*(1.0 / (1.0 - offset))
        pdraw = mpdf.pdf(m) * zpdf.pdf(z) * qpdf.pdf(q)
        
        m1d = m * (1 + z)
        iota = np.arccos(rng.uniform(low=-1, high=1, size=ndraw))
        
        ra = rng.uniform(low=0, high=2*np.pi, size=ndraw)
        dec = np.arcsin(rng.uniform(low=-1, high=1, size=ndraw))
        
        # 0 < psi < pi, uniformly distributed
        psi = rng.uniform(low=0, high=np.pi, size=ndraw)
        gmst = rng.uniform(low=0, high=2*np.pi, size=ndraw)
        m1_det = m * (1 + z)
        points = np.column_stack([m1_det, q])
        rho0 = np.exp(log_snr_interp(points))
        
        print("assigning spins")
        
        s1x, s1y, s1z = 0,0,0#rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3,ndraw))
        s2x, s2y, s2z = 0,0,0#rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3,ndraw))
        
        
        print("calculating dLs")
        
        dm1sz_dm1ddl = weighting.dm1sz_dm1ddl(z, cosmo=population_parameters['cosmo'])
        dL = population_parameters['cosmo'].dL(z)# dL in Gpc 
        
        det = detectors[0]
        
        #A0 = np.sqrt(Fp0**2 + Fc0**2)
        rho0 = np.exp(log_snr_interp(points))
        #rho0 /= A0
        rho = rho0 * (dL_fid / dL)
        
        #Fp = np.zeros(ndraw)
        #Fc = np.zeros(ndraw)
        
        #for i in range(ndraw):
        #    Fp[i], Fc[i] = lal.ComputeDetAMResponse(lal.cached_detector_by_prefix[detectors[0]].response, ra[i], dec[i], psi[i], gmst[i])
        
        cosi = np.cos(iota)
        #Theta_sq = (Fp**2 * (1 + cosi**2)**2 / 4 + Fc**2 * cosi**2)
        #Theta = np.sqrt(Theta_sq)
        #SNR_comp = rho * Theta
        zeros=jnp.zeros(len(m))
        
        df = {
            'm1': jnp.array(m),
            'q': jnp.array(q),
            'z': jnp.array(z),
            'dL': jnp.array(dL), #in GPC here
            'm1d': jnp.array(m1d),
            'iota': jnp.array(iota),
            'ra': jnp.array(ra),
            'dec': jnp.array(dec),
            'psi': jnp.array(psi),
            'gmst': jnp.array(gmst),
            's1x': zeros, 
            's1y': zeros, 
            's1z': zeros, #jnp.zeros(len(m)), 
            's2x': zeros, #jnp.zeros(len(m)), 
            's2y': zeros, #jnp.zeros(len(m)), 
            's2z': zeros, #jnp.zeros(len(m)), 
            'pdraw_mqz': jnp.array(pdraw),
            'dm1sz_dm1ddl': jnp.array(dm1sz_dm1ddl),
            'SNR_0': jnp.array(rho),
            'ndraw': zeros+ndraw
        }
        
        #df["SNR"] = SNR_comp
        
        # Convert dict of JAX arrays -> dict of NumPy arrays
        df_np = {k: np.asarray(v) for k, v in df.items()}
        
        # Build DataFrame
        df_pd = pd.DataFrame(df_np)
        cosmo = intensity_models.FlatwCDMCosmology(population_parameters['h'], population_parameters['Om'],
                                           population_parameters['w'], population_parameters['zmax'])


        log_dN_obj = intensity_models.LogDNDMDQDV
        pop_params = {key: population_parameters[key] for key in getfullargspec(log_dN_obj)[0][1:] if key in population_parameters.keys()}
        log_dN_func=log_dN_obj(**pop_params)
        log_dN_vals = log_dN_func(df_pd['m1'].values, df_pd['q'].values, df_pd['z'].values)
        log_w = log_dN_vals - jnp.log(pdraw) +jnp.log(cosmo.dVCdz(df_pd['z'].values)) -jnp.log1p(df_pd['z'].values)
        pdraw_sel=np.zeros_like(pdraw)
        
        # stable accept-reject
        log_w_max = np.nanmax(log_w)
        accept_prob = np.exp(log_w - log_w_max)
        u = rng.uniform(size=len(accept_prob))
        sel_mask = u < accept_prob
        df_det = df_pd[sel_mask]
        Fp = np.zeros(len(df_det))
        Fc = np.zeros(len(df_det))
        
        ra = rng.uniform(low=0, high=2*np.pi, size=len(df_det))
        dec = np.arcsin(rng.uniform(low=-1, high=1, size=len(df_det)))
        
        
        psi = rng.uniform(low=0, high=np.pi, size=len(df_det))
        gmst = rng.uniform(low=0, high=2*np.pi, size=len(df_det))
        iota = np.arccos(rng.uniform(low=-1, high=1, size=len(df_det)))
        
        for i in range(len(df_det)):
            Fp[i], Fc[i] = lal.ComputeDetAMResponse(lal.cached_detector_by_prefix[detectors[0]].response, ra[i], dec[i], psi[i], gmst[i])
        
        cosi = np.cos(iota)
        Theta_sq = (Fp**2 * (1 + cosi**2)**2 / 4 + Fc**2 * cosi**2)
        Theta = np.sqrt(Theta_sq)
        SNR_comp = df_det['SNR_0'] * Theta
        df_det['Theta']=Theta

        df_det['SNR']=SNR_comp
        
        
        df_det = df_det[df_det['SNR'] > snr_threshold]
        df_det=df_det.drop(columns=['SNR_0'])
        #print(f"Retained {len(df_det)} samples after rejection sampling and applying snr cut.")
        if i==0:
            df_det.to_hdf('con3_SNRcomp_ext.h5', key='true_parameters', mode='a', format='table', append=False)
        else:
            df_det.to_hdf('con3_SNRcomp_ext.h5', key='true_parameters', mode='a', format='table', append=True)

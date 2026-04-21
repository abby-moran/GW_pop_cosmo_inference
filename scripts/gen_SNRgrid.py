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
sys.path.append('../src/')
import jax.numpy as jnp
from tqdm import tqdm, trange
import weighting
import scipy.integrate as sint
import intensity_models
from inspect import getfullargspec
import scipy
import fisher_snrs
#import mock_injections
#from mock_injections import *
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import scipy
import jax
import h5py
import jax.scipy.special as jss
jax.config.update("jax_enable_x64", True)

SENSITIVITIES = {'aligo': lalsim.SimNoisePSDaLIGODesignSensitivityP1200087,
                'aplus': lalsim.SimNoisePSDaLIGOAPlusDesignSensitivityT1800042,
                'CE': lalsim.SimNoisePSDCosmicExplorerP1600143}

population_parameters = dict()
config_file='pop_configs/mock_GWTC5_evo.txt'
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
sensitivity='aplus_PSD'
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


def make_frequency_grid(fmin=20., fmax=2048., deltaf=0.25):
    fs = np.arange(fmin, fmax + deltaf, deltaf)
    return jnp.array(fs), fs[1] - fs[0]

def build_snr_grid(df_grid, m1_grid, q_grid, detectors, sensitivity, batch_num=400):
    Ngrid = len(df_grid['m1'])
    snr_out = np.zeros(Ngrid)

    num_left = Ngrid
    tot_num  = Ngrid
    num_loops = int(np.trunc(Ngrid / batch_num) + 1)

    for i in range(num_loops):
        if num_left == 0:
            break

        start = int(tot_num - num_left)
        stop  = int(min(start + batch_num, tot_num))

        df_here = {k: v[start:stop] for k, v in df_grid.items()}

        snr_batch = fisher_snrs.compute_snrs_batch(df_here, detectors=detectors, 
                                                   sensitivity=sensitivity, use_antenna=False)
        if i% 10 ==0:
            print("batch done, num left: ", num_left)

        snr_out[start:stop] = np.array(snr_batch)
        num_left -= (stop - start)

    return snr_out.reshape(len(m1_grid), len(q_grid))

if __name__ == "__main__":
    N_m1=1000
    m1_src_max = 450
    z_max = population_parameters["zmax"]
    
    m1_det_min = .1
    m1_det_max = m1_src_max * (1 + z_max)
    q_min=.0001 #population_parameters['mbh_min']/m1_src_max
    N_q=N_m1
    m1_grid = jnp.logspace(jnp.log10(m1_det_min), jnp.log10(m1_det_max), N_m1)
    q_grid  = jnp.linspace(q_min, 1.0, N_q)
    M1, Q = jnp.meshgrid(m1_grid, q_grid, indexing="ij")
    
    m1_flat = M1.flatten()
    q_flat  = Q.flatten()
    Ngrid   = len(m1_flat)
    zeros=jnp.zeros(len(m1_flat))
    
    df_grid = {
        'm1':  m1_flat,
        'q':   q_flat,
        'z':   jnp.full(Ngrid, 0), #we dont use this yet
        'dL':  jnp.full(Ngrid, 1),  # Gpc
        'iota': jnp.full(Ngrid, 0),
        'psi':  jnp.full(Ngrid, 0),
        'ra':   jnp.full(Ngrid, 0),
        'dec':  jnp.full(Ngrid, 0),
        'gmst': jnp.full(Ngrid, 0),
        's1x': zeros,
        's1y': zeros,
        's1z': zeros,
        's2x': zeros,
        's2y': zeros,
        's2z': zeros,
        'pdraw_mqz': jnp.ones(Ngrid),
        'dm1sz_dm1ddl': jnp.ones(Ngrid),
    }
    SNRgrid=build_snr_grid(df_grid, m1_grid, q_grid, detectors, sensitivity)
    np.savez("snr_grid_aplus.npz", m1_grid=m1_grid, q_grid=q_grid, snr_grid=SNRgrid, dL_fid=1.0) # dL in Gpc
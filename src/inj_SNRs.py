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

import fisher_snrs
#from fisher_snrs import compute_snrs
import mock_injections
from mock_injections import *
import matplotlib.pyplot as plt

import jax
import h5py
import jax.scipy.special as jss
jax.config.update("jax_enable_x64", True)

ASD_FILES = 'sensitivity_files/aligo_O4high.txt'
PSD_FILE ='sensitivity_files/H1_o3_PSD.txt'

freqs, sens = np.loadtxt(ASD_FILES, unpack=True)
psd = sens**2 #assuming ASD here

freqs_o3, sens_03 = np.loadtxt(PSD_FILE, unpack=True)
psd_03 = sens_03 #assuming ASD here
SENSITIVITIES = {'aligo': lalsim.SimNoisePSDaLIGODesignSensitivityP1200087,
                'aplus': lalsim.SimNoisePSDaLIGOAPlusDesignSensitivityT1800042,
                'CE': lalsim.SimNoisePSDCosmicExplorerP1600143}

population_parameters = dict()
config_file = '../reproduce/configs/config_zp1.txt'

population_parameters = dict()
with open(config_file) as param_file:
    for line in param_file:
        (key, val) = line.split('=')
        population_parameters[key.strip()] = val.strip()
        try:
            population_parameters[key.strip()] = float(val.strip())
        except ValueError:
            pass
            
sensitivity='o3_PSD'
detectors = population_parameters.pop('detectors', 'H1').split(',')


if __name__ == "__main__":
    m1, q, z, a1, a2, cos_tilt1, cos_tilt2, pdraw, ndraw = weighting.sel_samples_mock('../endo3_bbhpop-LIGO-T2100113-v12.hdf5', 
                                                                                  detectors=detectors, SNR=8, 
                                                                                  sensitivity=sensitivity,  SNR_load=False,
                                                                                  SNR_write=True,  SNR_file='LIGO_SNR_1222.txt')
    df = pd.DataFrame({'m1': m1, 'q': q, 'z': z, 'a1': a1, 'a2':a2, 'cos_tilt_1': cos_tilt1, 'cos_tilt_2': cos_tilt2, 'pdraw_m1sqz': pdraw, 'ndraw': ndraw}) 
    df['dm1sz_dm1ddl'] = weighting.dm1sz_dm1ddl(df['z']) #gives us factor to go soruce to detector frame mass, and z to dL in prior
    df.to_hdf('./selection_samples_o3_1222.h5', 'samples')
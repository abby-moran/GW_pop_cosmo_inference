from astropy.cosmology import Planck18
import astropy.units as u
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import sys
sys.path.append('../src/')
import lalsimulation as lalsim
import numpy as np
import pandas as pd
import jax.numpy as jnp
import weighting
import scipy.integrate as sint
import intensity_models
import scipy
from scipy.interpolate import RegularGridInterpolator
import configparser
import argparse
from pathlib import Path
import shutil
import subprocess
import os
import jax
import jax.scipy.special as jss
jax.config.update("jax_enable_x64", True)

SENSITIVITIES = {'aligo': lalsim.SimNoisePSDaLIGODesignSensitivityP1200087,
                'aplus': lalsim.SimNoisePSDaLIGOAPlusDesignSensitivityT1800042,
                'CE': lalsim.SimNoisePSDCosmicExplorerP1600143}

population_parameters = dict()
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to run config file")
args = parser.parse_args()

cfg = configparser.ConfigParser()
cfg.read(args.config)
base_runs_dir = "../runs"
run_name = cfg["run"]["run_dir"]

os.makedirs(base_runs_dir, exist_ok=True)
run_dir = os.path.join(base_runs_dir, f"{run_name}")
os.makedirs(run_dir, exist_ok=False)

# Real output file lives on ceph
ceph_run_dir = os.path.abspath(os.path.join("../../ceph/GW_pop_cosmo_inference/", run_name))
os.makedirs(ceph_run_dir, exist_ok=True)
ceph_output_file = os.path.join(ceph_run_dir, cfg["run"]["output_file_inj"])

# Symlink the local path to the ceph file
output_file = os.path.join(run_dir, cfg["run"]["output_file_inj"])
os.symlink(ceph_output_file, output_file)

base_dir = Path("pop_configs")

ndet = int(cfg["run"]["ndet"])
snr_threshold = float(cfg["run"]["snr_threshold_0"])
sensitivity=cfg["run"]["sensitivity"]
#config_file = '../reproducepop_configs/configs/c_new_zm55.txt'
try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    git_hash = "UNKNOWN"

config_file = base_dir / cfg["run"]["pop_config_file"]
config_file = config_file.resolve()

cfg["run"]["git_hash"] = git_hash
ini_file = cfg["run"]["ini_file"]
run_ini_path = os.path.join(run_dir, ini_file)
with open(run_ini_path, "w") as f:
    cfg.write(f)
pop_config_copy = os.path.join(run_dir, os.path.basename(config_file))
shutil.copy(config_file, run_dir)

snr_file=cfg["run"]["snr_grid"]
grid = np.load(snr_file)

population_parameters = dict()
with open(config_file) as param_file:
    for line in param_file:
        (key, val) = line.split('=')
        population_parameters[key.strip()] = val.strip()
        try:
            population_parameters[key.strip()] = float(val.strip())
        except ValueError:
            pass

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
    
    m1_grid  = grid["m1_grid"]
    q_grid   = grid["q_grid"]
    snr_grid = grid["snr_grid"]
    dL_fid   = float(grid["dL_fid"])
    
    log_snr_interp = RegularGridInterpolator((m1_grid, q_grid), np.log(snr_grid), bounds_error=False, fill_value=-np.inf)
    num_loops=120
    for i in range(num_loops):
        ndraw=int(1e7)
        zpdf = scipy.stats.uniform(loc=0, scale=population_parameters["zmax"])        
        a=(.4-population_parameters["mpisn"])/(2*population_parameters["mpisn"])
        #mpdf = scipy.stats.powerlaw(.8, loc=.4, scale=np.inf, loc=population_parameters["mpisn"], scale=(2*population_parameters["mpisn"]))
        mpdf= scipy.stats.powerlaw(.5, loc=3, scale=2500)
        
        rng = np.random.default_rng()
        z = zpdf.ppf(rng.uniform(low=0, high=1, size=ndraw))
        m = mpdf.ppf(rng.uniform(low=0, high=1, size=ndraw))
        
        offset=.4/m#population_parameters['mbh_min']/m
        qpdf = scipy.stats.uniform(loc=0+offset, scale=1-offset) #goes from loc to loc+scale
        q = qpdf.ppf(rng.uniform(0, 1, size=ndraw))  
        
        mt=m+q*m
        m2 = mt - m
        print("calculating pdraws")
        #pdraw = mpdf.pdf(m)*zpdf.pdf(z)*(1.0 / (1.0 - offset))
        pdraw = mpdf.pdf(m) * zpdf.pdf(z) * qpdf.pdf(q)
        
        m1d = m * (1 + z)
        points = np.column_stack([m1d, q])
        rho0 = np.exp(log_snr_interp(points))
        
        print("assigning spins")
        
        s1x, s1y, s1z = 0,0,0#rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3,ndraw))
        s2x, s2y, s2z = 0,0,0#rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3,ndraw))
        
        
        print("calculating dLs")
        
        dm1sz_dm1ddl = weighting.dm1sz_dm1ddl(z, cosmology=population_parameters['cosmo'])
        dL = population_parameters['cosmo'].dL(z)# dL in Gpc 
        
        det = detectors[0]
        
        rho0 = np.exp(log_snr_interp(points))
        rho = rho0 * (dL_fid / dL)
        
        zeros=jnp.zeros(len(m))
        
        df = {
            'm1': jnp.array(m),
            'q': jnp.array(q),
            'z': jnp.array(z),
            'dL': jnp.array(dL), #in GPC here
            'm1d': jnp.array(m1d),
            's1x': zeros, 
            's1y': zeros, 
            's1z': zeros, #jnp.zeros(len(m)), 
            's2x': zeros, #jnp.zeros(len(m)), 
            's2y': zeros, #jnp.zeros(len(m)), 
            's2z': zeros, #jnp.zeros(len(m)), 
            'pdraw_mqz': jnp.array(pdraw),
            'dm1sz_dm1ddl': jnp.array(dm1sz_dm1ddl),
            'SNR_0': jnp.array(rho),
            'ndraw': zeros+ndraw*num_loops
        }
                
        # Convert dict of JAX arrays -> dict of NumPy arrays
        df_np = {k: np.asarray(v) for k, v in df.items()}
        
        # Build DataFrame
        df_pd = pd.DataFrame(df_np)
        cosmo = intensity_models.FlatwCDMCosmology(population_parameters['h'], population_parameters['Om'],
                                           population_parameters['w'], population_parameters['zmax'])
        
        df_det = df_pd
        Fp = np.zeros(len(df_det))
        Fc = np.zeros(len(df_det))
        
        ra = rng.uniform(low=0, high=2*np.pi, size=len(df_det))
        dec = np.arcsin(rng.uniform(low=-1, high=1, size=len(df_det)))
        
        Theta=np.random.beta(2,4,len(df_det))
        SNR_comp = df_det['SNR_0'] * Theta* np.sqrt(ndet)
        df_det['Theta']=Theta

        df_det['SNR']=SNR_comp
        
        
        df_det = df_det[df_det['SNR'] > snr_threshold]
        df_det=df_det.drop(columns=['SNR_0'])
        print(len(df_det))
        if i==0:
            df_det.to_hdf(output_file, key='true_parameters', mode='a', format='table', append=False)
        else:
            df_det.to_hdf(output_file, key='true_parameters', mode='a', format='table', append=True)

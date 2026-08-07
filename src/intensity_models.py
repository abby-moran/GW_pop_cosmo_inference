from astropy.cosmology import Planck18
import astropy.units as u
import dataclasses
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import jax.scipy.stats as jsst
from jax import lax
import numpy as np
import numpyro 
import numpyro.distributions as dist
from utils import jnp_cumtrapz, sample_parameters_from_dict, log_expit
from jax.scipy.ndimage import map_coordinates
from functools import partial



@jax.jit
def mean_mbh_from_mco(mco, mpisn, mbhmax):
    a = 1 / (4*(mpisn - mbhmax))
    mcomax = 2*mbhmax - mpisn
    return jnp.where(mco < mpisn, mco, mbhmax + a*jnp.square(mco - mcomax))

@jax.jit
def largest_mco(mpisn, mbhmax):
    mcomax = 2*mbhmax - mpisn
    return mcomax + jnp.sqrt(4*mbhmax*(mbhmax - mpisn))

@jax.jit
def log_dNdmCO(mco, a, b, mco_floor=6.0):
    mtr = 20.0
    mco_eff = jnp.maximum(mco, mco_floor)   # floors mco before evaluating the power law, mco_floor>=mco_min
    x = mco_eff / mtr
    return jnp.where(mco_eff < mtr, -a*jnp.log(x), -b*jnp.log(x))

@jax.jit
def smooth_log_dNdmCO(xx, a, b):
    xtr = 20
    delta = 0.05
    return -a * jnp.log(xx / xtr) + delta * (a - b) * jnp.log(0.5 * (1 + (xx/xtr)**(1/delta)))

@jax.jit
def log_smooth_turnon(m, mmin, width=0.05):
    dm = mmin*width
    return -jnp.log1p(jnp.exp(-(m-mmin)/dm))

@jax.jit
def mmin_log_smooth_turnon(m, delta_m, mmin):
    shifted_mass = jnp.nan_to_num((m - mmin) / delta_m, nan=0)
    shifted_mass = jnp.clip(shifted_mass, 1e-6, 1 - 1e-6)
    exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
    exponent = jnp.where(exponent > 87.0, 87.0, exponent)
    window = jax.lax.logistic(-exponent)
    logwindow = jnp.where(m < mmin, -jnp.inf, jnp.log(window))
    return logwindow

@jax.jit
def log_gaussian_bump(m, mu, sigma):
    return -0.5*jnp.square((m - mu) / sigma)

@jax.jit
def log_trapz_grid(log_f, x):
    log_dx = jnp.log(jnp.diff(x))
    return jss.logsumexp(
        jnp.log(0.5) + jnp.logaddexp(log_f[..., :-1], log_f[..., 1:]) + log_dx,
        axis=-1,
    )

@jax.jit
def log_normalized_gaussian(m, mu, sigma):
    return log_gaussian_bump(m, mu, sigma) - 0.5*jnp.log(2*jnp.pi) - jnp.log(sigma)

@jax.jit
def log_normalized_power_law_tail(m, mbhmax, c):
    return jnp.log(c - 1) - jnp.log(mbhmax) - c*jnp.log(m/mbhmax)


@dataclass
class LogDNDMPISN(object):
    a: object
    b: object
    mpisn: object
    mbhmax: object
    sigma: object
    mco_min: object = 4.0
    n_m: object = 512
    mbh_grid: object = dataclasses.field(init=False)
    log_dN_grid: object = dataclasses.field(init=False)
    log_Z_grid: object = dataclasses.field(init=False)

    def __post_init__(self):
        min_bh_mass = 1.5
        min_co_mass = 1.0
        max_bh_mass = 100.0
        max_co_mass = 100.0

        log_mbh = jnp.linspace(jnp.log(min_bh_mass), jnp.log(max_bh_mass), self.n_m+2)
        log_mco = jnp.linspace(jnp.log(min_co_mass), jnp.log(max_co_mass), self.n_m)

        sigma = self.sigma
        log_mco = log_mco[None,:,None]
        log_mbh = log_mbh[None,None,:]
        mpisn = self.mpisn[:,None,None]
        mbhmax = self.mbhmax[:,None,None]

        mbh = jnp.exp(log_mbh)
        mco = jnp.exp(log_mco)

        mu = mean_mbh_from_mco(mco, mpisn, mbhmax)
        mu_min = 0.1
        mu = jnp.where(mu > 0, mu, mu_min)
        log_mu = jnp.log(mu)

        log_p = -0.5*jnp.square((log_mbh - log_mu)/sigma) - 0.5*jnp.log(2*jnp.pi) - jnp.log(sigma) - log_mbh

        log_mco_window = log_smooth_turnon(mco, self.mco_min, width=0.05)
        log_wts = log_dNdmCO(mco, self.a, self.b) + log_mco_window + log_p

        log_trapz = np.log(0.5) + jnp.logaddexp(log_wts[:,:-1,:], log_wts[:,1:,:]) + jnp.log(jnp.diff(mco, axis=1))

        self.log_dN_grid = jss.logsumexp(log_trapz, axis=1)
        self.mbh_grid = mbh[0,0,:]

        self.log_Z_grid = log_trapz_grid(self.log_dN_grid, self.mbh_grid)

@jax.jit
def safe_log(x, eps=1e-300):
    return jnp.log(jnp.clip(x, eps, None))

@dataclass
class LogDNDM(object):
    a: object
    b: object
    c: object
    mpisn: object
    mpisndot: object
    mbhmax: object
    sigma: object
    fpl: object      
    mp_low: object
    msigma_low: object
    flow: object      
    mco_min: object = 4.0
    mbh_min: object = 3.0
    delta_m: object = 2.5
    zmax: object = 20
    mref: object = 30.0
    zref: object = 0.001
    use_low_bump: bool = True          # static flag, not sampled/traced
    log_dndm_pisn: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.dmbhmax = self.mbhmax - self.mpisn
        self.setup_interp()

    def setup_interp(self):
        self.z_array = jnp.expm1(jnp.linspace(np.log(1), jnp.log(1+self.zmax), 30))
        mpisns = self.mpisn + self.mpisndot * (1 - 1/(1+self.z_array))
        mbhmaxs = mpisns + self.dmbhmax
        self.log_dndm_pisn = LogDNDMPISN(self.a, self.b, mpisns, mbhmaxs, self.sigma, mco_min=self.mco_min)
        self.mbh_grid = self.log_dndm_pisn.mbh_grid
        self.log_dndm_pisn_grid = self.log_dndm_pisn.log_dN_grid.T
        self.log_Z_pisn_grid = self.log_dndm_pisn.log_Z_grid
        self.mbhmaxs = jnp.asarray(mbhmaxs)

    def interp_2d_dndmpisn(self, m, z):
        m, z = jnp.broadcast_arrays(m, z)
        m_idx = jnp.interp(m, self.mbh_grid, jnp.arange(self.mbh_grid.shape[0]))
        z_idx = jnp.interp(z, self.z_array, jnp.arange(self.z_array.shape[0]))
        coords = jnp.stack([m_idx, z_idx], axis=0)
        return map_coordinates(self.log_dndm_pisn_grid, coords, order=1, mode='nearest')

    def log_Z_pisn_at_z(self, z):
        return jnp.interp(z, self.z_array, self.log_Z_pisn_grid)

    def __call__(self, m, z):
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        log_p_pisn_raw = self.interp_2d_dndmpisn(m, z)
        log_p_pisn_raw = jnp.where(m >= self.log_dndm_pisn.mbh_grid[-1], -np.inf, log_p_pisn_raw)
        log_p_pisn = log_p_pisn_raw - self.log_Z_pisn_at_z(z)   # unit-area shape

        mbhmax_at_samples = jnp.array(self.mpisn + self.mpisndot*(1 - 1/(1+z)) + self.dmbhmax)
        # Tail shape: closed-form normalized power law (integrates to unit area over m > mbhmax exactly) with a smooth turn-on at mbhmax (continuity)
        log_p_pl_raw = jnp.where(m < mbhmax_at_samples, -jnp.inf,
            log_normalized_power_law_tail(m, mbhmax_at_samples, self.c))
        log_p_pl = log_p_pl_raw + log_smooth_turnon(m, mbhmax_at_samples)

         # flow and fpl are mixture-weight ratios (relative to the PISN component's weight of 1), not height-anchored amplitudes. Converting to a proper simplex
        if self.use_low_bump:
            log_p_low = log_normalized_gaussian(m, self.mp_low, self.msigma_low)   # unit-area shape
            # simplex over {w_pisn, w_low, w_pl}, always integrates to 1 no matter params
            log_denom = jnp.log1p(self.flow + self.fpl)
            log_w_pisn = -log_denom
            log_w_low = safe_log(self.flow) - log_denom
            log_w_pl = safe_log(self.fpl) - log_denom

            log_dNdm = jnp.logaddexp(log_w_pisn + log_p_pisn, log_w_low + log_p_low)
            log_dNdm = jnp.logaddexp(log_dNdm, log_w_pl + log_p_pl)
        else:
            # simplex over just {w_pisn, w_pl} -- no bump term at all,
            # exactly zero contribution rather than a numerically tiny one
            log_denom = jnp.log1p(self.fpl)
            log_w_pisn = -log_denom
            log_w_pl = safe_log(self.fpl) - log_denom

            log_dNdm = jnp.logaddexp(log_w_pisn + log_p_pisn, log_w_pl + log_p_pl)

        log_dNdm = jnp.where(m < self.mbh_min, -np.inf, log_dNdm)
        logwindow = mmin_log_smooth_turnon(m, delta_m=self.delta_m, mmin=self.mbh_min)
        return log_dNdm + logwindow
    
@dataclass
class LogDNDV(object):

    r"""
    Madau-Dickinson-like merger rate density over cosmic time:

    .. math::
        \frac{\mathrm{d} N}{\mathrm{d} V \mathrm{d} t} \propto \frac{\left( 1 + z \right)^\lambda}{1 + \left( \frac{1 + z}{1 + z_p} \right)^\kappa}
    """
    lam: object
    kappa: object
    zp: object
    zref: object = 0.001
    zmax: object = 20
    log_norm: object = 0.0

    def __post_init__(self):
        self.log_norm = -self(self.zref)

    def __call__(self, z):
        z = jnp.array(z)

        return jnp.where(z < self.zmax, self.lam*jnp.log1p(z) - jnp.log1p(((1+z)/(1+self.zp))**self.kappa) + self.log_norm, -np.inf)

@dataclass
class LogDNDMDQDV(object):
    r"""
    TODO: Document pairing function, arguments.
    """
    a: object
    b: object
    c: object
    mpisn: object
    mpisndot: object
    mbhmax: object
    sigma: object
    fpl: object
    beta: object
    lam: object
    kappa: object
    zp: object
    mp_low: object
    msigma_low: object
    flow: object
    mref: object = 30.0
    qref: object = 1.0
    zref: object = 0.001
    zmax: object = 20
    mbh_min: object = 3.0
    delta_m: object = 2.5
    mco_min: object= 4.0
    log_dndm: object = dataclasses.field(init=False)
    log_dndv: object = dataclasses.field(init=False)
    use_low_bump: object = True


    def __post_init__(self):
        self.log_dndm = LogDNDM(self.a, self.b, self.c, self.mpisn, self.mpisndot, self.mbhmax, self.sigma, self.fpl,
                                mp_low=self.mp_low, msigma_low=self.msigma_low, flow=self.flow, mref=self.mref, zmax=self.zmax, 
                                zref=self.zref, mbh_min=self.mbh_min,delta_m=self.delta_m, mco_min=self.mco_min,
                                use_low_bump=self.use_low_bump)
        self.log_dndv = LogDNDV(self.lam, self.kappa, self.zp, self.zref, zmax=self.zmax)
        self._normalize()

    def _normalize(self):
        self.log_norm = 0
        log_dN_ref = self(self.mref, self.qref, self.zref)

        # Want m_1 dN/d(m1)d(q)d(V)d(t) == 1 at reference for normalization (then the `R` parameter is the fitted dN/d(m1)d(q)d(V)d(t) at reference)
        self.log_norm = jnp.log(self.mref) + log_dN_ref

    def __call__(self, m1, q, z):
        m1 = jnp.array(m1)
        q = jnp.array(q)
        z = jnp.array(z)

        m2 = q*m1
        mt = m1+m2
        return self.log_dndm(m1, z) + self.log_dndm(m2, z) + self.beta*jnp.log(mt/(self.mref*(1 + self.qref))) + jnp.log(m1) + self.log_dndv(z) - self.log_norm

@dataclass
class FlatwCDMCosmology(object):
    """
    Function-like object representing a flat w-CDM cosmology.
    """
    h: object
    Om: object
    w: object
    zmax: object = 20.0
    ninterp: object = 1024
    zinterp: object = dataclasses.field(init=False)
    dcinterp: object = dataclasses.field(init=False)
    dlinterp: object = dataclasses.field(init=False)
    ddlinterp: object = dataclasses.field(init=False)
    vcinterp: object = dataclasses.field(init=False)
    dvcinterp: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.zinterp = jnp.expm1(jnp.linspace(np.log(1), jnp.log(1+self.zmax), self.ninterp))
        self.dcinterp = self.dH*jnp_cumtrapz(1/self.E(self.zinterp), self.zinterp)
        self.dlinterp = self.dcinterp*(1+self.zinterp)
        self.ddlinterp = self.dcinterp + self.dH*(1+self.zinterp)/self.E(self.zinterp)
        self.vcinterp = 4/3*np.pi*self.dcinterp*self.dcinterp*self.dcinterp
        self.dvcinterp = 4*np.pi*jnp.square(self.dcinterp)*self.dH/self.E(self.zinterp)
        self.dlinterp_dimless = self.dlinterp / self.dH   # (c/H0)*f(z,Om,w) → just f(z,Om,w)
        self.dcinterp_dimless = self.dcinterp / self.dH

    @property
    def dH(self):
        return 2.99792 / self.h
    
    @property
    def Ol(self):
        return 1-self.Om
    
    @property
    def om(self):
        return self.Om*jnp.square(self.h)
    
    @property
    def ol(self):
        return self.Ol*jnp.square(self.h)
    
    def E(self, z):
        opz = 1 + z
        opz3 = opz*opz*opz
        return jnp.sqrt(self.Om*opz3 + (1-self.Om)*opz**(3*(1 + self.w)))

    def dC(self, z):
        return jnp.interp(z, self.zinterp, self.dcinterp)
    def dL(self, z):
        return jnp.interp(z, self.zinterp, self.dlinterp)
    def VC(self, z):
        return jnp.interp(z, self.zinterp, self.vcinterp)
    def dVCdz(self, z):
        return jnp.interp(z, self.zinterp, self.dvcinterp)
    
    def ddL_dz(self, z):
        return jnp.interp(z, self.zinterp, self.ddlinterp)

    def z_of_dC(self, dC):
        return jnp.interp(dC / self.dH, self.dcinterp_dimless, self.zinterp)
    def z_of_dL(self, dL):
        return jnp.interp(dL / self.dH, self.dlinterp_dimless, self.zinterp)

coords = {
    'm_grid': np.exp(np.linspace(np.log(1), np.log(450), 128)),
    'q_grid': np.linspace(0, 1, 129)[1:],
    'z_grid': np.expm1(np.linspace(np.log1p(0), np.log1p(20), 128))
}

@partial(jax.jit, static_argnames=['use_low_bump'])
def get_deterministic_parameters(sample, use_low_bump=True):
    kappa = numpyro.deterministic('kappa', sample['lam'] + sample['dkappa'])
    mbhmax = numpyro.deterministic('mbhmax', sample['mpisn'] + sample['dmbhmax'])
 
    out = dict(kappa=kappa, mbhmax=mbhmax)
 
    if use_low_bump:
        if 'logit_flow' in sample:
            out['flow'] = numpyro.deterministic('flow', jax.nn.sigmoid(sample['logit_flow']))
        elif 'flow' in sample:
            out['flow'] = sample['flow']
        elif 'log_flow' in sample:
            out['flow'] = numpyro.deterministic('flow', jnp.exp(sample['log_flow']))
        else:
            raise KeyError("Need one of logit_flow, flow, or log_flow")
    
    if 'logit_fpl' in sample:
        out['fpl'] = numpyro.deterministic('fpl', jax.nn.sigmoid(sample['logit_fpl']))
    elif 'fpl' in sample:
        out['fpl'] = sample['fpl']
    elif 'log_fpl' in sample:
        out['fpl'] = numpyro.deterministic('fpl', jnp.exp(sample['log_fpl']))
    else:
        raise KeyError("Need one of logit_fpl, fpl, or log_fpl")
 
 
    return out
def log_smooth_neff_boundary(values, criteria):
        scaled_x = (values - criteria) / (0.05 * criteria)
        # Linear ramp below threshold: gradient is bounded to 1 everywhere,
        # preventing the step-size collapse that power-4 (gradient ~32000 at
        # scaled_x=-20) and power-10 (gradient ~5e12) caused.
        return jnp.minimum(0.0, scaled_x)

def build_population_model(sample, use_low_bump=True):
    return LogDNDMDQDV(
        a=sample['a'], b=sample['b'], c=sample['c'], mpisn=sample['mpisn'], mpisndot=sample['mpisndot'],
        mbhmax=sample['mbhmax'], sigma=sample['sigma'], fpl=sample['fpl'],
        beta=sample['beta'], lam=sample['lam'], kappa=sample['kappa'], zp=sample['zp'],
        zmax=sample['zmax'], mbh_min=sample['mbh_min'], delta_m=sample['delta_m'],
        mp_low=sample.get('mp_low', 1.0), msigma_low=sample.get('msigma_low', 1.0), flow=sample.get('flow', 0.0), use_low_bump=use_low_bump,
        #dummy values for mp_low to prevent errors when the bump is turned off, not in use
        )
#H_GRID = jnp.linspace(0.60, .8, 50)
    
def pop_cosmo_model(m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel, Ndraw, priors=None, use_low_bump=True):
    """
    Ndraw is # of events in the injection samples used to estimate the selection function
    """
    m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel = map(jnp.array, (m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel))

    log_pdraw_sel = jnp.log(pdraw_sel)
    nobs = m1s_det.shape[0]
    
    nsamp = m1s_det.shape[1]

    nsel = m1s_det_sel.shape[0]

    sample = sample_parameters_from_dict(priors)
    deterministic_parameters = get_deterministic_parameters(sample, use_low_bump=use_low_bump)
    sample.update(deterministic_parameters, ) #sample from hyperparameters, set up cosmology (cosmo) and population model (dN)

    cosmo = FlatwCDMCosmology(sample['h'], sample['Om'], sample['w'], zmax=sample['zmax'])
    log_dN = build_population_model(sample, use_low_bump=use_low_bump) 
  
    zs = cosmo.z_of_dL(dls)
    m1s = m1s_det / (1 + zs) # convert to source-frame masses

    # event-wise log weight for deteced events = log of pop density (model's differential merger rate density) 
    #      + Jacobian from source to detector frame 
    #      + cosmology factors (for volume element, D_L to z) + log_pdraw (corrects for non-uniform prior over parameter space)
    log_wts = log_dN(m1s, qs, zs)  - log_pdraw -2*jnp.log1p(zs) - jnp.log(cosmo.ddL_dz(zs)) + jnp.log(cosmo.dVCdz(zs)) 
    # dLdz/1+z is to deal with pdraw being in detector frame mass, dL
    log_like_per_event = jss.logsumexp(log_wts, axis=1) - jnp.log(nsamp)  # shape (nobs,)
    log_like_per_event = jnp.nan_to_num(log_like_per_event, nan=0, posinf=1e30, neginf=-1e30)
    _ = numpyro.deterministic("loglik_array_dim", log_like_per_event)

    log_like = jnp.sum(log_like_per_event)
    _ = numpyro.factor('loglike', log_like)

    zs_sel = cosmo.z_of_dL(dls_sel) 
    m1s_sel = m1s_det_sel / (1 + zs_sel)

    # now get weights for injected events as with log_wts
    log_sel_wts =  log_dN(m1s_sel, qs_sel, zs_sel)  - log_pdraw_sel- 2*jnp.log1p(zs_sel) + jnp.log(cosmo.dVCdz(zs_sel)) - jnp.log(cosmo.ddL_dz(zs_sel))
    #make sure that m1s_sel is in source frame, and pdraw_sel is in det mass frame
    log_mu_sel =  jss.logsumexp(log_sel_wts) - jnp.log(Ndraw)
    _ = numpyro.factor('selfactor', jnp.nan_to_num(jnp.nan_to_num(-nobs*log_mu_sel, nan=-np.inf))) #fix the np.inf if this is throwing the errors during inittialization

    log_mu2 = jss.logsumexp(2*log_sel_wts) - 2*jnp.log(Ndraw)
    log_s2 = log_mu2 + jnp.log1p(-jnp.exp(2*log_mu_sel - jnp.log(Ndraw) - log_mu2))

    #N eff cuts
    neff = jnp.exp(2 * jss.logsumexp(log_wts, axis=1) - jss.logsumexp(2 * log_wts, axis=1))
    min_neff = jnp.min(neff)
    numpyro.deterministic("neff", neff)
    numpyro.factor("neff_criteria",jnp.nan_to_num(log_smooth_neff_boundary(min_neff, nobs),nan=0, neginf=-1e30, posinf=1e30),)

    neff_sel = jnp.exp(2 * log_mu_sel - log_s2)
    numpyro.deterministic("neff_sel", neff_sel)
    numpyro.factor("neff_sel_criteria",jnp.nan_to_num(log_smooth_neff_boundary(neff_sel, 4 * nobs),nan=0, neginf=-1e30, posinf=1e30), )
    mu_sel = jnp.exp(log_mu_sel)

    R_unit = numpyro.sample('R_unit', dist.Normal(0, 1))
    R = numpyro.deterministic('R', nobs/mu_sel + jnp.sqrt(nobs)/mu_sel*R_unit)

    # differential merger rate with respect to mass, at fixed reference values of mass ratio, z
    _ = numpyro.deterministic('mdNdmdVdt_fixed_qz', coords['m_grid']*R*jnp.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)))

    # now varying q at fixed mass, z and then varying z at fixed mass, q (redshift evolution of merger rate)
    _ = numpyro.deterministic('dNdqdVdt_fixed_mz', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)))
    _ = numpyro.deterministic('dNdVdt_fixed_mq', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])))

    # dimensionless Hubble parameter at z
    _ = numpyro.deterministic('hz', cosmo.h*cosmo.E(coords['z_grid']))
"""LVK-style control mass model: GWTC-5 Default "PowerLaw + 2 Peaks"
(arXiv 2605.27226), ignoring spins.

The source-frame primary-mass distribution is a mixture of a broken power law
(continuous at the break) and two truncated Gaussian peaks, multiplied by the
low-mass smoothing window S(m | mmin, delta_m); the default pairing function
is p(q | m1) propto q^beta S(q m1) with its m1-dependent normalization
included explicitly (it does not cancel into R).  A second, static ``pairing
= "mt"`` mode instead pairs the LVK mass function the way the original PISN
model does (f(m1) f(m2) with a total-mass power law; see LogDNDMDQDV_LVK),
which disentangles mass-model effects from pairing-function effects across
run families.  Everything else -- cosmology,
Madau-Dickinson redshift evolution, detector->source conversion, selection
integral, n_eff guards, R sampling, float32 recentering -- is identical to
``intensity_models_fast`` and imported from there unchanged.

The intensity keeps the same point-normalization convention as
``LogDNDMDQDV._normalize``: m1 * dN/dm1 dq dV dt == 1 at (mref=30, qref=1,
zref=0.001), so R keeps its exact meaning across model families.  The overall
normalization deficit from smoothing is a parameter-dependent constant and
cancels exactly in the R-marginalized likelihood prod_i mu_i / mu_sel^nobs.

The single numerical approximation specific to this model is the tabulated
q-normalization Zq(m1) (``LogQNorm``); one instance per likelihood call is
shared by the event term, the selection term, and the log_norm reference
evaluation, per notes/2026-08-08-tabulated-selection-consistency.md.
"""
import dataclasses
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, ndtr
import numpy as np
import numpyro
import numpyro.distributions as dist
from utils import sample_parameters_from_dict

from intensity_models_fast import (
    FlatwCDMCosmology, LogDNDV, LogQNorm, coords, log_smooth_neff_boundary,
    mmin_log_smooth_turnon, safe_log, _logsumexp_and_neff,
    _LOG_ZERO_FLOOR,
)

_LOG_2PI = float(np.log(2 * np.pi))


def log_powerlaw_norm(a, log_lo, log_hi):
    """Log of the normalization constant of m^-a on [lo, hi],
    (1-a) / (hi^(1-a) - lo^(1-a)), stable through a = 1.

    Written in log space: log|1-a| - log|hi^(1-a) - lo^(1-a)| with the span
    via max + log1p(-exp(-.)), which avoids the float32 cancellation of the
    direct difference near a = 1.  At a ~= 1 exactly it switches to the
    analytic limit 1/log(hi/lo); the unselected branch is evaluated at a
    dummy value so no inf/nan (or their gradients) are ever formed.
    """
    one_m_a = 1.0 - a
    close = jnp.abs(one_m_a) < 1e-6
    safe = jnp.where(close, 1.0, one_m_a)
    u = safe * log_hi
    v = safe * log_lo
    log_span = jnp.maximum(u, v) + jnp.log1p(-jnp.exp(-jnp.abs(u - v)))
    return jnp.where(close, -jnp.log(log_hi - log_lo),
                     jnp.log(jnp.abs(safe)) - log_span)


def log_mass_mixture(m, log_m, alpha_1, alpha_2, mbreak, mpp_1, sigpp_1,
                     mpp_2, sigpp_2, f_peaks, f_p1, mmin, mmax):
    """log g(m): broken power law + two truncated Gaussians, each component
    unit-area on [mmin, mmax]; -inf outside.  No smoothing window here."""
    log_mmin = jnp.log(mmin)
    log_mmax = jnp.log(mmax)
    log_mbreak = jnp.log(mbreak)

    # Broken power law, continuous at mbreak, normalized on [mmin, mmax].
    norm_low = log_powerlaw_norm(alpha_1, log_mmin, log_mbreak)
    norm_high = log_powerlaw_norm(alpha_2, log_mbreak, log_mmax)
    # Continuity correction C = PL_high(mbreak) / PL_low(mbreak): the low
    # segment is scaled by C so the density is continuous, then the total
    # area 1 + C is divided out.
    log_C = (-alpha_2 * log_mbreak + norm_high) - (-alpha_1 * log_mbreak + norm_low)
    log_bpl = (jnp.where(m < mbreak,
                         -alpha_1 * log_m + norm_low + log_C,
                         -alpha_2 * log_m + norm_high)
               - jnp.logaddexp(0.0, log_C))

    # Gaussian peaks truncated and renormalized on [mmin, mmax].
    trunc_1 = ndtr((mmax - mpp_1) / sigpp_1) - ndtr((mmin - mpp_1) / sigpp_1)
    log_peak_1 = (-0.5 * jnp.square((m - mpp_1) / sigpp_1) - 0.5 * _LOG_2PI
                  - jnp.log(sigpp_1) - safe_log(trunc_1))
    trunc_2 = ndtr((mmax - mpp_2) / sigpp_2) - ndtr((mmin - mpp_2) / sigpp_2)
    log_peak_2 = (-0.5 * jnp.square((m - mpp_2) / sigpp_2) - 0.5 * _LOG_2PI
                  - jnp.log(sigpp_2) - safe_log(trunc_2))

    # Stick-breaking mixture weights (matches the popsummary lam_0 / lam_1).
    log_g = jnp.logaddexp(
        jnp.logaddexp(safe_log(1.0 - f_peaks) + log_bpl,
                      safe_log(f_peaks * f_p1) + log_peak_1),
        safe_log(f_peaks * (1.0 - f_p1)) + log_peak_2)

    inside = jnp.logical_and(m >= mmin, m <= mmax)
    return jnp.where(inside, log_g, -jnp.inf)


def log_smoothed_mixture(m, log_m, alpha_1, alpha_2, mbreak, mpp_1, sigpp_1,
                         mpp_2, sigpp_2, f_peaks, f_p1, mmin, mmax, delta_m):
    """log f(m) = log g(m) + log S(m | mmin, delta_m)."""
    return (log_mass_mixture(m, log_m, alpha_1, alpha_2, mbreak, mpp_1, sigpp_1,
                             mpp_2, sigpp_2, f_peaks, f_p1, mmin, mmax)
            + mmin_log_smooth_turnon(m, delta_m, mmin))


# LogQNorm (the tabulated q-normalization Zq(m1)) moved to
# intensity_models_fast, where the PISN model's pairing="lvk" mode also needs
# it; imported above so this module's API is unchanged.


@dataclass
class LogDNDMDQDV_LVK(object):
    """LVK PowerLaw + 2 Peaks intensity.

    Two pairing conventions, selected by the STATIC ``pairing`` flag (a plain
    Python string, never traced -- like ``tail_anchor`` in the fast module):

    pairing = "lvk" (default, the LVK convention):
        log dN/dm1 dq dV dt = log f(m1) + beta log q + log S(q m1)
                              - log Zq(m1) + log dNdV(z) - log_norm

    pairing = "mt" (the original PISN model's total-mass pairing, see
    ``intensity_models_fast.LogDNDMDQDV.call_from_logs``): both masses drawn
    from the same single-mass function f, paired by total mass, with the
    m2 -> q Jacobian + log(m1) and NO m1-dependent q normalization (the
    constant normalization is absorbed into R by the point normalization):
        log dN/dm1 dq dV dt = log f(m1) + log f(m2)
                              + beta log(mt / (mref (1 + qref)))
                              + log m1 + log dNdV(z) - log_norm
    with m2 = q m1, mt = m1 + m2.  log f already carries S(m) and the
    [mmin, mmax] support, so S(m2) comes for free (and m2 <= m1 keeps the
    mmax cut on m2 from ever binding).  Here beta is the TOTAL-MASS exponent,
    not the q exponent.
    """
    alpha_1: object
    alpha_2: object
    mbreak: object
    mpp_1: object
    sigpp_1: object
    mpp_2: object
    sigpp_2: object
    f_peaks: object
    f_p1: object
    beta: object
    lam: object
    kappa: object
    zp: object
    mmin: object
    mmax: object
    delta_m: object
    mref: object = 30.0
    qref: object = 1.0
    zref: object = 0.001
    zmax: object = 20
    pairing: str = "lvk"
    n_m1_qnorm: int = 256
    n_q_qnorm: int = 256
    log_dndv: object = dataclasses.field(init=False)
    log_qnorm: object = dataclasses.field(init=False)

    def __post_init__(self):
        if self.pairing not in ("lvk", "mt"):
            raise ValueError(f"unknown pairing: {self.pairing!r} "
                             f"(known: 'lvk', 'mt')")
        self.log_dndv = LogDNDV(self.lam, self.kappa, self.zp, self.zref,
                                zmax=self.zmax)
        # The Zq(m1) table only exists in the LVK pairing; the mt pairing has
        # no m1-dependent q normalization (constants are absorbed into R).
        self.log_qnorm = (LogQNorm(self.beta, self.mmin, self.delta_m,
                                   n_m1=self.n_m1_qnorm, n_q=self.n_q_qnorm)
                          if self.pairing == "lvk" else None)
        self._normalize()

    def _normalize(self):
        # Same convention as LogDNDMDQDV: m1 dN/dm1 dq dV dt == 1 at
        # (mref, qref, zref), so R is the physical rate density there.  The
        # reference evaluation goes through the same Zq table as the data.
        self.log_norm = 0
        log_dN_ref = self(self.mref, self.qref, self.zref)
        self.log_norm = jnp.log(self.mref) + log_dN_ref

    def log_dndm(self, m, log_m=None):
        """log f(m): the smoothed single-mass mixture (no pairing, no z)."""
        m = jnp.asarray(m)
        if log_m is None:
            log_m = jnp.log(m)
        return log_smoothed_mixture(
            m, log_m, self.alpha_1, self.alpha_2, self.mbreak,
            self.mpp_1, self.sigpp_1, self.mpp_2, self.sigpp_2,
            self.f_peaks, self.f_p1, self.mmin, self.mmax, self.delta_m)

    def __call__(self, m1, q, z):
        m1, q, z = jnp.broadcast_arrays(
            jnp.atleast_1d(jnp.asarray(m1)), jnp.atleast_1d(jnp.asarray(q)),
            jnp.atleast_1d(jnp.asarray(z)))
        return self.call_from_logs(m1, jnp.log(m1), jnp.log(q), z, jnp.log1p(z))

    def call_from_logs(self, m1, log_m1, log_q, z, log1p_z):
        """Same value as ``__call__`` but takes the logs the caller already has."""
        m2 = m1 * jnp.exp(log_q)
        if self.pairing == "lvk":
            # No + log(m1) Jacobian here: unlike the pairing construction in
            # LogDNDMDQDV, p(m1) q^beta already is a density in (m1, q).
            return (self.log_dndm(m1, log_m1)
                    + self.beta * log_q
                    + mmin_log_smooth_turnon(m2, self.delta_m, self.mmin)
                    - self.log_qnorm.log_Zq_from_log(log_m1)
                    + self.log_dndv.from_log1p(log1p_z)
                    - self.log_norm)
        # "mt": total-mass pairing of intensity_models_fast.LogDNDMDQDV --
        # f(m1) f(m2) with beta on mt, + log(m1) is the m2 -> q Jacobian;
        # log_dndm carries S(m) and the support cuts for both masses.
        log_m2 = log_m1 + log_q
        mt = m1 + m2
        return (self.log_dndm(m1, log_m1)
                + self.log_dndm(m2, log_m2)
                + self.beta * jnp.log(mt / (self.mref * (1 + self.qref)))
                + log_m1
                + self.log_dndv.from_log1p(log1p_z)
                - self.log_norm)


def get_deterministic_parameters(sample):
    """Derived parameters for the LVK model.  Tolerant: each rule fires only
    when its source keys are present, so a prior may sample kappa or Om
    directly instead.  No PISN-model rules."""
    out = {}
    if 'log_h' in sample and 'h' not in sample:
        out['h'] = numpyro.deterministic('h', jnp.exp(sample['log_h']))
    h = out.get('h', sample.get('h'))

    if 'dkappa' in sample and 'kappa' not in sample:
        out['kappa'] = numpyro.deterministic('kappa', sample['lam'] + sample['dkappa'])

    if 'Omh2' in sample and 'Om' not in sample:
        out['Om'] = numpyro.deterministic('Om', sample['Omh2'] / jnp.square(h))

    # Mixture fractions in the flat (Dirichlet-like) coordinates, for
    # downstream plotting; the stick-breaking (f_peaks, f_p1) stay primary.
    if 'f_peaks' in sample and 'f_p1' in sample:
        out['frac_bpl'] = numpyro.deterministic('frac_bpl', 1.0 - sample['f_peaks'])
        out['frac_p1'] = numpyro.deterministic('frac_p1', sample['f_peaks'] * sample['f_p1'])
        out['frac_p2'] = numpyro.deterministic('frac_p2', sample['f_peaks'] * (1.0 - sample['f_p1']))
    return out


def build_population_model(sample, pairing="lvk", **_ignored):
    """Construct the LVK intensity from a parameter dict.  ``pairing`` selects
    the static pairing convention ("lvk" or "mt", see LogDNDMDQDV_LVK).  Extra
    kwargs (use_low_bump, smooth_tail_edge, ...) are accepted and ignored so
    call sites shared with the PISN model stay uniform."""
    return LogDNDMDQDV_LVK(
        alpha_1=sample['alpha_1'], alpha_2=sample['alpha_2'], mbreak=sample['mbreak'],
        mpp_1=sample['mpp_1'], sigpp_1=sample['sigpp_1'],
        mpp_2=sample['mpp_2'], sigpp_2=sample['sigpp_2'],
        f_peaks=sample['f_peaks'], f_p1=sample['f_p1'], beta=sample['beta'],
        lam=sample['lam'], kappa=sample['kappa'], zp=sample['zp'],
        mmin=sample['mmin'], mmax=sample['mmax'], delta_m=sample['delta_m'],
        zmax=sample['zmax'], pairing=pairing)


def map_truths_to_prior_coords(truths, prior):
    """Map canonical truth values into the prior's sampled coordinates."""
    tv = dict(truths)
    if 'log_h' in prior and 'h' in tv:
        tv['log_h'] = jnp.log(tv['h'])
    return tv


def pop_cosmo_model(m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel,
                    pdraw_sel, Ndraw, priors=None, store_per_event=False,
                    neff_criterion=None, neff_penalty="mc_variance",
                    mc_variance_budget=5.0, loglike_ref=None,
                    log_mu_sel_ref=None, log_pdraw_sel_scale=0.0,
                    pairing="lvk"):
    """LVK-control counterpart of ``intensity_models_fast.pop_cosmo_model``.

    Data conventions are identical (m1 detector-frame, dl in Gpc, event
    ``log_pdraw`` already log, selection ``pdraw_sel`` linear); the body is
    the fast module's direct (non-tabulated) branch plus its shared tail --
    the closed-form PL+2Peaks needs no mass tabulation.  See the fast module
    for the meaning of the keyword arguments.  ``pairing`` is the static
    pairing convention ("lvk" or "mt", see LogDNDMDQDV_LVK).
    """
    (m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel) = map(
        jnp.asarray,
        (m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel))

    # Numerical scale only: does not mutate the caller's pdraw_sel array.
    log_pdraw_sel = jnp.log(pdraw_sel) + log_pdraw_sel_scale
    nobs = m1s_det.shape[0]
    nsamp = m1s_det.shape[1]
    if neff_criterion is None:
        neff_criterion = nobs

    log_m1s_det = jnp.log(m1s_det)
    log_qs = jnp.log(qs)
    log_dls = jnp.log(dls)
    log_m1s_det_sel = jnp.log(m1s_det_sel)
    log_qs_sel = jnp.log(qs_sel)
    log_dls_sel = jnp.log(dls_sel)

    sample = sample_parameters_from_dict(priors)
    sample.update(get_deterministic_parameters(sample))

    cosmo = FlatwCDMCosmology(sample['h'], sample['Om'], sample['w'],
                              zmax=sample['zmax'])
    log_dN = build_population_model(sample, pairing=pairing)

    # detected events
    log1p_zs, J = cosmo.z_and_log_jacobian(log_dls)
    opz = jnp.exp(log1p_zs)
    zs = opz - 1.0
    m1s = m1s_det / opz
    log_m1s = log_m1s_det - log1p_zs

    log_wts = log_dN.call_from_logs(m1s, log_m1s, log_qs, zs, log1p_zs) - log_pdraw + J

    # selection samples
    log1p_zs_sel, J_sel = cosmo.z_and_log_jacobian(log_dls_sel)
    opz_sel = jnp.exp(log1p_zs_sel)
    zs_sel = opz_sel - 1.0
    m1s_sel = m1s_det_sel / opz_sel
    log_m1s_sel = log_m1s_det_sel - log1p_zs_sel

    log_sel_wts = (log_dN.call_from_logs(m1s_sel, log_m1s_sel, log_qs_sel,
                                         zs_sel, log1p_zs_sel)
                   - log_pdraw_sel + J_sel)

    # ---- shared tail: identical to intensity_models_fast ------------------
    lse1, lse2, neff = _logsumexp_and_neff(log_wts, axis=1)
    log_like_per_event = lse1 - jnp.log(nsamp)
    if store_per_event:
        _ = numpyro.deterministic("loglik_array_dim", log_like_per_event)

    if loglike_ref is not None:
        log_like = jnp.sum(log_like_per_event - jnp.asarray(loglike_ref))
    else:
        log_like = jnp.sum(log_like_per_event)
    _ = numpyro.factor('loglike', log_like)

    lse_sel, lse2_sel, _ = _logsumexp_and_neff(log_sel_wts[None, :], axis=1)
    log_mu_sel_scaled = jnp.squeeze(lse_sel) - jnp.log(Ndraw)
    log_mu_sel = log_mu_sel_scaled + log_pdraw_sel_scale
    numpyro.deterministic('log_mu_sel', log_mu_sel)
    sel_dead = jnp.squeeze(lse_sel) <= _LOG_ZERO_FLOOR
    if log_mu_sel_ref is not None:
        sel_log_factor = -nobs * (log_mu_sel_scaled - log_mu_sel_ref)
    else:
        sel_log_factor = -nobs * log_mu_sel_scaled
    _ = numpyro.factor('selfactor', jnp.where(sel_dead, _LOG_ZERO_FLOOR, sel_log_factor))

    log_mu2 = jnp.squeeze(lse2_sel) - 2 * jnp.log(Ndraw)
    x = 2 * log_mu_sel_scaled - jnp.log(Ndraw) - log_mu2
    log_s2 = log_mu2 + jnp.log(-jnp.expm1(jnp.minimum(x, -1e-7)))

    min_neff = jnp.min(neff)
    mc_var = jnp.sum(1.0 / jnp.clip(neff, 1.0, None))
    if store_per_event:
        numpyro.deterministic("neff", neff)
    numpyro.deterministic("min_neff", min_neff)
    numpyro.deterministic("mc_var_loglike", mc_var)
    if neff_penalty == "min_neff":
        numpyro.factor("neff_criteria", log_smooth_neff_boundary(min_neff, neff_criterion))
    elif neff_penalty == "mc_variance":
        numpyro.factor("neff_criteria",
                       -jax.nn.softplus((mc_var - mc_variance_budget) / (0.05 * mc_variance_budget)))
    elif neff_penalty not in (None, "none"):
        raise ValueError(f"unknown neff_penalty: {neff_penalty!r}")

    neff_sel = jnp.exp(2 * log_mu_sel_scaled - log_s2)
    numpyro.deterministic("neff_sel", neff_sel)
    if neff_penalty == "min_neff":
        numpyro.factor("neff_sel_criteria", log_smooth_neff_boundary(neff_sel, 4 * nobs))
    else:
        numpyro.factor("neff_sel_criteria",
                       -jax.nn.softplus((4 * nobs - neff_sel) / (0.05 * 4 * nobs)))

    mu_sel = jnp.exp(log_mu_sel)

    R_unit = numpyro.sample('R_unit', dist.Normal(0, 1))
    R = numpyro.deterministic('R', nobs / mu_sel + jnp.sqrt(nobs) / mu_sel * R_unit)

    _ = numpyro.deterministic('mdNdmdVdt_fixed_qz',
        coords['m_grid'] * R * jnp.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)),)
    _ = numpyro.deterministic('dNdqdVdt_fixed_mz',
        log_dN.mref * R * jnp.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)),)
    _ = numpyro.deterministic('dNdVdt_fixed_mq',
        log_dN.mref * R * jnp.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])),)
    _ = numpyro.deterministic('hz', cosmo.h * cosmo.E(coords['z_grid']))


def recentering_baselines(model_args, ref_params, rng_seed=0, max_dead_events=0,
                          **model_kwargs):
    """Per-event log likelihoods and log_mu_sel at a fixed reference point,
    for float32 recentering.  Identical contract to
    ``intensity_models_fast.recentering_baselines`` but traces this module's
    ``pop_cosmo_model``."""
    import numpyro.handlers as handlers

    model_kwargs = dict(model_kwargs)
    model_kwargs["store_per_event"] = True
    model_kwargs.pop("loglike_ref", None)
    model_kwargs.pop("log_mu_sel_ref", None)
    model_kwargs.pop("log_pdraw_sel_scale", None)
    ref_params = {k: jnp.asarray(v) for k, v in ref_params.items()}
    with handlers.seed(rng_seed=rng_seed), handlers.substitute(data=ref_params):
        tr = handlers.trace(pop_cosmo_model).get_trace(*model_args, **model_kwargs)

    loglike_ref = np.asarray(tr["loglik_array_dim"]["value"], dtype=np.float64)
    log_mu_sel_phys = float(np.asarray(tr["log_mu_sel"]["value"]))
    # Hard error on a bad reference point, mirroring
    # intensity_models_fast.recentering_baselines (see there for rationale).
    from intensity_models_fast import _ref_point_summary
    n_dead = int(np.sum(~(loglike_ref > 0.5 * _LOG_ZERO_FLOOR)))  # counts NaN too
    if n_dead > max_dead_events:
        raise RuntimeError(
            f"recentering_baselines: {n_dead} of {loglike_ref.shape[0]} event(s) "
            f"have (near-)zero likelihood at the recentering reference point "
            f"(log L <= {0.5 * _LOG_ZERO_FLOOR:.3g}); float32 recentering would "
            f"be useless and the sampler will stall.  Reference point: "
            f"{_ref_point_summary(tr)}.  Provide an explicit reference via a "
            f"[ref_params] section in the run-config ini (name = value lines "
            f"for the SAMPLED prior parameters, e.g. posterior medians of a "
            f"healthy run) instead of relying on the seed-{rng_seed} prior draw.")
    if not np.isfinite(log_mu_sel_phys) or log_mu_sel_phys <= 0.5 * _LOG_ZERO_FLOOR:
        raise RuntimeError(
            f"recentering_baselines: log_mu_sel = {log_mu_sel_phys} at the "
            f"recentering reference point is non-finite or floored; the "
            f"selection integral has no support there.  Reference point: "
            f"{_ref_point_summary(tr)}.  Provide an explicit reference via a "
            f"[ref_params] section in the run-config ini.")
    nobs = loglike_ref.shape[0]
    return dict(loglike_ref=loglike_ref, log_pdraw_sel_scale=log_mu_sel_phys,
                log_mu_sel_ref=0.0, log_mu_sel_phys_ref=log_mu_sel_phys,
                offset=float(loglike_ref.sum() - nobs * log_mu_sel_phys))

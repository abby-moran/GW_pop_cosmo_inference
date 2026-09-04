"""Validation of intensity_models_lvk (LVK 'PowerLaw + 2 Peaks' control model).

Checks, in order:
  1. mass pdf vs an independent float64 numpy/scipy reference (BPL continuity,
     truncated-Gaussian peaks, smoothing window, support edges, unit area)
  2. the tabulated q-normalization Zq(m1) vs a dense float64 trapezoid
  3. the selection-consistency identity of
     notes/2026-08-08-tabulated-selection-consistency.md: with the selection
     set equal to the flattened event samples, log_mu_sel + log(Ndraw) must
     equal logsumexp_i(loglike_i) + log(nsamp) -- the structural guard that
     every approximation (Zq table, dl table) hits the event and selection
     terms identically
  4. full model: finite potential and gradients at truth, clean trace,
     dead-event robustness
  5. float32 recentering: centered and raw potentials differ by exactly the
     reported offset; gradients unchanged
  6. the "mt" (PISN-style total-mass) pairing: algebraic recomposition of
     log_dN from the object's own pieces, the selection-consistency identity,
     finite potential/gradients, and bit-identity of pairing="lvk" with the
     default
  7. (with --mcmc) a short NUTS smoke run on both redshift parametrizations
  8. traced mmin/delta_m (the mminfree parametrization): finite d(log Zq)
     gradients across the prior box and finite potential/gradients for both
     pairings with mmin and delta_m sampled
  9. the redshift-evolution dispatch: a prior WITHOUT dkappa/kappa selects
     LogDNDVPowerLaw, one WITH dkappa (+ zp) selects the shared
     intensity_models_fast.LogDNDV bit-for-bit, both are point-normalized,
     and the misconfigurations (kappa without zp, zp without kappa) raise
 10. the Madau-Dickinson branch end to end: recorded kappa = lam + dkappa,
     the selection-consistency identity, and finite potential/gradients

Run from scripts/ (sys.path.append("../src/") below is CWD-relative, so
invoking this from scripts/testing_scripts/ will not find the src modules):
    uv run python testing_scripts/test_lvk_model.py            # checks 1-6, 8-10
    uv run python testing_scripts/test_lvk_model.py --mcmc 50  # + check 7
"""
import argparse
import dataclasses
import os
import sys
import tempfile

sys.path.append("../src/")

import numpy as np
from scipy.special import logsumexp as np_logsumexp
from scipy.stats import norm as scipy_norm
import jax
import jax.numpy as jnp

import intensity_models_fast as ifast
import intensity_models_lvk as ilvk
from utils import get_priors_from_file
from bench_model import make_synthetic_data, diagnose

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


LVK_PRIOR_TEXT = """h = 0.674
Om = 0.315
w = -1
alpha_1 = Uniform(-4.0, 12.0)
alpha_2 = Uniform(-4.0, 12.0)
mbreak = Uniform(20.0, 50.0)
mpp_1 = Uniform(5.0, 15.0)
sigpp_1 = Uniform(0.1, 10.0)
mpp_2 = Uniform(25.0, 100.0)
sigpp_2 = Uniform(0.1, 10.0)
f_peaks = Uniform(0.0, 1.0)
f_p1 = Uniform(0.0, 1.0)
beta = Uniform(-4.0, 7.0)
mmin = 4.5
mmax = 300.0
delta_m = 4.0
lam = TruncatedNormal(2.7, 2.0, low=-1.3, high=6.7)
zmax = 6.5
"""

# The same model with the Madau-Dickinson redshift sector (the
# lvk_gwtc5_control.prior coordinates): supplying dkappa/zp is what selects
# it -- there is no flag.
LVK_PRIOR_TEXT_MD = LVK_PRIOR_TEXT + """dkappa = TruncatedNormal(2.9, 2.0, low=1, high=6.9)
zp = TruncatedNormal(1.9, 1, low=0, high=3.9)
"""

# Roughly the GWTC-5 Default release posterior medians (see the plan notes).
LVK_TRUTH = dict(alpha_1=1.5, alpha_2=5.4, mbreak=37.0, mpp_1=9.9, sigpp_1=0.8,
                 mpp_2=32.0, sigpp_2=5.0, f_peaks=0.4, f_p1=0.55, beta=1.0,
                 lam=2.5)
LVK_TRUTH_MD = dict(LVK_TRUTH, dkappa=3.0, zp=1.9)
FIXED = dict(mmin=4.5, mmax=300.0, delta_m=4.0)


def build_lvk_prior(text=LVK_PRIOR_TEXT):
    fd, path = tempfile.mkstemp(suffix=".prior", prefix="lvk_test_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return get_priors_from_file(path)


def build_from_prior(prior, truth, **build_kwargs):
    """prior -> sample dict -> deterministics -> intensity: the run_inf.py
    path, inside a seed handler (the deterministic sites need one)."""
    import numpyro.handlers as handlers
    from utils import sample_parameters_from_dict
    values = {k: jnp.asarray(v) for k, v in truth.items()}
    with handlers.seed(rng_seed=0), handlers.substitute(data=values):
        sample = sample_parameters_from_dict(prior)
        sample.update(ilvk.get_deterministic_parameters(sample))
        return sample, ilvk.build_population_model(sample, **build_kwargs)


# ---------------------------------------------------------------------------
# Independent float64 reference implementation (numpy/scipy only)
# ---------------------------------------------------------------------------
def ref_log_S(m, delta_m, mmin):
    """Reference of mmin_log_smooth_turnon, same clipping conventions."""
    m = np.asarray(m, np.float64)
    sm = np.clip(np.nan_to_num((m - mmin) / delta_m, nan=0.0), 1e-6, 1 - 1e-6)
    expnt = np.minimum(1.0 / sm - 1.0 / (1.0 - sm), 87.0)
    out = -np.logaddexp(0.0, expnt)          # log logistic(-e)
    return np.where(m < mmin, -np.inf, out)


def ref_log_g(m, p):
    """Reference of log_mass_mixture (no smoothing window)."""
    m = np.asarray(m, np.float64)
    a1, a2, mb = p["alpha_1"], p["alpha_2"], p["mbreak"]
    mmin, mmax = p["mmin"], p["mmax"]

    def pl_norm(a, lo, hi):
        if abs(1.0 - a) < 1e-9:
            return 1.0 / np.log(hi / lo)
        return (1.0 - a) / (hi ** (1.0 - a) - lo ** (1.0 - a))

    N1 = pl_norm(a1, mmin, mb)
    N2 = pl_norm(a2, mb, mmax)
    C = (mb ** -a2 * N2) / (mb ** -a1 * N1)
    bpl = np.where(m < mb, m ** -a1 * N1 * C, m ** -a2 * N2) / (1.0 + C)

    def peak(mu, sig):
        z = scipy_norm.pdf(m, mu, sig)
        return z / (scipy_norm.cdf(mmax, mu, sig) - scipy_norm.cdf(mmin, mu, sig))

    g = ((1 - p["f_peaks"]) * bpl
         + p["f_peaks"] * p["f_p1"] * peak(p["mpp_1"], p["sigpp_1"])
         + p["f_peaks"] * (1 - p["f_p1"]) * peak(p["mpp_2"], p["sigpp_2"]))
    with np.errstate(divide="ignore"):
        out = np.log(g)
    return np.where((m >= mmin) & (m <= mmax), out, -np.inf)


def jax_log_f(m, p):
    m = jnp.asarray(m)
    return np.asarray(ilvk.log_smoothed_mixture(
        m, jnp.log(m), p["alpha_1"], p["alpha_2"], p["mbreak"],
        p["mpp_1"], p["sigpp_1"], p["mpp_2"], p["sigpp_2"],
        p["f_peaks"], p["f_p1"], p["mmin"], p["mmax"], p["delta_m"]))


def test_mass_pdf_reference():
    print("\n=== 1. mass pdf vs float64 reference ===")
    p = {**LVK_TRUTH, **FIXED}

    m = np.exp(np.linspace(np.log(3.0), np.log(320.0), 4001))
    ref = ref_log_g(m, p) + ref_log_S(m, p["delta_m"], p["mmin"])
    got = jax_log_f(m, p)

    check("inf pattern matches (support edges)",
          bool(np.all(np.isfinite(ref) == np.isfinite(got))),
          f"{int(np.sum(np.isfinite(ref) != np.isfinite(got)))} mismatches")

    live = np.isfinite(ref) & (ref > -60.0)
    d = np.abs(got[live] - ref[live])
    check("log f(m1) matches reference where log f > -60",
          bool(np.max(d) < 1e-3), f"max |dlog| = {np.max(d):.2e}")

    # BPL continuity at the break (peaks switched ~off).
    p_bpl = {**p, "f_peaks": 1e-10}
    mb = p["mbreak"]
    lo, hi = (float(x) for x in np.asarray(
        jax_log_f(np.array([mb * (1 - 1e-5), mb * (1 + 1e-5)]), p_bpl)))
    check("BPL continuous at mbreak", abs(hi - lo) < 1e-3,
          f"|dlog| = {abs(hi - lo):.2e}")

    # Unit area of the unsmoothed mixture (float64 reference construction).
    mg = np.linspace(p["mmin"], p["mmax"], 400001)
    area = np.trapezoid(np.exp(ref_log_g(mg, p)), mg)
    check("reference mixture has unit area on [mmin, mmax]",
          abs(area - 1.0) < 1e-5, f"integral = {area:.8f}")

    # ... and the jax mixture agrees with that reference (subsampled grid).
    mg_s = mg[::100]
    ref_g = ref_log_g(mg_s, p)
    got_g = np.asarray(ilvk.log_mass_mixture(
        jnp.asarray(mg_s), jnp.log(jnp.asarray(mg_s)), p["alpha_1"], p["alpha_2"],
        p["mbreak"], p["mpp_1"], p["sigpp_1"], p["mpp_2"], p["sigpp_2"],
        p["f_peaks"], p["f_p1"], p["mmin"], p["mmax"]))
    live = ref_g > -60.0
    d = np.abs(got_g[live] - ref_g[live])
    check("log g(m1) matches reference on [mmin, mmax]",
          bool(np.max(d) < 1e-3), f"max |dlog| = {np.max(d):.2e}")

    # Smoothing window shape over [mmin, mmin + delta_m].
    mw = np.linspace(p["mmin"] * (1 + 1e-6), p["mmin"] + p["delta_m"], 200)
    dS = np.abs(np.asarray(ref_log_S(mw, p["delta_m"], p["mmin"]))
                - np.asarray(ilvk.mmin_log_smooth_turnon(
                    jnp.asarray(mw), p["delta_m"], p["mmin"])))
    check("smoothing window matches across the turn-on",
          bool(np.max(dS) < 1e-3), f"max |dlog S| = {np.max(dS):.2e}")


def test_q_norm():
    print("\n=== 2. Zq(m1) table vs dense float64 trapezoid ===")
    p = FIXED
    m1_cases = [5.0, 6.0, 10.0, 35.0, 80.0, 250.0]
    # near mmin the table's log-m1 interpolation is coarsest; loosen there
    tols = {5.0: 0.1, 6.0: 0.02}
    for beta in (-3.0, 0.0, 1.0, 3.0, 6.0):
        qn = ilvk.LogQNorm(beta, p["mmin"], p["delta_m"])
        worst = (0.0, None)
        ok = True
        for m1 in m1_cases:
            qlo = min(p["mmin"] / m1, 1 - 1e-6)
            q = np.linspace(qlo, 1.0, 65537)
            integ = q ** beta * np.exp(ref_log_S(q * m1, p["delta_m"], p["mmin"]))
            ref = np.log(np.trapezoid(integ, q))
            got = float(np.asarray(qn.log_Zq_from_log(jnp.log(m1))))
            err = abs(got - ref)
            tol = tols.get(m1, 5e-3)
            if err > tol:
                ok = False
            if err > worst[0]:
                worst = (err, m1)
        check(f"log Zq, beta = {beta:+.1f}", ok,
              f"worst |dlog| = {worst[0]:.2e} at m1 = {worst[1]}")


def _trace_at(model_args, values, **model_kwargs):
    import numpyro.handlers as handlers
    values = {k: jnp.asarray(v) for k, v in values.items()}
    with handlers.seed(rng_seed=0), handlers.substitute(data=values):
        return handlers.trace(ilvk.pop_cosmo_model).get_trace(*model_args, **model_kwargs)


def test_selection_identity(prior):
    print("\n=== 3. selection-consistency identity (sel := flattened events) ===")
    data = make_synthetic_data(nobs=32, nsamp=64, nsel=100)
    nobs, nsamp = data["m1s_det"].shape
    nsel = nobs * nsamp
    model_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det"].reshape(-1), data["qs"].reshape(-1),
        data["dls"].reshape(-1), np.exp(data["log_pdraw"].reshape(-1)),
        float(nsel), prior)

    corner = {**LVK_TRUTH, "sigpp_1": 0.12, "alpha_2": 11.0, "beta": 6.5}
    for label, values in (("truth", LVK_TRUTH), ("sharp corner", corner)):
        tr = _trace_at(model_args, values, store_per_event=True)
        loglike = np.asarray(tr["loglik_array_dim"]["value"], np.float64)
        log_mu_sel = float(np.asarray(tr["log_mu_sel"]["value"]))
        lhs = log_mu_sel + np.log(nsel)
        rhs = np_logsumexp(loglike) + np.log(nsamp)
        check(f"log_mu_sel identity at {label}", abs(lhs - rhs) < 5e-4,
              f"|d| = {abs(lhs - rhs):.2e} nats")


def _init_model(model_args, truth, **model_kwargs):
    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value
    truth = {k: jnp.asarray(v) for k, v in truth.items()}
    mi = initialize_model(
        jax.random.PRNGKey(0), ilvk.pop_cosmo_model,
        model_args=model_args, model_kwargs=model_kwargs, dynamic_args=False,
        init_strategy=init_to_value(values=truth))
    return mi.param_info.z, mi.potential_fn


def test_full_model(prior):
    print("\n=== 4. full model: potential/gradients at truth, trace, dead event ===")
    data = make_synthetic_data(nobs=48, nsamp=128, nsel=20000)
    model_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
        data["Ndraw"], prior)

    _, bad = diagnose(ilvk.pop_cosmo_model, model_args, {}, LVK_TRUTH)
    check("trace at truth has no non-finite sites", not bad, str(bad))

    z0, pe_fn = _init_model(model_args, LVK_TRUTH)
    val, grad = jax.value_and_grad(pe_fn)(z0)
    bad_g = [k for k, g in grad.items() if not np.all(np.isfinite(np.asarray(g)))]
    check("potential finite at truth", bool(np.isfinite(val)), f"U = {float(val):.6e}")
    check("all parameter gradients finite", not bad_g, str(bad_g))

    # Dead event: force one event entirely below mmin (source frame).
    m1s_dead = np.array(data["m1s_det"])
    m1s_dead[0, :] = 3.0
    dead_args = (m1s_dead,) + model_args[1:]
    z0d, pe_d = _init_model(dead_args, LVK_TRUTH)
    val_d, grad_d = jax.value_and_grad(pe_d)(z0d)
    bad_gd = [k for k, g in grad_d.items() if not np.all(np.isfinite(np.asarray(g)))]
    check("dead-event potential finite", bool(np.isfinite(val_d)),
          f"U = {float(val_d):.6e}")
    check("dead-event gradients finite", not bad_gd, str(bad_gd))
    return model_args


def test_recentering(prior, model_args):
    print("\n=== 5. recentering baselines ===")
    baselines = ilvk.recentering_baselines(model_args, LVK_TRUTH)
    rk = dict(loglike_ref=baselines["loglike_ref"],
              log_mu_sel_ref=baselines["log_mu_sel_ref"],
              log_pdraw_sel_scale=baselines["log_pdraw_sel_scale"])

    z0, pe_raw = _init_model(model_args, LVK_TRUTH)
    z0c, pe_cen = _init_model(model_args, LVK_TRUTH, **rk)
    assert set(z0) == set(z0c)

    u_raw, g_raw = jax.value_and_grad(pe_raw)(z0)
    u_cen, g_cen = jax.value_and_grad(pe_cen)(z0)
    dpot = float(u_cen - u_raw)
    off = baselines["offset"]
    check("centered - raw potential == offset",
          abs(dpot - off) < 1e-4 * max(1.0, abs(off)),
          f"d = {dpot:.6e}, offset = {off:.6e}")
    gerr = max(float(np.max(np.abs(np.asarray(g_cen[k]) - np.asarray(g_raw[k]))
                            / max(1e-6, np.max(np.abs(np.asarray(g_raw[k]))))))
               for k in g_raw)
    check("gradients unchanged by recentering", gerr < 1e-4,
          f"max rel |dgrad| = {gerr:.2e}")
    # After centering only the prior terms remain, so the potential sits at
    # O(10) regardless of nobs/nsel -- the float32-friendly regime.
    check("centered potential parked near zero",
          abs(float(u_cen)) < 1e3,
          f"|U_cen| = {abs(float(u_cen)):.3e} (raw {abs(float(u_raw)):.3e})")


def test_mt_pairing(prior):
    print("\n=== 6. 'mt' (PISN-style total-mass) pairing ===")

    # (c) algebraic spot-check: model log_dN vs a direct recomputation from
    # the same object's own pieces (log_dndm, log_dndv, log_norm).
    ld = ilvk.LogDNDMDQDV_LVK(
        alpha_1=LVK_TRUTH["alpha_1"], alpha_2=LVK_TRUTH["alpha_2"],
        mbreak=LVK_TRUTH["mbreak"], mpp_1=LVK_TRUTH["mpp_1"],
        sigpp_1=LVK_TRUTH["sigpp_1"], mpp_2=LVK_TRUTH["mpp_2"],
        sigpp_2=LVK_TRUTH["sigpp_2"], f_peaks=LVK_TRUTH["f_peaks"],
        f_p1=LVK_TRUTH["f_p1"], beta=LVK_TRUTH["beta"],
        lam=LVK_TRUTH["lam"], mmin=FIXED["mmin"], mmax=FIXED["mmax"],
        delta_m=FIXED["delta_m"], zmax=6.5, pairing="mt")
    check("mt pairing builds no Zq table", ld.log_qnorm is None)
    check("mt + no kappa selects LogDNDVPowerLaw",
          type(ld.log_dndv) is ilvk.LogDNDVPowerLaw,
          type(ld.log_dndv).__name__)

    # ... and the Madau-Dickinson variant of the same object (kappa/zp given).
    ld_md = dataclasses.replace(
        ld, kappa=LVK_TRUTH["lam"] + LVK_TRUTH_MD["dkappa"],
        zp=LVK_TRUTH_MD["zp"])
    check("mt + kappa/zp selects the shared LogDNDV",
          type(ld_md.log_dndv) is ifast.LogDNDV,
          type(ld_md.log_dndv).__name__)

    m1 = jnp.array([8.0, 20.0, 35.0, 60.0, 120.0])
    q = jnp.array([0.9, 0.5, 0.7, 0.95, 0.6])
    z = jnp.array([0.1, 0.5, 1.0, 2.0, 0.05])
    got = np.asarray(ld(m1, q, z), np.float64)
    m2 = m1 * q
    manual = np.asarray(
        ld.log_dndm(m1) + ld.log_dndm(m2)
        + ld.beta * jnp.log((m1 + m2) / (30.0 * (1.0 + 1.0)))
        + jnp.log(m1) + ld.log_dndv(z) - ld.log_norm, np.float64)
    d = np.max(np.abs(got - manual))
    check("mt log_dN == f(m1) f(m2) (mt/60)^beta m1 dNdV / norm",
          bool(d < 1e-4), f"max |dlog| = {d:.2e}")
    # the same recomposition for the M-D variant: only log_dndv / log_norm
    # change, so this is the mt-pairing algebra on the other branch.
    got_md = np.asarray(ld_md(m1, q, z), np.float64)
    manual_md = np.asarray(
        ld_md.log_dndm(m1) + ld_md.log_dndm(m2)
        + ld_md.beta * jnp.log((m1 + m2) / (30.0 * (1.0 + 1.0)))
        + jnp.log(m1) + ld_md.log_dndv(z) - ld_md.log_norm, np.float64)
    d_md = np.max(np.abs(got_md - manual_md))
    check("mt log_dN recomposes with the M-D log_dndv too",
          bool(d_md < 1e-4), f"max |dlog| = {d_md:.2e}")
    # and the two branches must actually differ away from zref
    dz = float(np.max(np.abs(got - got_md)))
    check("power-law and M-D mt intensities differ off zref", dz > 0.1,
          f"max |dlog| = {dz:.3f}")
    # point normalization: m1 dN == 1 at (mref=30, qref=1, zref=1e-3)
    for label, obj in (("power law", ld), ("M-D", ld_md)):
        ref = float(np.asarray(jnp.log(30.0) + obj(30.0, 1.0, 0.001))[0])
        check(f"mt point normalization at (30, 1, 0.001), {label}",
              abs(ref) < 1e-5, f"log(m dN) = {ref:.2e}")

    # (a) selection-consistency identity with pairing="mt".
    data = make_synthetic_data(nobs=32, nsamp=64, nsel=100)
    nobs, nsamp = data["m1s_det"].shape
    nsel = nobs * nsamp
    ident_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det"].reshape(-1), data["qs"].reshape(-1),
        data["dls"].reshape(-1), np.exp(data["log_pdraw"].reshape(-1)),
        float(nsel), prior)
    tr = _trace_at(ident_args, LVK_TRUTH, store_per_event=True, pairing="mt")
    loglike = np.asarray(tr["loglik_array_dim"]["value"], np.float64)
    log_mu_sel = float(np.asarray(tr["log_mu_sel"]["value"]))
    lhs = log_mu_sel + np.log(nsel)
    rhs = np_logsumexp(loglike) + np.log(nsamp)
    check("log_mu_sel identity at truth (pairing='mt')",
          abs(lhs - rhs) < 5e-4, f"|d| = {abs(lhs - rhs):.2e} nats")

    # (b) finite potential and gradients at truth with pairing="mt".
    data = make_synthetic_data(nobs=48, nsamp=128, nsel=20000)
    model_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
        data["Ndraw"], prior)
    z0, pe_fn = _init_model(model_args, LVK_TRUTH, pairing="mt")
    val, grad = jax.value_and_grad(pe_fn)(z0)
    bad_g = [k for k, g in grad.items() if not np.all(np.isfinite(np.asarray(g)))]
    check("mt potential finite at truth", bool(np.isfinite(val)),
          f"U = {float(val):.6e}")
    check("mt parameter gradients all finite", not bad_g, str(bad_g))

    # (d) pairing="lvk" is bit-identical to the default (no kwarg) path.
    tr_def = _trace_at(ident_args, LVK_TRUTH, store_per_event=True)
    tr_lvk = _trace_at(ident_args, LVK_TRUTH, store_per_event=True,
                       pairing="lvk")
    same = all(np.array_equal(np.asarray(tr_def[k]["value"]),
                              np.asarray(tr_lvk[k]["value"]))
               for k in ("loglik_array_dim", "log_mu_sel", "min_neff",
                         "neff_sel", "mdNdmdVdt_fixed_qz"))
    check("pairing='lvk' bit-identical to default", same)

    # unknown pairing must be rejected at construction
    try:
        ilvk.LogDNDMDQDV_LVK(**{**{f.name: getattr(ld, f.name)
                                   for f in ld.__dataclass_fields__.values()
                                   if f.init}, "pairing": "bogus"})
        check("unknown pairing raises ValueError", False)
    except ValueError:
        check("unknown pairing raises ValueError", True)


def test_traced_bounds():
    """Gradients w.r.t. SAMPLED mmin/delta_m (the mminfree runs).

    Regression guard for the LogQNorm nan-gradient bug: with mmin traced,
    q-rows whose lower edge clips to 1-1e-6 have spacing below the float32
    ULP at 1.0, so a log(jnp.diff(q)) trapezoid produces log(0) and a nan
    d/d(mmin) at every draw.  The table now uses the analytic row spacing.
    """
    print("\n=== 8. traced mmin/delta_m (mminfree parametrization) ===")

    # Direct probe of the Zq table gradient.
    def zq_at_30(mmin, delta_m):
        qn = ilvk.LogQNorm(1.0, mmin, delta_m)
        return qn.log_Zq_from_log(jnp.log(30.0))

    g = jax.grad(zq_at_30, argnums=(0, 1))
    ok = True
    for mm, dm in ((3.0, 0.01), (4.5, 4.0), (7.0, 10.0), (9.99, 0.01)):
        gm, gd = (float(x) for x in g(mm, dm))
        if not (np.isfinite(gm) and np.isfinite(gd)):
            ok = False
    check("d(log Zq)/d(mmin, delta_m) finite across the prior box", ok)

    # Full model with mmin/delta_m as sample sites.
    prior_free = build_lvk_prior()
    import numpyro.distributions as dist
    prior_free["mmin"] = dist.Uniform(3.0, 10.0)
    prior_free["delta_m"] = dist.Uniform(0.01, 10.0)
    data = make_synthetic_data(nobs=32, nsamp=64, nsel=10000)
    model_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
        data["Ndraw"], prior_free)
    truth = {**LVK_TRUTH, "mmin": 4.5, "delta_m": 4.0}
    for pairing in ("lvk", "mt"):
        z0, pe_fn = _init_model(model_args, truth, pairing=pairing)
        assert "mmin" in z0 and "delta_m" in z0
        val, grad = jax.value_and_grad(pe_fn)(z0)
        bad = [k for k, gv in grad.items()
               if not np.all(np.isfinite(np.asarray(gv)))]
        check(f"pairing={pairing}: potential + all grads finite with "
              f"sampled mmin/delta_m", bool(np.isfinite(val)) and not bad,
              f"U = {float(val):.4e}" + (f", bad: {bad}" if bad else ""))


def test_redshift_dispatch(prior, prior_md):
    """Section 9: the prior alone selects the redshift form."""
    print("\n=== 9. redshift-evolution dispatch (prior-driven) ===")

    sample_pl, ld_pl = build_from_prior(prior, LVK_TRUTH)
    sample_md, ld_md = build_from_prior(prior_md, LVK_TRUTH_MD)

    check("prior without dkappa -> LogDNDVPowerLaw",
          type(ld_pl.log_dndv) is ilvk.LogDNDVPowerLaw,
          type(ld_pl.log_dndv).__name__)
    check("no kappa/zp leak into the power-law sample dict",
          "kappa" not in sample_pl and "zp" not in sample_pl,
          str(sorted(k for k in ("kappa", "dkappa", "zp") if k in sample_pl)))
    check("prior with dkappa -> intensity_models_fast.LogDNDV",
          type(ld_md.log_dndv) is ifast.LogDNDV,
          type(ld_md.log_dndv).__name__)

    kap = float(np.asarray(sample_md["kappa"]))
    check("kappa == lam + dkappa",
          abs(kap - (LVK_TRUTH["lam"] + LVK_TRUTH_MD["dkappa"])) < 1e-6,
          f"kappa = {kap}")
    check("zp passed through to the M-D object",
          abs(float(np.asarray(ld_md.log_dndv.zp)) - LVK_TRUTH_MD["zp"]) < 1e-6)

    # point normalization survives both branches in both pairings
    for label, pr, tv in (("power law", prior, LVK_TRUTH),
                          ("M-D", prior_md, LVK_TRUTH_MD)):
        for pairing in ("lvk", "mt"):
            _, ld = build_from_prior(pr, tv, pairing=pairing)
            v = float(np.asarray(jnp.log(30.0) + ld(30.0, 1.0, 0.001))[0])
            check(f"point normalization, {label}, pairing={pairing}",
                  abs(v) < 1e-5, f"log(m dN) = {v:.2e}")

    zs = np.array([1e-3, 0.05, 0.3, 1.0, 1.9, 3.0, 6.0, 6.49])
    lam, zp, zmax = LVK_TRUTH["lam"], LVK_TRUTH_MD["zp"], 6.5
    zref = float(ilvk.LogDNDMDQDV_LVK.zref)
    kappa = lam + LVK_TRUTH_MD["dkappa"]

    # (a) the M-D branch IS the PISN model's object, so it is bit-identical to
    #     constructing intensity_models_fast.LogDNDV directly -- i.e. exactly
    #     the pre-power-law-change numbers.
    direct = ifast.LogDNDV(lam, kappa, zp, zref, zmax=zmax)
    got = np.asarray(ld_md.log_dndv(jnp.asarray(zs)))
    want = np.asarray(direct(jnp.asarray(zs)))
    check("M-D log_dndv bit-identical to intensity_models_fast.LogDNDV",
          bool(np.array_equal(got, want)),
          f"max |d| = {np.max(np.abs(got - want)):.3e}")

    # (b) ... and it matches an independent float64 Madau-Dickinson formula.
    def ref_md(z):
        def f(zz):
            l1p = np.log1p(np.asarray(zz, np.float64))
            return lam * l1p - np.log1p(np.exp(kappa * (l1p - np.log1p(zp))))
        return f(z) - f(zref)

    d = np.max(np.abs(got.astype(np.float64) - ref_md(zs)))
    check("M-D log_dndv matches the float64 M-D reference",
          bool(d < 1e-4), f"max |d| = {d:.2e}")

    # (c) the power-law branch matches lam * [log1p(z) - log1p(zref)].
    got_pl = np.asarray(ld_pl.log_dndv(jnp.asarray(zs)), np.float64)
    ref_pl = lam * (np.log1p(zs) - np.log1p(zref))
    d = np.max(np.abs(got_pl - ref_pl))
    check("power-law log_dndv == lam (log1p(z) - log1p(zref))",
          bool(d < 1e-4), f"max |d| = {d:.2e}")

    # (d) kappa -> 0 degenerates M-D to the power law (the log 2 in the
    #     denominator is z-independent and cancels through log_norm).
    md0 = ifast.LogDNDV(lam, 0.0, zp, zref, zmax=zmax)
    d = float(np.max(np.abs(np.asarray(md0(jnp.asarray(zs)), np.float64)
                            - got_pl)))
    check("M-D at kappa = 0 reproduces the power law", d < 1e-4,
          f"max |d| = {d:.2e}")

    # (e) both truncate hard at zmax
    for label, obj in (("power law", ld_pl.log_dndv), ("M-D", ld_md.log_dndv)):
        v = np.asarray(obj(jnp.asarray([zmax, zmax + 1.0])))
        check(f"{label} log_dndv = -inf at/above zmax",
              bool(np.all(np.isneginf(v))), str(v))

    # (f) the half-specified turnover is a misconfiguration, not a branch
    base = {f.name: getattr(ld_pl, f.name)
            for f in ld_pl.__dataclass_fields__.values() if f.init}
    for label, kw in (("kappa without zp", dict(kappa=kappa)),
                      ("zp without kappa", dict(zp=zp))):
        try:
            ilvk.LogDNDMDQDV_LVK(**{**base, **kw})
            check(f"{label} raises ValueError", False)
        except ValueError:
            check(f"{label} raises ValueError", True)


def test_madau_dickinson(prior_md):
    """Section 10: the M-D branch through the full model."""
    print("\n=== 10. Madau-Dickinson branch, full model ===")

    # selection-consistency identity with sel := flattened events
    data = make_synthetic_data(nobs=32, nsamp=64, nsel=100)
    nobs, nsamp = data["m1s_det"].shape
    nsel = nobs * nsamp
    ident_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det"].reshape(-1), data["qs"].reshape(-1),
        data["dls"].reshape(-1), np.exp(data["log_pdraw"].reshape(-1)),
        float(nsel), prior_md)
    tr = None
    for pairing in ("lvk", "mt"):
        tr = _trace_at(ident_args, LVK_TRUTH_MD, store_per_event=True,
                       pairing=pairing)
        loglike = np.asarray(tr["loglik_array_dim"]["value"], np.float64)
        log_mu_sel = float(np.asarray(tr["log_mu_sel"]["value"]))
        lhs = log_mu_sel + np.log(nsel)
        rhs = np_logsumexp(loglike) + np.log(nsamp)
        check(f"log_mu_sel identity at M-D truth (pairing={pairing!r})",
              abs(lhs - rhs) < 5e-4, f"|d| = {abs(lhs - rhs):.2e} nats")
    check("'kappa' recorded as a deterministic site", "kappa" in tr,
          f"kappa = {float(np.asarray(tr['kappa']['value'])):.4f}"
          if "kappa" in tr else "")

    # finite potential / gradients, d/d(dkappa) and d/d(zp) included
    data = make_synthetic_data(nobs=48, nsamp=128, nsel=20000)
    model_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
        data["Ndraw"], prior_md)
    _, bad = diagnose(ilvk.pop_cosmo_model, model_args, {}, LVK_TRUTH_MD)
    check("M-D trace at truth has no non-finite sites", not bad, str(bad))
    z0, pe_fn = _init_model(model_args, LVK_TRUTH_MD)
    check("dkappa and zp are sample sites",
          "dkappa" in z0 and "zp" in z0, str(sorted(z0)))
    val, grad = jax.value_and_grad(pe_fn)(z0)
    bad_g = [k for k, g in grad.items() if not np.all(np.isfinite(np.asarray(g)))]
    check("M-D potential finite at truth", bool(np.isfinite(val)),
          f"U = {float(val):.6e}")
    check("M-D gradients all finite (dkappa/zp included)", not bad_g,
          str(bad_g))


def test_mcmc_smoke(prior, n):
    print(f"\n=== 7. NUTS smoke run ({n} warmup + {n} samples) ===")
    from numpyro.infer import MCMC, NUTS, init_to_value
    data = make_synthetic_data(nobs=64, nsamp=32, nsel=20000)
    model_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
        data["Ndraw"], prior)
    kernel = NUTS(ilvk.pop_cosmo_model, max_tree_depth=6,
                  init_strategy=init_to_value(values={k: jnp.asarray(v)
                                                      for k, v in LVK_TRUTH.items()}))
    mcmc = MCMC(kernel, num_warmup=n, num_samples=n, num_chains=1, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(7), *model_args)
    post = mcmc.get_samples()
    finite = all(np.all(np.isfinite(np.asarray(v))) for v in post.values())
    check("all posterior draws finite", finite)
    expect = ["R", "log_mu_sel", "frac_bpl"]
    if "dkappa" in prior:      # M-D prior -> derived `kappa` deterministic
        expect.append("kappa")
    for k in expect:
        check(f"site {k!r} recorded", k in post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc", type=int, default=0,
                    help="if >0, also run this many NUTS warmup+sample steps")
    args = ap.parse_args()

    print(f"jax {jax.__version__} on {jax.devices()}  x64={jax.config.jax_enable_x64}")
    prior = build_lvk_prior()
    prior_md = build_lvk_prior(LVK_PRIOR_TEXT_MD)

    test_mass_pdf_reference()
    test_q_norm()
    test_selection_identity(prior)
    model_args = test_full_model(prior)
    test_recentering(prior, model_args)
    test_mt_pairing(prior)
    test_traced_bounds()
    test_redshift_dispatch(prior, prior_md)
    test_madau_dickinson(prior_md)
    if args.mcmc:
        test_mcmc_smoke(prior, args.mcmc)
        test_mcmc_smoke(prior_md, args.mcmc)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

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

Run from scripts/testing_scripts:
    uv run python test_lvk_model.py             # tests 1-5
    uv run python test_lvk_model.py --mcmc 50   # + short NUTS smoke run
"""
import argparse
import os
import sys
import tempfile

sys.path.append("../src/")

import numpy as np
from scipy.special import logsumexp as np_logsumexp
from scipy.stats import norm as scipy_norm
import jax
import jax.numpy as jnp

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

# Roughly the GWTC-5 Default release posterior medians (see the plan notes).
LVK_TRUTH = dict(alpha_1=1.5, alpha_2=5.4, mbreak=37.0, mpp_1=9.9, sigpp_1=0.8,
                 mpp_2=32.0, sigpp_2=5.0, f_peaks=0.4, f_p1=0.55, beta=1.0,
                 lam=2.5)
FIXED = dict(mmin=4.5, mmax=300.0, delta_m=4.0)


def build_lvk_prior():
    fd, path = tempfile.mkstemp(suffix=".prior", prefix="lvk_test_")
    with os.fdopen(fd, "w") as f:
        f.write(LVK_PRIOR_TEXT)
    return get_priors_from_file(path)


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
    # point normalization: m1 dN == 1 at (mref=30, qref=1, zref=1e-3)
    ref = float(np.asarray(jnp.log(30.0) + ld(30.0, 1.0, 0.001))[0])
    check("mt point normalization at (30, 1, 0.001)", abs(ref) < 1e-5,
          f"log(m dN) = {ref:.2e}")

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
    for k in ("R", "log_mu_sel", "frac_bpl"):
        check(f"site {k!r} recorded", k in post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc", type=int, default=0,
                    help="if >0, also run this many NUTS warmup+sample steps")
    args = ap.parse_args()

    print(f"jax {jax.__version__} on {jax.devices()}  x64={jax.config.jax_enable_x64}")
    prior = build_lvk_prior()

    test_mass_pdf_reference()
    test_q_norm()
    test_selection_identity(prior)
    model_args = test_full_model(prior)
    test_recentering(prior, model_args)
    test_mt_pairing(prior)
    test_traced_bounds()
    if args.mcmc:
        test_mcmc_smoke(prior, args.mcmc)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

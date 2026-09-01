"""Regression tests for the two-scale h-pivot reparametrization
(`h_pivot = true` in the [run] ini section; implemented in
utils.sample_parameters_from_dict, threaded through both pop_cosmo_model
twins).  See notes/2026-09-01-h-divergences-float32.md.

Checks, in order:
  1. u-bounds are computed at runtime from the actual prior objects:
     reproduce the note's numbers for the real qpair_h prior, the *tighter*
     mpisndot-run mpisn_ref support (23.5, 49.1), and error clearly on a
     pathologically wide h prior or an unbounded mass prior
  2. configuration errors: fixed h, no pivotable site, log_mp_low prior
  3. pivot-off default equality: an explicit h_pivot=False trace is
     bit-identical to the no-kwarg default trace (fast model)
  4. DENSITY IDENTITY (the correctness proof): at matched physical points,
     pivot log-density - base log-density + (g1+g2) log h is CONSTANT (the
     +gamma log h term is the exact change-of-variables Jacobian), for
       (a) the fast model with the real qpair_h prior (mp_low + mpisn_ref
           pivoted, tail_anchor=ref_z, pairing=lvk), and
       (b) the slow twin with the bench prior (mp_low-only pivot; the slow
           twin has no mpisn_ref support)
     Also: mp_low / mpisn_ref appear as deterministics with the original
     names, u_mp_low / u_mpisn_ref as the new sample sites.
  5. finite potential + gradients (incl. d/du_* and d/dh) at the remapped
     reference point with the pivot on, for tail_anchor = ref_z and per_z

Run from scripts/:
    uv run python testing_scripts/test_h_pivot.py
"""
import sys

sys.path.append("../src/")

import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model, log_density

import intensity_models as slow
import intensity_models_fast as fast
from utils import (H_PIVOT_DEFAULT_GAMMAS, get_priors_from_file,
                   h_pivot_u_bounds, sample_parameters_from_dict)
from bench_model import make_synthetic_data, build_prior, TRUTH

FAILURES = []
G1 = H_PIVOT_DEFAULT_GAMMAS["mp_low"]
G2 = H_PIVOT_DEFAULT_GAMMAS["mpisn_ref"]
QPAIR_H_PRIOR = "priors/real_dat_noevo_qpair_h_r.prior"


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def expect_error(name, fn, snippet):
    try:
        fn()
    except ValueError as e:
        check(name, snippet in str(e), f"message: {e}")
    else:
        check(name, False, "no ValueError raised")


def test_u_bounds():
    print("\n=== 1. runtime u-bounds ===")
    qh = get_priors_from_file(QPAIR_H_PRIOR)
    h_d = qh["h"]  # TruncatedNormal(0.7, 0.2, low=0.4, high=1.2)

    lo, hi = h_pivot_u_bounds(qh["mp_low"], h_d, G1, "mp_low")
    check("u_mp_low bounds match the note (5.1293, 13.1941)",
          np.allclose([lo, hi], [5.1293, 13.1941], atol=2e-3),
          f"got ({lo:.4f}, {hi:.4f})")
    lo, hi = h_pivot_u_bounds(qh["mpisn_ref"], h_d, G2, "mpisn_ref")
    check("u_mpisn_ref bounds match the note (20.6672, 41.6344)",
          np.allclose([lo, hi], [20.6672, 41.6344], atol=2e-3),
          f"got ({lo:.4f}, {hi:.4f})")

    # The mpisndot-run prior has the TIGHTER support (23.5, 49.1): the bounds
    # must come from the actual dist object, not the note's hardcoded numbers.
    tight = dist.TruncatedNormal(35.0, 5.0, low=23.5, high=49.1)
    lo, hi = h_pivot_u_bounds(tight, h_d, G2, "mpisn_ref")
    exp_lo, exp_hi = 23.5 * 1.2 ** G2, 49.1 * 0.4 ** G2
    check("tighter mpisn_ref support (23.5, 49.1) -> tighter u bounds",
          np.allclose([lo, hi], [exp_lo, exp_hi], rtol=1e-12),
          f"got ({lo:.4f}, {hi:.4f}), expected ({exp_lo:.4f}, {exp_hi:.4f})")

    expect_error("pathologically wide h prior errors",
                 lambda: h_pivot_u_bounds(qh["mp_low"],
                                          dist.Uniform(0.01, 100.0),
                                          G1, "mp_low"),
                 "empty u-range")
    expect_error("unbounded mass prior errors",
                 lambda: h_pivot_u_bounds(
                     dist.TruncatedNormal(0.1, 0.1, low=0.05), h_d, G1, "m"),
                 "finite positive truncation")
    expect_error("gamma <= 0 errors",
                 lambda: h_pivot_u_bounds(qh["mp_low"], h_d, 0.0, "mp_low"),
                 "must be > 0")


def test_config_errors():
    print("\n=== 2. configuration errors ===")
    mp = dist.TruncatedNormal(9.0, 2.0, low=5.0, high=15.0)
    h_d = dist.TruncatedNormal(0.7, 0.2, low=0.4, high=1.2)
    expect_error("fixed h errors",
                 lambda: sample_parameters_from_dict(
                     {"h": 0.674, "mp_low": mp}, h_pivot=True),
                 "requires the prior to SAMPLE h")
    expect_error("nothing to pivot errors",
                 lambda: sample_parameters_from_dict(
                     {"h": h_d, "mp_low": 9.0, "mpisn_ref": 33.0},
                     h_pivot=True),
                 "nothing to pivot")
    expect_error("log_mp_low prior errors",
                 lambda: sample_parameters_from_dict(
                     {"h": h_d, "log_mp_low": dist.Normal(np.log(9.0), 0.2)},
                     h_pivot=True),
                 "does not support a log_mp_low")


def _trace(model, model_args, kwargs, subs, seed=0):
    with handlers.seed(rng_seed=seed), handlers.substitute(data=subs):
        return handlers.trace(model).get_trace(*model_args, **kwargs)


def test_default_bit_identity(data):
    print("\n=== 3. h_pivot=False is bit-identical to the default ===")
    prior = build_prior(True, "/tmp/h_pivot_prior_default.prior")
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)
    tr_a = _trace(fast.pop_cosmo_model, model_args, {}, {}, seed=11)
    tr_b = _trace(fast.pop_cosmo_model, model_args, dict(h_pivot=False), {},
                  seed=11)
    same_sites = list(tr_a) == list(tr_b)
    check("identical site inventory", same_sites,
          "" if same_sites else f"{sorted(set(tr_a) ^ set(tr_b))}")
    bad = []
    for name in tr_a:
        va, vb = tr_a[name].get("value"), tr_b[name].get("value")
        if va is None or (hasattr(va, "size") and va.size == 0):
            continue
        if not np.array_equal(np.asarray(va), np.asarray(vb)):
            bad.append(name)
    check("all site values bitwise equal", not bad, f"differ: {bad}")


def _qpair_points(n=20, seed=42):
    """Matched physical points: the qpair_h [ref_params] medians, jittered,
    with h swept across most of its prior support."""
    ref = dict(a=1.264, b=1.170, c=4.844, mpisn_ref=33.04, dmbhmax=3.267,
               sigma=0.1063, beta=2.522, lam=2.258, dkappa=3.243, zp=1.877,
               mp_low=9.895, msigma_low=0.6435, log_fpeak=1.136,
               log_r=-0.7533)
    rng = np.random.default_rng(seed)
    hs = np.linspace(0.45, 1.15, n)
    pts = []
    for i in range(n):
        p = {k: v * (1.0 + 0.03 * rng.uniform(-1, 1)) for k, v in ref.items()}
        p["h"] = hs[i]
        p["R_unit"] = float(rng.normal(0, 0.3))
        pts.append(p)
    return pts


def _identity_report(label, diffs, hs, jac_gammas):
    diffs = np.asarray(diffs, dtype=np.float64)
    corr = diffs + jac_gammas * np.log(np.asarray(hs))
    spread = corr.max() - corr.min()
    raw_spread = diffs.max() - diffs.min()
    check(f"{label}: pivot - base + {jac_gammas:.2f} log h constant "
          f"(spread {spread:.2e} < 1e-3)", spread < 1e-3,
          f"const = {corr.mean():.6f}")
    # sanity that the check has teeth: without the Jacobian the difference is
    # NOT constant across the h sweep
    check(f"{label}: Jacobian term is load-bearing",
          raw_spread > 10 * max(spread, 1e-12),
          f"raw spread {raw_spread:.3e}")


def test_density_identity_fast(data):
    print("\n=== 4a. density identity: fast model, real qpair_h prior ===")
    prior = get_priors_from_file(QPAIR_H_PRIOR)
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)
    kw_base = dict(use_low_bump=True, tail_anchor="ref_z", pairing="lvk")
    kw_piv = dict(kw_base, h_pivot=True)

    pts = _qpair_points()
    diffs, hs = [], []
    tr_piv = None
    for p in pts:
        base = {k: jnp.asarray(v) for k, v in p.items()}
        piv = dict(base)
        piv["u_mp_low"] = jnp.asarray(p["mp_low"] * p["h"] ** G1)
        piv["u_mpisn_ref"] = jnp.asarray(p["mpisn_ref"] * p["h"] ** G2)
        del piv["mp_low"], piv["mpisn_ref"]
        ld_base, _ = log_density(fast.pop_cosmo_model, model_args, kw_base,
                                 base)
        ld_piv, tr_piv = log_density(fast.pop_cosmo_model, model_args, kw_piv,
                                     piv)
        diffs.append(float(ld_piv) - float(ld_base))
        hs.append(p["h"])
    _identity_report("fast/qpair_h (mp_low + mpisn_ref)", diffs, hs, G1 + G2)

    # site inventory of the pivot trace
    p = pts[-1]
    ok_sites = (tr_piv["u_mp_low"]["type"] == "sample"
                and tr_piv["u_mpisn_ref"]["type"] == "sample"
                and tr_piv["mp_low"]["type"] == "deterministic"
                and tr_piv["mpisn_ref"]["type"] == "deterministic")
    check("u_* are sample sites; mp_low/mpisn_ref are deterministics",
          ok_sites)
    check("derived mp_low matches the physical value",
          np.allclose(float(tr_piv["mp_low"]["value"]), p["mp_low"],
                      rtol=1e-5))
    check("derived mpisn_ref matches the physical value",
          np.allclose(float(tr_piv["mpisn_ref"]["value"]), p["mpisn_ref"],
                      rtol=1e-5))


def test_density_identity_slow(data):
    print("\n=== 4b. density identity: slow twin, bench prior (mp_low only) ===")
    prior = build_prior(True, "/tmp/h_pivot_prior_slow.prior")
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)

    rng = np.random.default_rng(7)
    hs = np.linspace(0.45, 1.15, 10)
    diffs = []
    for h in hs:
        p = {k: jnp.asarray(v * (1.0 + 0.02 * rng.uniform(-1, 1)))
             for k, v in TRUTH.items() if k != "mpisndot"}
        p["h"] = jnp.asarray(h)
        p["R_unit"] = jnp.asarray(0.0)
        piv = dict(p)
        piv["u_mp_low"] = jnp.asarray(float(p["mp_low"]) * h ** G1)
        del piv["mp_low"]
        ld_base, _ = log_density(slow.pop_cosmo_model, model_args, {}, p)
        ld_piv, _ = log_density(slow.pop_cosmo_model, model_args,
                                dict(h_pivot=True), piv)
        diffs.append(float(ld_piv) - float(ld_base))
    _identity_report("slow/bench (mp_low)", diffs, hs, G1)


def test_gradients(data):
    print("\n=== 5. finite gradients at the remapped reference point ===")
    prior = get_priors_from_file(QPAIR_H_PRIOR)
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)
    ref = _qpair_points(n=1)[0]
    ref_piv = {k: jnp.asarray(v) for k, v in ref.items()
               if k not in ("mp_low", "mpisn_ref", "R_unit")}
    ref_piv["u_mp_low"] = jnp.asarray(ref["mp_low"] * ref["h"] ** G1)
    ref_piv["u_mpisn_ref"] = jnp.asarray(ref["mpisn_ref"] * ref["h"] ** G2)

    for ta in ("ref_z", "per_z"):
        mi = initialize_model(
            jax.random.PRNGKey(0), fast.pop_cosmo_model,
            model_args=model_args,
            model_kwargs=dict(use_low_bump=True, tail_anchor=ta,
                              pairing="lvk", h_pivot=True),
            init_strategy=init_to_value(values=ref_piv))
        z = mi.param_info.z
        pot = mi.potential_fn(z)
        grads = jax.grad(mi.potential_fn)(z)
        bad = [k for k, v in grads.items()
               if not np.all(np.isfinite(np.asarray(v)))]
        have = all(k in grads for k in ("u_mp_low", "u_mpisn_ref", "h"))
        check(f"tail_anchor={ta}: finite potential", np.isfinite(float(pot)),
              f"pot = {float(pot):.3f}")
        check(f"tail_anchor={ta}: finite gradients incl. u_*, h",
              not bad and have, f"non-finite: {bad}" if bad else
              f"|dU/du_mp_low| = {abs(float(grads['u_mp_low'])):.3g}, "
              f"|dU/dh| = {abs(float(grads['h'])):.3g}")


if __name__ == "__main__":
    test_u_bounds()
    test_config_errors()
    data = make_synthetic_data(60, 200, 20000, seed=3)
    test_default_bit_identity(data)
    test_density_identity_fast(data)
    test_density_identity_slow(data)
    test_gradients(data)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL PASS")

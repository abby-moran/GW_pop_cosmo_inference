"""Validation of the PISN model's ``pairing = "lvk"`` mode (LVK-style
normalized q^beta pairing), which completes the 2x2 comparison design
{PISN mass, LVK mass} x {mt pairing, q^beta pairing}.

Checks, in order:
  1. algebraic recomposition: fast LogDNDMDQDV(pairing="lvk") equals
     log_dndm(m1,z) + beta log q + log S(q m1) - log Zq(m1) + log dNdV(z)
     - log_norm, assembled from the object's OWN pieces (exact, 0 tolerance)
  2. default-pairing regression: an explicit pairing="mt" object is
     bit-identical to the no-kwarg default construction
  3. slow vs fast for pairing="lvk": direct-path log_dN values on an
     (m, q, z) sweep, and initialize_model potential + gradients on synthetic
     data (test_fast_equivalence.test_full_model tolerances)
  4. tabulated vs direct (fast, pairing="lvk"): potential and gradients agree
     to tabulation accuracy, and the selection-consistency identity of
     notes/2026-08-08-tabulated-selection-consistency.md holds in BOTH modes
     (sel := flattened events => log_mu_sel + log Ndraw ==
     logsumexp(loglike) + log nsamp)
  5. finite potential/gradients at truth for pairing="lvk" under all three
     tail_anchor modes (simplex with log_fpl; ref_z/per_z with log_r)

Run from scripts/:
    uv run python testing_scripts/test_pisn_qpair.py
"""
import sys

sys.path.append("../src/")

import numpy as np
from scipy.special import logsumexp as np_logsumexp
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro.handlers as handlers

import intensity_models as slow
import intensity_models_fast as fast

from bench_model import make_synthetic_data, build_prior, TRUTH, FIDUCIAL

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(label, a, b, rtol=1e-4, atol=1e-5):
    """test_fast_equivalence-style comparison (finite entries + inf pattern)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{label}: shape {a.shape} vs {b.shape}"
    both = np.isfinite(a) & np.isfinite(b)
    n_pat = int(((~np.isfinite(a)) != (~np.isfinite(b))).sum())
    d = np.abs(a[both] - b[both])
    scale = np.maximum(np.abs(a[both]), np.abs(b[both]))
    ok = bool(np.all(d <= atol + rtol * scale)) and n_pat == 0
    check(label, ok, f"max|d|={d.max() if d.size else 0.0:.3e} "
                     f"pattern mismatches={n_pat}")


def full_sample(pt, mpisndot=0.0, zmax=6.5, mbh_min=3.0, delta_m=1.6):
    """TRUTH-style point -> full parameter dict (as test_fast_equivalence)."""
    s = dict(pt)
    s.update(Om=FIDUCIAL["Om"], w=FIDUCIAL["w"], mpisndot=mpisndot,
             zmax=zmax, mbh_min=mbh_min, delta_m=delta_m)
    s["kappa"] = s["lam"] + s["dkappa"]
    s["mbhmax"] = s["mpisn"] + s["dmbhmax"]
    s["fpl"] = float(np.exp(s["log_fpl"]))
    s["flow"] = float(np.exp(s["log_flow"]))
    return s


def _grids():
    m = np.exp(np.linspace(np.log(2.0), np.log(200.0), 137))
    q = np.linspace(0.02, 1.0, 41)
    z = np.expm1(np.linspace(np.log1p(1e-3), np.log1p(6.0), 23))
    return np.meshgrid(m, q, z, indexing="ij")


def test_recomposition():
    print("\n=== 1. algebraic recomposition (fast, pairing='lvk') ===")
    M, Q, Z = _grids()
    for mpisndot, anchor in ((0.0, "simplex"), (1.5, "per_z")):
        s = full_sample(dict(TRUTH), mpisndot=mpisndot)
        ld = fast.build_population_model(s, pairing="lvk", tail_anchor=anchor)
        got = np.asarray(ld(M, Q, Z))

        # The same primitive ops, in the same order, on the same broadcast
        # arrays the model uses -- so the comparison can be EXACT.
        m1, q, z = jnp.broadcast_arrays(
            jnp.atleast_1d(jnp.asarray(M)), jnp.atleast_1d(jnp.asarray(Q)),
            jnp.atleast_1d(jnp.asarray(Z)))
        log_m1, log_q, log1p_z = jnp.log(m1), jnp.log(q), jnp.log1p(z)
        lm = ld.log_dndm
        mbhmax_at = jnp.broadcast_to(jnp.asarray(lm.mbhmax_at_z(z)), jnp.shape(m1))
        log_f_pl = (lm.log_f_eff(mbhmax_at, z, log1p_z)
                    if lm.tail_anchor == "per_z" else None)
        m2 = m1 * jnp.exp(log_q)
        manual = np.asarray(
            lm.call_from_logs(m1, log_m1, z, log1p_z, mbhmax_at, log_f_pl)
            + ld.beta * log_q
            + fast.mmin_log_smooth_turnon(m2, ld.delta_m, ld.mbh_min)
            - ld.log_qnorm.log_Zq_from_log(log_m1)
            + ld.log_dndv.from_log1p(log1p_z) - ld.log_norm)
        same = np.array_equal(got, manual, equal_nan=True)
        check(f"log_dN == f(m1) q^beta S(q m1) / Zq(m1) * dNdV / norm exactly "
              f"(mpisndot={mpisndot}, {anchor})", same,
              "bit-identical" if same else
              f"max|d|={np.nanmax(np.abs(got - manual)):.3e}")

        # point normalization: m1 dN == 1 at (mref=30, qref=1, zref=1e-3)
        ref = float(np.asarray(jnp.log(30.0) + ld(30.0, 1.0, 0.001))[0])
        check(f"point normalization at (30, 1, 0.001) (mpisndot={mpisndot}, "
              f"{anchor})", abs(ref) < 1e-5, f"log(m dN) = {ref:.2e}")

    # mt pairing must not build a Zq table; lvk must.
    s = full_sample(dict(TRUTH))
    check("mt pairing builds no Zq table",
          fast.build_population_model(s).log_qnorm is None)
    check("lvk pairing builds a Zq table",
          fast.build_population_model(s, pairing="lvk").log_qnorm is not None)


def test_default_regression():
    print("\n=== 2. default-pairing regression (pairing='mt' == no kwarg) ===")
    M, Q, Z = _grids()
    for mpisndot in (0.0, 1.5):
        s = full_sample(dict(TRUTH), mpisndot=mpisndot)
        ld_def = fast.build_population_model(s)
        ld_mt = fast.build_population_model(s, pairing="mt")
        a = np.asarray(ld_def(M, Q, Z))
        b = np.asarray(ld_mt(M, Q, Z))
        check(f"mt explicit bit-identical to default (mpisndot={mpisndot})",
              np.array_equal(a, b, equal_nan=True))
        check(f"log_norm bit-identical (mpisndot={mpisndot})",
              np.array_equal(np.asarray(ld_def.log_norm),
                             np.asarray(ld_mt.log_norm)))


def _init_model(mod, model_args, truth, **model_kwargs):
    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value
    truth = {k: jnp.asarray(v) for k, v in truth.items()}
    mi = initialize_model(
        jax.random.PRNGKey(0), mod.pop_cosmo_model,
        model_args=model_args, model_kwargs=model_kwargs, dynamic_args=False,
        init_strategy=init_to_value(values=truth))
    return mi.param_info.z, mi.potential_fn


def _model_args(data, prior):
    return (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
            data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
            data["pdraw_sel"], data["Ndraw"], prior)


def test_slow_vs_fast(prior, truth):
    print("\n=== 3. slow vs fast, pairing='lvk' ===")
    # direct-path value sweep
    M, Q, Z = _grids()
    for mpisndot in (0.0, 1.5):
        s = full_sample(dict(TRUTH), mpisndot=mpisndot)
        ds = slow.build_population_model(s, pairing="lvk")
        df = fast.build_population_model(s, pairing="lvk")
        report(f"log_dN(m1,q,z) sweep (mpisndot={mpisndot})",
               ds(M, Q, Z), df(M, Q, Z), rtol=3e-4, atol=3e-4)
        report(f"log_norm (mpisndot={mpisndot})", ds.log_norm, df.log_norm,
               rtol=3e-4, atol=3e-4)

    # full-model potential + gradient (direct path; the slow model has no
    # tabulation and its legacy guards match neff_penalty="min_neff")
    data = make_synthetic_data(400, 300, 40000, seed=3)
    model_args = _model_args(data, prior)
    exact = dict(tabulate_mass_function=False, smooth_tail_edge=False,
                 neff_penalty="min_neff")
    outs = {}
    for label, mod, kwargs in (("slow", slow, {}), ("fast", fast, dict(exact))):
        z0, pe_fn = _init_model(mod, model_args, truth,
                                use_low_bump=True, pairing="lvk", **kwargs)
        v, g = jax.jit(jax.value_and_grad(pe_fn))(z0)
        outs[label] = (float(v), {k: float(x) for k, x in g.items()})
        bad = [k for k, x in g.items() if not np.isfinite(x)]
        print(f"  {label:5s} potential={float(v):+.8e}  "
              f"non-finite grads: {bad or 'none'}")
        if bad:
            FAILURES.append(f"{label} non-finite grads (pairing='lvk')")
    vs, gs = outs["slow"]
    vf, gf = outs["fast"]
    report("potential energy", np.array([vs]), np.array([vf]),
           rtol=2e-4, atol=1e-2)
    keys = sorted(set(gs) & set(gf))
    report("gradient wrt all sampled params",
           np.array([gs[k] for k in keys]), np.array([gf[k] for k in keys]),
           rtol=5e-3, atol=5e-3)
    for k in keys:
        print(f"      d/d{k:12s} slow={gs[k]:+.6e}  fast={gf[k]:+.6e}")


def test_tabulated(prior, truth):
    print("\n=== 4. tabulated vs direct + selection identity (pairing='lvk') ===")
    data = make_synthetic_data(400, 300, 40000, seed=11)
    model_args = _model_args(data, prior)
    outs = {}
    for label, kwargs in (
        ("direct", dict(tabulate_mass_function=False)),
        ("tab", dict(tabulate_mass_function=True)),
    ):
        z0, pe_fn = _init_model(fast, model_args, truth, use_low_bump=True,
                                pairing="lvk", smooth_tail_edge=True,
                                neff_penalty="min_neff", **kwargs)
        v, g = jax.jit(jax.value_and_grad(pe_fn))(z0)
        outs[label] = (float(v), {k: float(x) for k, x in g.items()})
        bad = [k for k, x in g.items() if not np.isfinite(x)]
        print(f"  {label:6s} potential={float(v):+.8e}  "
              f"non-finite grads: {bad or 'none'}")
        if bad:
            FAILURES.append(f"{label} non-finite grads (pairing='lvk')")
    vd, gd = outs["direct"]
    vt, gt = outs["tab"]
    report("tab vs direct: potential", np.array([vd]), np.array([vt]),
           rtol=1e-4, atol=0.5)
    # Same edge-parameter carve-out as test_fast_equivalence.test_tabulated_path
    # (they move sample masses across the table's smeared tail edge).
    edge_params = {"h", "Omh2", "mpisn", "dmbhmax"}
    keys = sorted(set(gd) & set(gt) - edge_params)
    report("tab vs direct: gradients (non-edge params)",
           np.array([gd[k] for k in keys]), np.array([gt[k] for k in keys]),
           rtol=2e-2, atol=5e-2)

    # Selection-consistency identity, tabulated AND direct: sel := flattened
    # event samples => log_mu_sel + log Ndraw == logsumexp(loglike) + log nsamp.
    data_i = make_synthetic_data(32, 64, 100, seed=0)
    nobs, nsamp = data_i["m1s_det"].shape
    nsel = nobs * nsamp
    ident_args = (
        data_i["m1s_det"], data_i["qs"], data_i["dls"], data_i["log_pdraw"],
        data_i["m1s_det"].reshape(-1), data_i["qs"].reshape(-1),
        data_i["dls"].reshape(-1), np.exp(data_i["log_pdraw"].reshape(-1)),
        float(nsel), prior)
    vals = {k: jnp.asarray(v) for k, v in truth.items()}
    for label, kwargs in (("tabulated", dict(tabulate_mass_function=True)),
                          ("direct", dict(tabulate_mass_function=False))):
        with handlers.seed(rng_seed=0), handlers.substitute(data=vals):
            tr = handlers.trace(fast.pop_cosmo_model).get_trace(
                *ident_args, use_low_bump=True, pairing="lvk",
                store_per_event=True, neff_penalty="none", **kwargs)
        loglike = np.asarray(tr["loglik_array_dim"]["value"], np.float64)
        log_mu_sel = float(np.asarray(tr["log_mu_sel"]["value"]))
        lhs = log_mu_sel + np.log(nsel)
        rhs = np_logsumexp(loglike) + np.log(nsamp)
        check(f"log_mu_sel identity, {label} mode", abs(lhs - rhs) < 1e-3,
              f"|d| = {abs(lhs - rhs):.2e} nats")


def test_tail_anchor_modes(prior, truth):
    print("\n=== 5. pairing='lvk' x tail_anchor modes: finite U and grads ===")
    data = make_synthetic_data(200, 128, 20000, seed=23)

    # simplex: the bench prior (samples log_fpl) as-is.
    z0, pe_fn = _init_model(fast, _model_args(data, prior), truth,
                            use_low_bump=True, pairing="lvk",
                            tail_anchor="simplex")
    v, g = jax.value_and_grad(pe_fn)(z0)
    bad = [k for k, x in g.items() if not np.isfinite(np.asarray(x))]
    check("simplex: potential + grads finite",
          bool(np.isfinite(v)) and not bad,
          f"U = {float(v):.4e}" + (f", bad: {bad}" if bad else ""))

    # ref_z / per_z: swap log_fpl for log_r (the r-modes' prior coordinate).
    prior_r = build_prior(True, "/tmp/qpair_prior_r.prior")
    del prior_r["log_fpl"]
    prior_r["log_r"] = dist.Uniform(np.log(1e-2), 0.0)
    truth_r = {k: v for k, v in truth.items() if k != "log_fpl"}
    truth_r["log_r"] = jnp.asarray(np.log(0.6))
    for mode in ("ref_z", "per_z"):
        z0, pe_fn = _init_model(fast, _model_args(data, prior_r), truth_r,
                                use_low_bump=True, pairing="lvk",
                                tail_anchor=mode)
        v, g = jax.value_and_grad(pe_fn)(z0)
        bad = [k for k, x in g.items() if not np.isfinite(np.asarray(x))]
        check(f"{mode}: potential + grads finite",
              bool(np.isfinite(v)) and not bad,
              f"U = {float(v):.4e}" + (f", bad: {bad}" if bad else ""))


def main():
    print(f"jax {jax.__version__} on {jax.devices()}  "
          f"x64={jax.config.jax_enable_x64}")
    prior = build_prior(True, "/tmp/qpair_prior.prior")   # mpisndot fixed to 0
    truth = {k: jnp.asarray(v) for k, v in TRUTH.items()
             if k in prior and not isinstance(prior[k], float)}

    test_recomposition()
    test_default_regression()
    test_slow_vs_fast(prior, truth)
    test_tabulated(prior, truth)
    test_tail_anchor_modes(prior, truth)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

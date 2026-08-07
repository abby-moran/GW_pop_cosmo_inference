"""
Numerical equivalence check: intensity_models_fast vs intensity_models.

Compares, at several parameter points, on data small enough to run quickly:
  1. FlatwCDMCosmology: the fused z_and_log_jacobian against the original
     three separate jnp.interp lookups.
  2. LogDNDMPISN.log_dN_grid / log_Z_grid (the max-subtracted trapezoid vs
     logaddexp+logsumexp).
  3. LogDNDM.__call__ and LogDNDMDQDV.__call__ over a wide (m, q, z) sweep.
  4. The full model potential energy and its gradient, via numpyro.

Run:  uv run python test_fast_equivalence.py
"""
import sys

sys.path.append("../src/")

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.handlers as handlers

import intensity_models as slow
import intensity_models_fast as fast

from bench_model import make_synthetic_data, build_prior, TRUTH, FIDUCIAL

FAIL = []


def report(label, a, b, rtol=1e-4, atol=1e-5, finite_only=True):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{label}: shape {a.shape} vs {b.shape}"
    both_finite = np.isfinite(a) & np.isfinite(b)
    same_nonfinite = (~np.isfinite(a)) == (~np.isfinite(b))
    n_mismatch_pattern = int((~same_nonfinite).sum())

    if finite_only:
        m = both_finite
    else:
        m = np.ones_like(a, dtype=bool)
    if m.sum() == 0:
        print(f"  {label:52s} (no finite points to compare)")
        return
    d = np.abs(a[m] - b[m])
    scale = np.maximum(np.abs(a[m]), np.abs(b[m]))
    rel = d / np.maximum(scale, 1e-30)
    ok = np.all(d <= atol + rtol * scale) and n_mismatch_pattern == 0
    status = "OK " if ok else "FAIL"
    if not ok:
        FAIL.append(label)
    print(f"  [{status}] {label:50s} max|d|={d.max():.3e} max rel={rel.max():.3e} "
          f"n={m.sum()} inf/nan-pattern mismatches={n_mismatch_pattern}")


PARAM_POINTS = [
    dict(TRUTH),
    dict(TRUTH, sigma=0.35, mpisn=45.0, dmbhmax=6.0, c=5.0, a=2.0, b=3.0),
    dict(TRUTH, sigma=0.05, mpisn=22.0, dmbhmax=0.7, h=0.9, lam=0.5, dkappa=1.2, zp=3.0),
]


def full_sample(pt, mpisndot=0.0, zmax=6.5, mbh_min=3.0, delta_m=1.6):
    s = dict(pt)
    s.update(Om=FIDUCIAL["Om"], w=FIDUCIAL["w"], mpisndot=mpisndot,
             zmax=zmax, mbh_min=mbh_min, delta_m=delta_m)
    s["kappa"] = s["lam"] + s["dkappa"]
    s["mbhmax"] = s["mpisn"] + s["dmbhmax"]
    s["fpl"] = float(np.exp(s["log_fpl"]))
    s["flow"] = float(np.exp(s["log_flow"]))
    return s


def _cosmo_reference(h, Om, w, zmax, z_probe):
    """float64, densely-sampled reference for dL(z), and for
    J(z) = log(dVC/dz) - log(ddL/dz) - 2 log1p(z)."""
    n = 1 << 20
    zi = np.expm1(np.linspace(0.0, np.log1p(zmax), n))
    E = np.sqrt(Om * (1 + zi) ** 3 + (1 - Om) * (1 + zi) ** (3 * (1 + w)))
    dH = 2.99792 / h
    dc = dH * np.concatenate(([0.0], np.cumsum(0.5 * np.diff(zi) * (1 / E[:-1] + 1 / E[1:]))))
    dl = dc * (1 + zi)

    dc_p = np.interp(z_probe, zi, dc)
    E_p = np.sqrt(Om * (1 + z_probe) ** 3 + (1 - Om) * (1 + z_probe) ** (3 * (1 + w)))
    ddl = dc_p + dH * (1 + z_probe) / E_p
    dvc = 4 * np.pi * dc_p ** 2 * dH / E_p
    J = np.log(dvc) - np.log(ddl) - 2 * np.log1p(z_probe)
    return np.interp(z_probe, zi, dl), J


def test_cosmology():
    print("\n=== 1. cosmology: fused z_and_log_jacobian vs 3 separate interps ===")
    print("    (both are scored against a float64 dense-quadrature reference,")
    print("     since neither implementation is exact)")
    for i, pt in enumerate(PARAM_POINTS):
        s = full_sample(pt)
        cs = slow.FlatwCDMCosmology(s["h"], s["Om"], s["w"], zmax=s["zmax"])
        cf = fast.FlatwCDMCosmology(s["h"], s["Om"], s["w"], zmax=s["zmax"])

        z_probe = np.expm1(np.linspace(np.log1p(1e-3), np.log1p(0.98 * s["zmax"]), 4000))
        dl_ref, J_ref = _cosmo_reference(s["h"], s["Om"], s["w"], s["zmax"], z_probe)
        dl = dl_ref

        z_slow = cs.z_of_dL(dl)
        J_slow = np.asarray(jnp.log(cs.dVCdz(z_slow)) - jnp.log(cs.ddL_dz(z_slow))
                            - 2 * jnp.log1p(z_slow), np.float64)
        log1p_z_fast, J_fast = cf.z_and_log_jacobian(jnp.log(dl))
        J_fast = np.asarray(J_fast, np.float64)

        report(f"pt{i}: log1p(z) from dL   slow vs fast", jnp.log1p(z_slow), log1p_z_fast,
               rtol=2e-4, atol=1e-6)

        # Accuracy of each against the reference, overall and restricted to the
        # z range that actually carries events.
        band = z_probe > 0.02
        for lab, arr in (("slow", J_slow), ("fast", J_fast)):
            e_all = np.abs(arr - J_ref).max()
            e_band = np.abs(arr - J_ref)[band].max()
            print(f"        J error vs float64 ref, {lab}: all z {e_all:.3e} | "
                  f"z>0.02 {e_band:.3e}")
        e_slow = np.abs(J_slow - J_ref)[band].max()
        e_fast = np.abs(J_fast - J_ref)[band].max()
        if e_fast > max(3e-4, 2 * e_slow):
            FAIL.append(f"pt{i}: fast J less accurate than slow")
            print(f"  [FAIL] pt{i}: fast J accuracy regressed")
        else:
            print(f"  [OK ] pt{i}: fast J accuracy >= slow "
                  f"({e_fast:.2e} vs {e_slow:.2e} for z>0.02)")


def _pisn_reference(a, b, mpisns, mbhmaxs, sigma, n_m=512):
    """float64 evaluation of the same mco integral the model does."""
    log_mbh = np.linspace(np.log(1.5), np.log(100.0), n_m + 2)
    log_mco = np.linspace(np.log(1.0), np.log(100.0), n_m)
    mco = np.exp(log_mco)
    mp = mpisns[:, None]
    mb = mbhmaxs[:, None]
    aa = 1 / (4 * (mp - mb))
    mcomax = 2 * mb - mp
    mu = np.where(mco[None, :] < mp, mco[None, :], mb + aa * (mco[None, :] - mcomax) ** 2)
    mu = np.where(mu > 0, mu, 0.1)
    log_mu = np.log(mu)
    mco_eff = np.maximum(mco, 6.0)
    x = mco_eff / 20.0
    lwco = (np.where(mco_eff < 20.0, -a * np.log(x), -b * np.log(x))
            - np.log1p(np.exp(-(mco - 4.0) / (4.0 * 0.05))))
    lw = lwco[None, None, :] - 0.5 * ((log_mbh[None, :, None] - log_mu[:, None, :]) / sigma) ** 2
    M = lw.max(-1, keepdims=True)
    p = np.exp(lw - M)
    I = 0.5 * ((p[..., :-1] + p[..., 1:]) * np.diff(mco)).sum(-1)
    return np.log(I) + M[..., 0] - 0.5 * np.log(2 * np.pi) - np.log(sigma) - log_mbh[None, :]


def test_pisn_grid():
    print("\n=== 2. PISN grid: max-subtracted trapezoid vs logaddexp+logsumexp ===")
    for i, pt in enumerate(PARAM_POINTS):
        s = full_sample(pt)
        zarr = jnp.expm1(jnp.linspace(0.0, jnp.log1p(s["zmax"]), 30))
        mpisns = s["mpisn"] + 1.5 * (1 - 1 / (1 + zarr))
        mbhmaxs = mpisns + s["dmbhmax"]

        gs = slow.LogDNDMPISN(s["a"], s["b"], mpisns, mbhmaxs, s["sigma"])
        gf = fast.LogDNDMPISN(s["a"], s["b"], mpisns, mbhmaxs, s["sigma"])
        # Both are float32 evaluations of the same integral, so compare each to
        # a float64 reference rather than to each other.
        ref = _pisn_reference(s["a"], s["b"], np.asarray(mpisns, np.float64),
                              np.asarray(mbhmaxs, np.float64), s["sigma"])
        mbh = np.asarray(gs.mbh_grid, np.float64)
        band = (mbh > 3.0) & (mbh < 100.0)
        e_slow = np.abs(np.asarray(gs.log_dN_grid, np.float64) - ref)[:, band].max()
        e_fast = np.abs(np.asarray(gf.log_dN_grid, np.float64) - ref)[:, band].max()
        status = "OK " if e_fast <= max(1e-3, 2 * e_slow) else "FAIL"
        if status == "FAIL":
            FAIL.append(f"pt{i}: fast log_dN_grid less accurate")
        print(f"  [{status}] pt{i}: log_dN_grid error vs float64 ref  "
              f"slow={e_slow:.3e}  fast={e_fast:.3e}")
        report(f"pt{i}: log_Z_grid", gs.log_Z_grid, gf.log_Z_grid, rtol=2e-4, atol=2e-4)
        report(f"pt{i}: mbh_grid", gs.mbh_grid, gf.mbh_grid, rtol=1e-6, atol=1e-4)


def test_population(mpisndot):
    print(f"\n=== 3. LogDNDMDQDV.__call__ sweep (mpisndot={mpisndot}) ===")
    m = np.exp(np.linspace(np.log(2.0), np.log(200.0), 137))
    q = np.linspace(0.02, 1.0, 41)
    z = np.expm1(np.linspace(np.log1p(1e-3), np.log1p(6.0), 23))
    M, Q, Z = np.meshgrid(m, q, z, indexing="ij")

    for i, pt in enumerate(PARAM_POINTS):
        s = full_sample(pt, mpisndot=mpisndot)
        ds = slow.build_population_model(s)
        df = fast.build_population_model(s)
        report(f"pt{i}: log_dndm(m,z)",
               ds.log_dndm(M[:, 0, :], Z[:, 0, :]), df.log_dndm(M[:, 0, :], Z[:, 0, :]),
               rtol=3e-4, atol=3e-4)
        report(f"pt{i}: log_dN(m1,q,z)", ds(M, Q, Z), df(M, Q, Z), rtol=3e-4, atol=3e-4)
        report(f"pt{i}: log_norm", ds.log_norm, df.log_norm, rtol=3e-4, atol=3e-4)


def test_full_model(nobs=400, nsamp=300, nsel=40000, mpisndot_free=False):
    print(f"\n=== 4. full model potential + gradient "
          f"(nobs={nobs}, nsamp={nsamp}, mpisndot_free={mpisndot_free}) ===")
    data = make_synthetic_data(nobs, nsamp, nsel, seed=3)
    prior = build_prior(not mpisndot_free, "/tmp/equiv_prior.prior")
    truth = {k: jnp.asarray(v) for k, v in TRUTH.items()
             if k in prior and not isinstance(prior[k], float)}

    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)

    from numpyro.infer.util import initialize_model, potential_energy
    from numpyro.infer import init_to_value

    outs = {}
    # tabulate_mass_function=False, smooth_tail_edge=False: this test checks
    # strict equivalence with the original evaluation, so both recommended
    # defaults are switched off.  The tabulated path and the smooth tail edge
    # are validated separately in test_tabulated_path (they *intentionally*
    # differ: the original's AD misses the contribution of samples crossing
    # the density's step at m = mbhmax, so its d/dh, d/dmpisn and d/ddmbhmax
    # disagree with finite differences of its own potential).
    exact = dict(tabulate_mass_function=False, smooth_tail_edge=False,
                 neff_penalty="min_neff")
    for label, mod, kwargs in (("slow", slow, {}),
                               ("fast", fast, dict(exact)),
                               ("fast/store", fast, dict(store_per_event=True, **exact))):
        mi = initialize_model(
            jax.random.PRNGKey(0), mod.pop_cosmo_model,
            model_args=model_args, model_kwargs=dict(use_low_bump=True, **kwargs),
            dynamic_args=False, init_strategy=init_to_value(values=truth),
        )
        z0 = mi.param_info.z
        v, g = jax.jit(jax.value_and_grad(mi.potential_fn))(z0)
        outs[label] = (float(v), {k: float(x) for k, x in g.items()}, sorted(z0))
        nan_g = [k for k, x in g.items() if not np.isfinite(x)]
        print(f"  {label:12s} potential={float(v):+.8e}  non-finite grads: {nan_g or 'none'}")

    vs, gs, _ = outs["slow"]
    vf, gf, _ = outs["fast"]
    report("potential energy", np.array([vs]), np.array([vf]), rtol=2e-4, atol=1e-2)
    keys = sorted(set(gs) & set(gf))
    report("gradient wrt all sampled params",
           np.array([gs[k] for k in keys]), np.array([gf[k] for k in keys]),
           rtol=5e-3, atol=5e-3)
    for k in keys:
        print(f"      d/d{k:12s} slow={gs[k]:+.6e}  fast={gf[k]:+.6e}")

    # And check that the deterministic sites the slow model loses are present.
    for label, mod in (("slow", slow), ("fast", fast)):
        names = set()
        with handlers.seed(rng_seed=0), handlers.substitute(data=truth):
            tr = handlers.trace(mod.pop_cosmo_model).get_trace(
                *model_args, use_low_bump=True)
        names = {n for n, s in tr.items() if s["type"] == "deterministic"}
        want = {"kappa", "mbhmax", "fpl", "flow"}
        print(f"  {label}: derived sites present = {sorted(want & names)}, "
              f"missing = {sorted(want - names)}")


def test_tabulated_path(nobs=400, nsamp=300, nsel=40000):
    """The tabulated mass-function path: same potential, and its AD gradient
    must agree with a finite difference of its own potential (which the direct
    path's does not, for parameters that move sample masses across the model's
    step discontinuity at m = mbhmax)."""
    print(f"\n=== 6. tabulated mass-function path (nobs={nobs}, nsamp={nsamp}) ===")
    data = make_synthetic_data(nobs, nsamp, nsel, seed=11)
    prior = build_prior(True, "/tmp/equiv_prior3.prior")
    truth = {k: jnp.asarray(v) for k, v in TRUTH.items()
             if k in prior and not isinstance(prior[k], float)}
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)

    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    outs = {}
    # neff_penalty is pinned to the original guard so the AD-vs-FD checks below
    # isolate the tabulation/smoothing behavior (the mc_variance default adds
    # its own kink at sum 1/n_eff = budget, which FD can straddle).
    for label, kwargs in (
        ("direct", dict(tabulate_mass_function=False, smooth_tail_edge=False,
                        neff_penalty="min_neff")),
        ("tab", dict(tabulate_mass_function=True, smooth_tail_edge=False,
                     neff_penalty="min_neff")),
        ("tab+smooth", dict(tabulate_mass_function=True, smooth_tail_edge=True,
                            neff_penalty="min_neff")),
    ):
        mi = initialize_model(
            jax.random.PRNGKey(0), fast.pop_cosmo_model,
            model_args=model_args, model_kwargs=dict(use_low_bump=True, **kwargs),
            dynamic_args=False, init_strategy=init_to_value(values=truth),
        )
        z0 = mi.param_info.z
        fn = jax.jit(mi.potential_fn)
        v, g = jax.jit(jax.value_and_grad(mi.potential_fn))(z0)
        outs[label] = (fn, z0, float(v), {k: float(x) for k, x in g.items()})
        print(f"  {label:10s} potential={float(v):+.8e}")

    _, _, vd, gd = outs["direct"]
    _, _, vt, gt = outs["tab"]
    report("tab vs direct: potential", np.array([vd]), np.array([vt]),
           rtol=1e-4, atol=0.5)
    edge_params = {"h", "mpisn", "dmbhmax"}
    keys = sorted(set(gd) & set(gt) - edge_params)
    report("tab vs direct: gradients (non-edge params)",
           np.array([gd[k] for k in keys]), np.array([gt[k] for k in keys]),
           rtol=2e-2, atol=5e-2)

    # AD-vs-FD self-consistency.  The tabulated path repairs d/dh (the query
    # points move across the table's smeared edge, so cell slopes carry the
    # jump).  d/dmpisn and d/ddmbhmax move the edge *through* the fixed nodes,
    # so their jump contribution stays invisible to AD in any evaluation
    # scheme -- only smooth_tail_edge (a continuous density) fixes those.
    # FD steps: small enough that curvature error is < 3% (at 1e-2 in mpisn
    # the FD is not converged and off by ~30%), large enough that float32
    # rounding of the potential stays a few % of the numerator.
    for label, must_pass in (("tab", {"h"}),
                             ("tab+smooth", {"h", "mpisn", "dmbhmax"})):
        fn_t, z0_t, _, gt_ = outs[label]
        for k in sorted(edge_params):
            eps = 1e-3 if k == "h" else 3e-3
            zp = dict(z0_t); zp[k] = z0_t[k] + eps
            zm = dict(z0_t); zm[k] = z0_t[k] - eps
            fd = (float(fn_t(zp)) - float(fn_t(zm))) / (2 * eps)
            ad = gt_[k]
            rel = abs(ad - fd) / max(abs(fd), 1e-6)
            if k in must_pass:
                ok = rel < 0.1
                if not ok:
                    FAIL.append(f"{label} AD-vs-FD inconsistent for {k}")
                tag = "OK " if ok else "FAIL"
            else:
                tag = "-- "  # informational: expected to disagree
            print(f"  [{tag}] {label:10s} d/d{k:8s} AD={ad:+.6e} FD={fd:+.6e} "
                  f"rel={rel:.2e}")


def test_nan_gradient_robustness():
    print("\n=== 5. gradient safety when an event has zero total weight ===")
    nobs, nsamp = 64, 32
    data = make_synthetic_data(nobs, nsamp, 5000, seed=7)
    # Force event 0 entirely out of support: masses below mbh_min.
    data["m1s_det"][0, :] = 0.5
    data["qs"][0, :] = 0.9
    prior = build_prior(True, "/tmp/equiv_prior2.prior")
    truth = {k: jnp.asarray(v) for k, v in TRUTH.items()
             if k in prior and not isinstance(prior[k], float)}
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)

    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    for label, mod in (("slow", slow), ("fast", fast)):
        try:
            mi = initialize_model(
                jax.random.PRNGKey(0), mod.pop_cosmo_model, model_args=model_args,
                model_kwargs=dict(use_low_bump=True), dynamic_args=False,
                init_strategy=init_to_value(values=truth),
            )
            v, g = jax.jit(jax.value_and_grad(mi.potential_fn))(mi.param_info.z)
            bad = [k for k, x in g.items() if not np.isfinite(float(x))]
            print(f"  {label:6s} potential={float(v):+.6e}  "
                  f"non-finite gradients: {bad or 'none'}")
            if bad and label == "fast":
                FAIL.append("fast model has NaN gradients on a dead event")
        except Exception as e:
            print(f"  {label:6s} initialize_model FAILED: {type(e).__name__}: {e}")
            if label == "fast":
                FAIL.append("fast model failed to initialize on a dead event")


def test_scatter_free_lookup_unit():
    """Direct check that the custom VJP matches naive reverse-mode through a
    lerp of a few-parameter table (no numpyro, no data arrays)."""
    print("\n=== 7. scatter-free table lookup: custom VJP vs naive AD ===")

    def build(a, b):
        # Tiny synthetic table depending on two scalars.
        x = jnp.linspace(0.0, 1.0, 64)
        tab = jnp.stack([a * jnp.exp(-b * x), a * b * x], axis=-1)
        return (0.0, 1.0), tab   # dummy aux

    a0, b0 = 1.7, 0.4
    ((_, _), tab), dtab, traced = fast._build_table_with_tangents(build, (a0, b0))
    t = jnp.array([0.0, 3.7, 12.2, 31.5, 63.0], dtype=jnp.float32)

    def loss_custom(a, b):
        ((_, _), tab_), dtab_, traced_ = fast._build_table_with_tangents(build, (a, b))
        c0, c1 = fast._scatter_free_lookup(tab_, dtab_, traced_, t)
        return jnp.sum(c0) + 0.3 * jnp.sum(c1)

    def loss_naive(a, b):
        _, tab_ = build(a, b)
        # Plain multi-lerp WITHOUT stop_gradient -- reverse mode scatters into tab.
        out = fast._multi_lerp(tab_, t)
        return jnp.sum(out[..., 0]) + 0.3 * jnp.sum(out[..., 1])

    g_c = jax.grad(loss_custom, argnums=(0, 1))(a0, b0)
    g_n = jax.grad(loss_naive, argnums=(0, 1))(a0, b0)
    report("d(loss)/da custom vs naive", np.array([float(g_c[0])]),
           np.array([float(g_n[0])]), rtol=1e-5, atol=1e-5)
    report("d(loss)/db custom vs naive", np.array([float(g_c[1])]),
           np.array([float(g_n[1])]), rtol=1e-5, atol=1e-5)

    # Static parameters: no tangent tables, falls back to replicated lerp.
    ((_, _), tab_s), dtab_s, traced_s = fast._build_table_with_tangents(
        build, (float(a0), float(b0))
    )
    assert dtab_s is None and traced_s == ()
    c0, c1 = fast._scatter_free_lookup(tab_s, dtab_s, traced_s, t)
    c0n, c1n = fast._scatter_free_lookup(tab, dtab, traced, t)
    report("static-fallback channel0 vs traced", c0, c0n, rtol=1e-6, atol=1e-6)
    report("static-fallback channel1 vs traced", c1, c1n, rtol=1e-6, atol=1e-6)


def test_full_cosmo(nobs=400, nsamp=300, nsel=40000):
    """Sample Om and w: potential/gradients match the slow module, and the
    fast AD gradients agree with finite differences of the potential."""
    print(f"\n=== 8. full cosmology (Om, w sampled; nobs={nobs}, nsamp={nsamp}) ===")
    data = make_synthetic_data(nobs, nsamp, nsel, seed=13)
    prior = build_prior(True, "/tmp/equiv_prior_cosmo.prior", cosmo_free=True)
    truth = {k: jnp.asarray(v) for k, v in TRUTH.items()
             if k in prior and not isinstance(prior[k], float)}
    assert "Om" in truth and "w" in truth
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)

    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    # Exact-reproduction mode so we can compare to the slow module.
    exact = dict(tabulate_mass_function=False, smooth_tail_edge=False,
                 neff_penalty="min_neff")
    outs = {}
    for label, mod, kwargs in (("slow", slow, {}),
                               ("fast", fast, dict(exact)),
                               ("fast/tab", fast, dict(tabulate_mass_function=True,
                                                       smooth_tail_edge=True,
                                                       neff_penalty="min_neff"))):
        mi = initialize_model(
            jax.random.PRNGKey(0), mod.pop_cosmo_model,
            model_args=model_args, model_kwargs=dict(use_low_bump=True, **kwargs),
            dynamic_args=False, init_strategy=init_to_value(values=truth),
        )
        z0 = mi.param_info.z
        fn = jax.jit(mi.potential_fn)
        v, g = jax.jit(jax.value_and_grad(mi.potential_fn))(z0)
        outs[label] = (fn, z0, float(v), {k: float(x) for k, x in g.items()})
        bad = [k for k, x in g.items() if not np.isfinite(float(x))]
        print(f"  {label:10s} potential={float(v):+.8e}  non-finite grads: {bad or 'none'}")
        if bad and label.startswith("fast"):
            FAIL.append(f"{label} has non-finite grads with Om/w free")

    vs, gs = outs["slow"][2], outs["slow"][3]
    vf, gf = outs["fast"][2], outs["fast"][3]
    report("cosmo_free potential slow vs fast", np.array([vs]), np.array([vf]),
           rtol=2e-4, atol=1e-2)
    keys = sorted(set(gs) & set(gf))
    report("cosmo_free gradients slow vs fast",
           np.array([gs[k] for k in keys]), np.array([gf[k] for k in keys]),
           rtol=5e-3, atol=5e-3)
    for k in ("Om", "w", "h"):
        if k in gs and k in gf:
            print(f"      d/d{k:4s} slow={gs[k]:+.6e}  fast={gf[k]:+.6e}")

    # AD-vs-FD on the production (tabulated + smooth) path for the cosmology
    # parameters -- these are the ones whose gradients used to go through the
    # table scatter.
    fn_t, z0_t, _, gt = outs["fast/tab"]
    for k, eps in (("Om", 3e-4), ("w", 3e-4), ("h", 1e-3)):
        zp = dict(z0_t); zp[k] = z0_t[k] + eps
        zm = dict(z0_t); zm[k] = z0_t[k] - eps
        fd = (float(fn_t(zp)) - float(fn_t(zm))) / (2 * eps)
        ad = gt[k]
        rel = abs(ad - fd) / max(abs(fd), 1e-6)
        ok = rel < 0.1
        if not ok:
            FAIL.append(f"cosmo_free AD-vs-FD inconsistent for {k}")
        print(f"  [{'OK ' if ok else 'FAIL'}] fast/tab d/d{k:4s} "
              f"AD={ad:+.6e} FD={fd:+.6e} rel={rel:.2e}")


def _likelihood_log_factor(model, model_args, model_kwargs, params):
    """Sum of the model's factor sites (loglike + selfactor + neff guards),
    excluding the prior.  Used to compare reparameterizations whose priors
    differ even at the same physical point."""
    with handlers.seed(rng_seed=0), handlers.substitute(data=params):
        tr = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    total = 0.0
    for site in tr.values():
        if site["type"] != "sample":
            continue
        fn = site.get("fn")
        if fn is not None and type(fn).__name__ == "Unit":
            total = total + fn.log_factor
    return total


def test_omh2_reparam(nobs=400, nsamp=300, nsel=40000):
    """Omh2 parameterization: Om is derived, likelihood matches the Om
    parameterization at the truth point, and d(like)/dOmh2 follows the chain
    rule.  (Full potentials are *not* compared: the Om and Omh2 priors differ.)
    """
    print(f"\n=== 9. Omh2 reparameterization (nobs={nobs}, nsamp={nsamp}) ===")
    data = make_synthetic_data(nobs, nsamp, nsel, seed=17)
    prior_om = build_prior(True, "/tmp/equiv_prior_om.prior",
                           cosmo_free=True, omh2=False)
    prior_h2 = build_prior(True, "/tmp/equiv_prior_omh2.prior",
                           cosmo_free=True, omh2=True)
    truth_om = {k: jnp.asarray(v) for k, v in TRUTH.items()
                if k in prior_om and not isinstance(prior_om[k], float)}
    truth_h2 = {k: jnp.asarray(v) for k, v in TRUTH.items()
                if k in prior_h2 and not isinstance(prior_h2[k], float)}
    assert "Om" in truth_om and "Omh2" in truth_h2 and "Om" not in truth_h2

    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    kwargs = dict(use_low_bump=True, tabulate_mass_function=True,
                  smooth_tail_edge=True, neff_penalty="min_neff")
    args_om = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
               data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
               data["pdraw_sel"], data["Ndraw"], prior_om)
    args_h2 = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
               data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
               data["pdraw_sel"], data["Ndraw"], prior_h2)

    like_om = float(_likelihood_log_factor(
        fast.pop_cosmo_model, args_om, kwargs, truth_om))
    like_h2 = float(_likelihood_log_factor(
        fast.pop_cosmo_model, args_h2, kwargs, truth_h2))
    report("Omh2 vs Om param: likelihood at truth",
           np.array([like_om]), np.array([like_h2]), rtol=1e-5, atol=1e-3)

    # Deterministic Om = Omh2/h^2 must appear in the trace.
    with handlers.seed(rng_seed=0), handlers.substitute(data=truth_h2):
        tr = handlers.trace(fast.pop_cosmo_model).get_trace(*args_h2, **kwargs)
    Om_det = float(tr["Om"]["value"])
    Om_expect = float(TRUTH["Omh2"] / TRUTH["h"] ** 2)
    report("derived Om == Omh2/h^2", np.array([Om_det]), np.array([Om_expect]),
           rtol=1e-6, atol=1e-6)

    # Likelihood-only gradients via the chain rule at fixed h:
    # dL/dOmh2 = (dL/dOm) * (1/h^2).
    def like_om_fn(Om):
        p = dict(truth_om)
        p["Om"] = Om
        return _likelihood_log_factor(fast.pop_cosmo_model, args_om, kwargs, p)

    def like_h2_fn(Omh2):
        p = dict(truth_h2)
        p["Omh2"] = Omh2
        return _likelihood_log_factor(fast.pop_cosmo_model, args_h2, kwargs, p)

    dL_dOm = float(jax.grad(like_om_fn)(truth_om["Om"]))
    dL_dOmh2 = float(jax.grad(like_h2_fn)(truth_h2["Omh2"]))
    chain = dL_dOm / (TRUTH["h"] ** 2)
    report("dL/dOmh2 vs (dL/dOm)/h^2",
           np.array([dL_dOmh2]), np.array([chain]), rtol=5e-3, atol=5e-3)
    print(f"      dL/dOmh2={dL_dOmh2:+.6e}  (dL/dOm)/h^2={chain:+.6e}")

    # And the full potential under the Omh2 prior must have finite gradients.
    mi = initialize_model(
        jax.random.PRNGKey(0), fast.pop_cosmo_model,
        model_args=args_h2, model_kwargs=kwargs, dynamic_args=False,
        init_strategy=init_to_value(values=truth_h2),
    )
    v, g = jax.jit(jax.value_and_grad(mi.potential_fn))(mi.param_info.z)
    bad = [k for k, x in g.items() if not np.isfinite(float(x))]
    print(f"  Omh2 potential={float(v):+.8e}  non-finite grads: {bad or 'none'}")
    if bad:
        FAIL.append(f"Omh2 param has non-finite grads: {bad}")
    else:
        print("  [OK ] all Omh2-parameterization gradients finite")


if __name__ == "__main__":
    test_cosmology()
    test_pisn_grid()
    test_population(mpisndot=0.0)
    test_population(mpisndot=1.5)
    test_full_model(mpisndot_free=False)
    test_full_model(mpisndot_free=True)
    test_tabulated_path()
    test_nan_gradient_robustness()
    test_scatter_free_lookup_unit()
    test_full_cosmo()
    test_omh2_reparam()

    print("\n" + "=" * 70)
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all equivalence checks passed")

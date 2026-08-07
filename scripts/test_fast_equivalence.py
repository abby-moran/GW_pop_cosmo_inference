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
    # Omh2 is an edge parameter for the same reason h is: it moves z(dL) and
    # therefore every source-frame mass across the m = mbhmax discontinuity
    # (added when the default prior switched from fixed Om to sampled Omh2).
    edge_params = {"h", "Omh2", "mpisn", "dmbhmax"}
    keys = sorted(set(gd) & set(gt) - edge_params)
    report("tab vs direct: gradients (non-edge params)",
           np.array([gd[k] for k in keys]), np.array([gt[k] for k in keys]),
           rtol=2e-2, atol=5e-2)

    # AD-vs-FD self-consistency.  The tabulated path repairs d/dh and d/dOmh2
    # (the query points move across the table's smeared edge, so cell slopes
    # carry the jump).  d/dmpisn and d/ddmbhmax move the edge *through* the
    # fixed nodes, so their jump contribution stays invisible to AD in any
    # evaluation scheme -- only smooth_tail_edge (a continuous density) fixes
    # those.  FD steps: small enough that curvature error is < 3% (at 1e-2 in
    # mpisn the FD is not converged and off by ~30%), large enough that
    # float32 rounding of the potential stays a few % of the numerator.
    for label, must_pass in (("tab", {"h", "Omh2"}),
                             ("tab+smooth", {"h", "Omh2", "mpisn", "dmbhmax"})):
        fn_t, z0_t, _, gt_ = outs[label]
        for k in sorted(edge_params):
            eps = 1e-3 if k in ("h", "Omh2") else 3e-3
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


def test_tabulated_path_zdep(nobs=400, nsamp=300, nsel=40000):
    """The 2-D (z-dependent) tabulated path, active when mpisndot is sampled:
    same potential as the direct evaluation, and AD gradients that agree with
    finite differences of its own potential -- including d/dmpisndot, whose
    direct-path gradient cannot see samples crossing the moving mbhmax(z)
    edge."""
    print(f"\n=== 7. 2-D tabulated path, mpisndot sampled "
          f"(nobs={nobs}, nsamp={nsamp}) ===")
    data = make_synthetic_data(nobs, nsamp, nsel, seed=19)
    prior = build_prior(False, "/tmp/equiv_prior4.prior")   # mpisndot free
    truth = {k: jnp.asarray(v) for k, v in TRUTH.items()
             if k in prior and not isinstance(prior[k], float)}
    # A nonzero evolution rate so the z axis of the table actually matters.
    truth["mpisndot"] = jnp.asarray(3.0)
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)

    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    outs = {}
    # neff_penalty="none" here: the min_neff guard takes a min over events,
    # which kinks whenever the argmin event switches; with mpisndot free the
    # FD probes below straddle such kinks and would contaminate the AD-vs-FD
    # comparison with guard noise unrelated to the table.  The guards' own
    # equivalence is covered by test 6.
    for label, kwargs in (
        ("direct", dict(tabulate_mass_function=False, smooth_tail_edge=False)),
        ("tab2d", dict(tabulate_mass_function=True, smooth_tail_edge=False)),
        ("direct+smooth", dict(tabulate_mass_function=False, smooth_tail_edge=True)),
        ("tab2d+smooth", dict(tabulate_mass_function=True, smooth_tail_edge=True)),
    ):
        mi = initialize_model(
            jax.random.PRNGKey(0), fast.pop_cosmo_model,
            model_args=model_args,
            model_kwargs=dict(use_low_bump=True, neff_penalty="none", **kwargs),
            dynamic_args=False, init_strategy=init_to_value(values=truth),
        )
        z0 = mi.param_info.z
        fn = jax.jit(mi.potential_fn)
        v, g = jax.jit(jax.value_and_grad(mi.potential_fn))(z0)
        outs[label] = (fn, z0, float(v), {k: float(x) for k, x in g.items()})
        bad = [k for k, x in g.items() if not np.isfinite(x)]
        print(f"  {label:14s} potential={float(v):+.8e}  "
              f"non-finite grads: {bad or 'none'}")
        if bad:
            FAIL.append(f"{label} non-finite grads with mpisndot free")

    # Selection is on the direct path in both tab2d models, so the
    # nobs-amplified selfactor is common and only the per-event z-lerp
    # differences remain.
    _, _, vd, gd = outs["direct"]
    _, _, vt, gt = outs["tab2d"]
    report("tab2d vs direct: potential (hard edge)", np.array([vd]), np.array([vt]),
           rtol=1e-4, atol=0.5)
    # Gradients are compared on the smooth pair (the production default): the
    # hard edge makes the density discontinuous at the moving mbhmax(z), so
    # z-lerping across the step amplifies tab-vs-direct gradient differences
    # that the smooth model does not have.
    _, _, vds, gds = outs["direct+smooth"]
    _, _, vts, gts = outs["tab2d+smooth"]
    report("tab2d vs direct: potential (smooth edge)", np.array([vds]),
           np.array([vts]), rtol=1e-4, atol=0.5)
    keys = sorted(set(gds) & set(gts))
    report("tab2d vs direct: gradients, all params (smooth edge)",
           np.array([gds[k] for k in keys]), np.array([gts[k] for k in keys]),
           rtol=2e-2, atol=0.2)

    # AD-vs-FD self-consistency on the recommended (tab2d+smooth) path.
    # NOTE eps=1e-3 for every parameter: the PISN remnant map's
    # `where(mco < mpisn, ...)` branch kinks the potential each time some
    # mpisn(z_i) crosses an mco grid node, and with 30 z slices those kinks
    # are ~9e-3 apart in mpisn -- an FD step of 3e-3 straddles them and
    # produces garbage (verified: FD swings 81 -> 69 -> 30 -> 41 over
    # eps = 1e-3..3e-2, identically for the direct path, so it is model
    # structure, not the table).  AD differentiates the actual branch and is
    # the trustworthy side there.
    # tab2d (hard edge) is informational only: unlike the 1-D case, d/dh FD
    # is not a usable reference here -- an h step moves every sample across
    # the hard tail edge at 30 distinct mbhmax(z_i) positions, so the FD
    # probe straddles discontinuities at any step size.  The hard edge is a
    # legacy-exact mode, not recommended for sampling.
    edge_params = {"h", "mpisn", "dmbhmax", "mpisndot"}
    for label, must_pass in (("tab2d", set()),
                             ("tab2d+smooth", {"h", "mpisn", "dmbhmax",
                                               "mpisndot"})):
        fn_t, z0_t, _, gt_ = outs[label]
        for k in sorted(edge_params):
            eps = 1e-3
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
                tag = "-- "
            print(f"  [{tag}] {label:12s} d/d{k:9s} AD={ad:+.6e} FD={fd:+.6e} "
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


if __name__ == "__main__":
    test_cosmology()
    test_pisn_grid()
    test_population(mpisndot=0.0)
    test_population(mpisndot=1.5)
    test_full_model(mpisndot_free=False)
    test_full_model(mpisndot_free=True)
    test_tabulated_path()
    test_tabulated_path_zdep()
    test_nan_gradient_robustness()

    print("\n" + "=" * 70)
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all equivalence checks passed")

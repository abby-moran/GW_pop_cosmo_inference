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

    # tab2d vs direct is NOT an equivalence check at the production n_z.  Both
    # sides of the ratio use the table now (tabulate_selection follows
    # tabulate_mass_function), so the z-lerp bias is common-mode and cancels --
    # but what is left is a genuine O(dz^2) discretization difference between
    # two slightly different models, and at n_z=30 with mpisndot=3 that is
    # ~1.2 nats at nobs=400.  Asserting a tight tolerance there would only be
    # satisfiable by the broken split path (which agrees with direct on the
    # selection factor by construction, and so looked deceptively good).
    #
    # The meaningful assertion is convergence: tab2d -> direct as n_z grows,
    # at the second-order rate linear interpolation must have.  Measured
    # 1.216 / 0.383 / 0.097 / 0.024 nats for n_z = 30 / 60 / 120 / 240.
    _, _, vds, gds = outs["direct+smooth"]
    _, _, vts, gts = outs["tab2d+smooth"]
    print(f"  tab2d - direct at production n_z=30: {vts - vds:+.4f} nats "
          f"(informational; see convergence below)")
    prev = None
    for nz in (30, 60, 120):
        pair = {}
        for label, kwargs in (("d", dict(tabulate_mass_function=False)),
                              ("t", dict(tabulate_mass_function=True))):
            mi = initialize_model(
                jax.random.PRNGKey(0), fast.pop_cosmo_model, model_args=model_args,
                model_kwargs=dict(use_low_bump=True, neff_penalty="none",
                                  smooth_tail_edge=True, n_z=nz, **kwargs),
                dynamic_args=False, init_strategy=init_to_value(values=truth),
            )
            pair[label] = float(jax.jit(mi.potential_fn)(mi.param_info.z))
        d = abs(pair["t"] - pair["d"])
        rate = "" if prev is None else f"  ({prev / max(d, 1e-9):.1f}x smaller)"
        # Linear interpolation is second order, so each doubling of n_z should
        # cut the gap ~4x; require a clear 2.5x to allow for float32 noise.
        if prev is not None and prev / max(d, 1e-9) < 2.5:
            FAIL.append(f"tab2d vs direct does not converge in n_z "
                        f"({prev:.4f} -> {d:.4f} at n_z={nz})")
        print(f"  n_z={nz:4d}: |tab2d - direct| = {d:9.5f} nats{rate}")
        prev = d

    # AD-vs-FD self-consistency on the recommended (tab2d+smooth) path.
    # NOTE eps=1e-3 for every parameter: the PISN remnant map's
    # `where(mco < mpisn, ...)` branch kinks the potential each time some
    # mpisn(z_i) crosses an mco grid node, and with 30 z slices those kinks
    # are ~9e-3 apart in mpisn -- an FD step of 3e-3 straddles them and
    # produces garbage (verified: FD swings 81 -> 69 -> 30 -> 41 over
    # eps = 1e-3..3e-2, identically for the direct path, so it is model
    # structure, not the table).
    #
    # What this checks is IMPLEMENTATION self-consistency only: that AD
    # returns the derivative of the potential actually evaluated (a tiny-eps
    # FD measures the same n_z=30 surface, ripples included).  It says
    # nothing about closeness to the z-continuum gradient -- at this
    # adversarial point (mpisndot=3) the n_z=30 value is dominated by a grid
    # ripple and is ~3x the n_z->inf answer (d/dmpisn ~ 93 here vs ~29
    # converged; see notes/2026-08-08-tabulated-selection-consistency.md).
    # That costs HMC efficiency, not correctness, and the ripple is
    # negligible where the real data puts posterior mass.
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


def test_tabulated_selection_consistency(nobs=400, nsamp=300, nsel=40000):
    """The tabulated event samples and the tabulated selection set must use the
    SAME density.

    The R-marginalized hierarchical likelihood is prod_i lambda(x_i) /
    (int lambda p_det)^nobs, so a table on the numerator and a direct
    evaluation on the denominator is not a probability model: the numerator's
    interpolation error survives uncancelled, it is parameter dependent, and
    the sampler climbs it.  That is what drove the mpisndot-free mock runs onto
    the prior walls (runs/endO5_evo: +125 nats of spurious log likelihood at
    the wall point, beating the truth by 99).

    Tested as an exact identity rather than a tolerance.  Feed the model a
    selection set that IS the event samples, flattened, with the matching
    pdraw.  Then the same points go through both code paths and

        log_mu_sel + log(Ndraw) == logsumexp_i(loglike_i) + log(nsamp)

    holds to roundoff *if and only if* the two paths evaluate the same
    density.  No tolerance on a physical difference, no dependence on how the
    bench data happens to distribute its selection samples -- and it cannot go
    blind the way an "is the split visibly worse" check can.  (It can be
    worse or better depending on how the two populations' interpolation
    biases happen to line up: on the real 9000-event data the split cost +125
    nats, but on this synthetic bench data, whose selection samples are drawn
    from a much broader population than its events, it lands closer to the
    direct path by luck.  Luck is not a guard.)"""
    print(f"\n=== 8. tabulated selection consistency (nobs={nobs}) ===")
    data = make_synthetic_data(nobs, nsamp, nsel, seed=19)
    prior = build_prior(False, "/tmp/equiv_prior5.prior")   # mpisndot free

    # Selection set := the event samples themselves.
    npe = nobs * nsamp
    Ndraw = float(npe)
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det"].ravel(), data["qs"].ravel(),
                  data["dls"].ravel(), np.exp(data["log_pdraw"]).ravel(),
                  Ndraw, prior)

    import numpyro.handlers as handlers

    # Both the truth and a corner like the one runs/endO5_evo walked to: the
    # mass function as narrow and as fast-moving in z as the priors allow,
    # where a 30-node z-lerp is least accurate.  Nudged just *inside* the
    # bounds -- exactly on them the truncated log prob is -inf.
    # `sharp` mirrors the actual failed mode of runs/endO5_evo.
    sharp = dict(TRUTH, sigma=0.051, dmbhmax=0.52, mpisndot=-1.99, a=-1.6,
                 b=-2.0, c=1.5, mpisn=38.5, log_fpl=0.577, beta=-3.78)
    resid = {}
    for label, point in (("truth", TRUTH), ("sharp", sharp)):
        vals = {k: jnp.asarray(v) for k, v in point.items()
                if k in prior and not isinstance(prior[k], float)}
        for cl, kw in (("default", dict()), ("split", dict(tabulate_selection=False))):
            with handlers.seed(rng_seed=0), handlers.substitute(data=vals):
                tr = handlers.trace(fast.pop_cosmo_model).get_trace(
                    *model_args, use_low_bump=True, neff_penalty="none",
                    store_per_event=True, **kw)
            loglike = np.asarray(tr["loglik_array_dim"]["value"], dtype=np.float64)
            log_mu_sel = float(np.asarray(tr["log_mu_sel"]["value"]))
            lhs = log_mu_sel + np.log(Ndraw)
            mx = loglike.max()
            rhs = float(mx + np.log(np.exp(loglike - mx).sum())) + np.log(nsamp)
            d = lhs - rhs
            resid[(label, cl)] = d
            # Only the default is asserted, and only that the identity holds:
            # it is structural, so anything above float32 roundoff on a
            # 1.2e5-term logsumexp means the two paths have drifted apart.
            # Threshold budgeted against production, where `selfactor`
            # multiplies this residual by nobs = 9000: 1e-4 nats caps the
            # permitted distortion at 0.9 nats of potential there, while the
            # measured float32 roundoff of the identity is < 5e-6 nats
            # (loglike and log_mu_sel are accumulated by different reduction
            # orders), so the threshold keeps a ~20x margin above noise.
            if cl == "default":
                ok = abs(d) < 1e-4
                if not ok:
                    FAIL.append(f"tabulated selection ({label}): identity broken, "
                                f"residual {d:.5f} nats -> {abs(d) * nobs:.1f} nats "
                                f"of potential at nobs={nobs}")
                tag = "OK " if ok else "FAIL"
            else:
                tag = "-- "
            print(f"  [{tag}] {label:6s} {cl:8s} identity residual = {d:+9.5f} nats"
                  f"  -> {d * nobs:+9.2f} nats of potential at nobs={nobs}")

    # The check must have teeth: at the sharp point the split has to be clearly
    # worse than the default, or this test is no longer a guard against the
    # split coming back.
    if abs(resid[("sharp", "split")]) < 10 * max(abs(resid[("sharp", "default")]), 1e-5):
        FAIL.append("tabulated selection: the sharp point no longer separates the "
                    "split from the default -- the regression guard has gone blind")


def test_scatter_free_vjp(nobs=400, nsamp=300, nsel=40000):
    """The scatter-free (tangent-table custom-VJP) backward must reproduce the
    ordinary reverse-mode gradient of the tabulated path.

    The two compute the same chain rule by different routes -- ordinary AD
    scatters d(potential)/d(table) into the table and contracts it with
    dT/dtheta, the custom VJP contracts per-point tangent-table lookups
    directly -- so they agree up to float32 summation order.  Checked for
    both table layouts (1-D log-m when mpisndot is pinned to 0, 2-D
    z x log-m when it is sampled) at the truth and at an edge-heavy point.
    The potential itself must be bit-identical: the forward pass is the same
    arithmetic either way."""
    print(f"\n=== 9. scatter-free VJP == replicated-scatter gradient (nobs={nobs}) ===")
    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    data = make_synthetic_data(nobs, nsamp, nsel, seed=11)
    edge = dict(TRUTH, sigma=0.06, mpisn=44.0, dmbhmax=0.8, c=6.5, h=0.9,
                mpisndot=2.5)

    for mpisndot_free in (False, True):
        prior = build_prior(not mpisndot_free, "/tmp/equiv_prior6.prior")
        model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                      data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                      data["pdraw_sel"], data["Ndraw"], prior)
        for label, point in (("truth", TRUTH), ("edge", edge)):
            vals = {k: jnp.asarray(v) for k, v in point.items()
                    if k in prior and not isinstance(prior[k], float)}
            out = {}
            for sf in (True, False):
                mi = initialize_model(
                    jax.random.PRNGKey(0), fast.pop_cosmo_model,
                    model_args=model_args,
                    model_kwargs=dict(use_low_bump=True, scatter_free_tables=sf),
                    dynamic_args=False, init_strategy=init_to_value(values=vals),
                )
                z0 = mi.param_info.z
                v = float(jax.jit(mi.potential_fn)(z0))
                g = jax.jit(jax.grad(mi.potential_fn))(z0)
                out[sf] = (v, {k: float(x) for k, x in g.items()})
            v1, g1 = out[True]
            v0, g0 = out[False]
            tag = f"zdep={mpisndot_free} {label}"
            if v1 != v0:
                FAIL.append(f"scatter-free VJP ({tag}): potential differs "
                            f"({v1!r} vs {v0!r})")
            gv1 = np.array([g1[k] for k in sorted(g1)])
            gv0 = np.array([g0[k] for k in sorted(g0)])
            # Normalize by the gradient's overall scale, not per-component:
            # a component passing through zero has no meaningful relative
            # error of its own.
            scale = max(np.abs(gv0).max(), 1e-10)
            worst = np.abs(gv1 - gv0).max() / scale
            ok = worst < 5e-4
            if not ok:
                FAIL.append(f"scatter-free VJP ({tag}): gradients differ, "
                            f"worst {worst:.2e} of gradient scale")
            print(f"  [{'OK ' if ok else 'FAIL'}] {tag:22s} dV={v1 - v0:+.1e}  "
                  f"worst |dg|/scale = {worst:.2e}")


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


def test_reparam_equivalence(nobs=400, nsamp=300, nsel=40000):
    """Reparameterized sampling (log_h / log_sigma / log_mp_low /
    log_mpisn_ref + zpivot) must hit the same likelihood as the canonical
    parameterization at matched points.

    The reparam coordinates are chosen first and the canonical values are
    derived with the same float32 ops the model uses (jnp.exp, the pivot
    subtraction), so every likelihood factor should agree to float32
    round-off.  Prior densities of course differ; only likelihood factors
    and derived deterministics are compared."""
    print(f"\n=== 10. reparameterized sampling: matched-point likelihood ===")
    data = make_synthetic_data(nobs, nsamp, nsel, seed=7)

    base_prior = build_prior(False, "/tmp/equiv_prior_base.prior")

    zpivot = 1.1
    xpivot = zpivot / (1.0 + zpivot)
    reparam_text = """log_h = TruncatedNormal(np.log(0.7), 0.29, low=np.log(0.4), high=np.log(1.2))
Omh2 = TruncatedNormal(0.143, 0.05, low=0.02, high=0.4)
w = -1
a = TruncatedNormal(2.35, 2, low=-1.65, high=6.35)
b = TruncatedNormal(1.9, 2, low=-2.1, high=5.9)
c = TruncatedNormal(4, 2, low=0, high=8)
log_mpisn_ref = TruncatedNormal(np.log(35.0), 0.145, low=np.log(20.0), high=np.log(50.0))
dmbhmax = TruncatedNormal(3.0, 2.0, low=0.5, high=7.0)
log_sigma = Normal(np.log(0.1), 1.0)
beta = Normal(0, 2)
log_fpl = Uniform(np.log(1e-2), np.log(2))
lam = TruncatedNormal(2.7, 2.0, low=-1.3, high=6.7)
dkappa = TruncatedNormal(2.9, 2.0, low=1, high=6.9)
zp = TruncatedNormal(1.9, 1, low=0, high=3.9)
log_mp_low = TruncatedNormal(np.log(9.0), 0.22, low=np.log(5.0), high=np.log(15.0))
msigma_low = TruncatedNormal(4.0, 2.0, low=0.5, high=8.0)
log_flow = Uniform(np.log(1e-3), np.log(2))
mpisndot = Uniform(low=-2, high=8)
zpivot = %s
zmax = 6.5
mbh_min = 3.0
delta_m = 1.6
""" % zpivot
    with open("/tmp/equiv_prior_reparam.prior", "w") as f:
        f.write(reparam_text)
    from utils import get_priors_from_file
    reparam_prior = get_priors_from_file("/tmp/equiv_prior_reparam.prior")

    factor_sites = ("loglike", "selfactor", "neff_criteria", "neff_sel_criteria")
    derived_sites = ("h", "sigma", "mp_low", "mpisn", "mbhmax", "Om", "kappa")

    def trace_model(prior, subs):
        model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                      data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                      data["pdraw_sel"], data["Ndraw"], prior)
        with handlers.seed(rng_seed=0), handlers.substitute(data=subs):
            return handlers.trace(fast.pop_cosmo_model).get_trace(
                *model_args, use_low_bump=True)

    for i, (mpisndot, dsig) in enumerate([(0.0, 0.0), (2.5, 0.4), (-1.0, -0.6)]):
        # reparam point first; canonical values derived with the model's ops
        rp = dict(
            log_h=jnp.asarray(np.log(0.674)),
            log_mpisn_ref=jnp.asarray(np.log(33.29 + 1.0 * i)),
            log_sigma=jnp.asarray(np.log(0.09) + dsig),
            log_mp_low=jnp.asarray(np.log(9.121)),
            mpisndot=jnp.asarray(mpisndot),
            Omh2=jnp.asarray(TRUTH["Omh2"]), a=jnp.asarray(TRUTH["a"]),
            b=jnp.asarray(TRUTH["b"]), c=jnp.asarray(TRUTH["c"]),
            dmbhmax=jnp.asarray(TRUTH["dmbhmax"]), beta=jnp.asarray(TRUTH["beta"]),
            log_fpl=jnp.asarray(TRUTH["log_fpl"]), lam=jnp.asarray(TRUTH["lam"]),
            dkappa=jnp.asarray(TRUTH["dkappa"]), zp=jnp.asarray(TRUTH["zp"]),
            msigma_low=jnp.asarray(TRUTH["msigma_low"]),
            log_flow=jnp.asarray(TRUTH["log_flow"]),
            R_unit=jnp.asarray(0.0),
        )
        cp = dict(rp)
        for name in ("log_h", "log_mpisn_ref", "log_sigma", "log_mp_low"):
            del cp[name]
        cp["h"] = jnp.exp(rp["log_h"])
        cp["sigma"] = jnp.exp(rp["log_sigma"])
        cp["mp_low"] = jnp.exp(rp["log_mp_low"])
        cp["mpisn"] = jnp.exp(rp["log_mpisn_ref"]) - rp["mpisndot"] * xpivot

        tr_base = trace_model(base_prior, cp)
        tr_rep = trace_model(reparam_prior, rp)

        for site in factor_sites:
            a = np.asarray(tr_base[site]["fn"].log_factor)
            b = np.asarray(tr_rep[site]["fn"].log_factor)
            report(f"pt{i} (mpisndot={mpisndot}) factor {site}", a, b,
                   rtol=1e-6, atol=1e-4)
        for site in derived_sites:
            a = np.asarray(tr_base[site]["value"] if site in tr_base
                           else tr_base[site])
            b = np.asarray(tr_rep[site]["value"])
            report(f"pt{i} derived {site}", np.atleast_1d(a), np.atleast_1d(b),
                   rtol=1e-6, atol=1e-6)

    # And the reparam model must initialize with finite gradients wrt the
    # *new* coordinates.
    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value
    truth_rp = {k: v for k, v in dict(
        log_h=np.log(0.674), log_mpisn_ref=np.log(33.29),
        log_sigma=np.log(0.0539), log_mp_low=np.log(9.121), mpisndot=0.0,
        Omh2=TRUTH["Omh2"], a=TRUTH["a"], b=TRUTH["b"], c=TRUTH["c"],
        dmbhmax=TRUTH["dmbhmax"], beta=TRUTH["beta"], log_fpl=TRUTH["log_fpl"],
        lam=TRUTH["lam"], dkappa=TRUTH["dkappa"], zp=TRUTH["zp"],
        msigma_low=TRUTH["msigma_low"], log_flow=TRUTH["log_flow"],
    ).items()}
    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], reparam_prior)
    mi = initialize_model(
        jax.random.PRNGKey(0), fast.pop_cosmo_model, model_args=model_args,
        model_kwargs=dict(use_low_bump=True), dynamic_args=False,
        init_strategy=init_to_value(values={k: jnp.asarray(v)
                                            for k, v in truth_rp.items()}),
    )
    v, g = jax.jit(jax.value_and_grad(mi.potential_fn))(mi.param_info.z)
    bad = [k for k, x in g.items() if not np.isfinite(float(x))]
    print(f"  reparam potential={float(v):+.6e}  non-finite grads: {bad or 'none'}")
    if bad:
        FAIL.append("reparam model has non-finite gradients at truth")


if __name__ == "__main__":
    test_cosmology()
    test_pisn_grid()
    test_population(mpisndot=0.0)
    test_population(mpisndot=1.5)
    test_full_model(mpisndot_free=False)
    test_full_model(mpisndot_free=True)
    test_tabulated_path()
    test_tabulated_path_zdep()
    test_tabulated_selection_consistency()
    test_scatter_free_vjp()
    test_nan_gradient_robustness()
    test_reparam_equivalence()

    print("\n" + "=" * 70)
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all equivalence checks passed")

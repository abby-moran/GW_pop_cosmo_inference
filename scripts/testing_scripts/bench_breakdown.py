"""
Where does the remaining gradient time in intensity_models_fast go, and do
scatter-replica count or data pre-sorting buy anything more?

Pieces timed (value_and_grad wrt the 15 sampled params, at the truth point):
  grid    : building the PISN mass-function grid + its normalisation
  events  : the (nobs, nsamp) PE weight array -> per-event logsumexp + neff
  sel     : the selection-sample term
  full    : events + sel + neff factors (everything the potential contains
            except the priors)

Variants:
  sort=True   : PE samples sorted within each event row by m1_det, selection
                samples sorted globally by m1_det.  Both reductions are
                permutation-invariant, so this is a free data-layout change.
  R           : intensity_models_fast.SCATTER_REPLICAS

Run:  uv run python bench_breakdown.py [--nobs 9000 --nsamp 4000 --nsel 1700000]
"""
import argparse
import sys
import time

sys.path.append("../src/")

import numpy as np
import jax
import jax.numpy as jnp

import intensity_models_fast as fast
from bench_model import make_synthetic_data, TRUTH, FIDUCIAL


def block(x):
    return jax.tree_util.tree_map(
        lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x
    )


def timeit(fn, *args, n_repeat=7, name=""):
    t0 = time.perf_counter()
    block(fn(*args))
    compile_time = time.perf_counter() - t0
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        block(fn(*args))
        times.append(time.perf_counter() - t0)
    tmin = min(times) * 1000
    print(f"  [{name:44s}] compile: {compile_time:6.1f}s | min: {tmin:8.2f} ms")
    return tmin


FIXED = dict(Om=FIDUCIAL["Om"], w=FIDUCIAL["w"], mpisndot=0.0, zmax=6.5,
             mbh_min=3.0, delta_m=1.6)

P0 = {k: jnp.asarray(v, jnp.float32) for k, v in TRUTH.items() if k != "mpisndot"}


def full_sample(p):
    s = dict(FIXED)
    s.update(
        h=p["h"], a=p["a"], b=p["b"], c=p["c"], mpisn=p["mpisn"],
        mbhmax=p["mpisn"] + p["dmbhmax"], sigma=p["sigma"],
        fpl=jnp.exp(p["log_fpl"]), beta=p["beta"], lam=p["lam"],
        kappa=p["lam"] + p["dkappa"], zp=p["zp"], mp_low=p["mp_low"],
        msigma_low=p["msigma_low"], flow=jnp.exp(p["log_flow"]),
    )
    return s


def make_pieces(data):
    m1s_det = jnp.asarray(data["m1s_det"], jnp.float32)
    log_pdraw = jnp.asarray(data["log_pdraw"], jnp.float32)
    log_m1s_det = jnp.log(m1s_det)
    log_qs = jnp.log(jnp.asarray(data["qs"], jnp.float32))
    log_dls = jnp.log(jnp.asarray(data["dls"], jnp.float32))

    m1s_det_sel = jnp.asarray(data["m1s_det_sel"], jnp.float32)
    log_pdraw_sel = jnp.log(jnp.asarray(data["pdraw_sel"], jnp.float32))
    log_m1s_det_sel = jnp.log(m1s_det_sel)
    log_qs_sel = jnp.log(jnp.asarray(data["qs_sel"], jnp.float32))
    log_dls_sel = jnp.log(jnp.asarray(data["dls_sel"], jnp.float32))
    Ndraw = data["Ndraw"]
    nobs, nsamp = m1s_det.shape

    def build(p):
        s = full_sample(p)
        cosmo = fast.FlatwCDMCosmology(s["h"], s["Om"], s["w"], zmax=s["zmax"])
        log_dN = fast.build_population_model(s, use_low_bump=True)
        return cosmo, log_dN

    def piece_grid(p):
        _, log_dN = build(p)
        ld = log_dN.log_dndm
        return jnp.sum(ld.log_dndm_pisn_grid) + jnp.sum(ld.log_Z_pisn_grid)

    def event_terms(p):
        cosmo, log_dN = build(p)
        log1p_zs, J = cosmo.z_and_log_jacobian(log_dls)
        opz = jnp.exp(log1p_zs)
        zs = opz - 1.0
        m1s = m1s_det / opz
        log_m1s = log_m1s_det - log1p_zs
        log_wts = log_dN.call_from_logs(m1s, log_m1s, log_qs, zs, log1p_zs) - log_pdraw + J
        return fast._logsumexp_and_neff(log_wts, axis=1)

    def piece_events(p):
        lse1, _, neff = event_terms(p)
        return jnp.sum(lse1 - jnp.log(nsamp)) + fast.log_smooth_neff_boundary(
            jnp.min(neff), nobs)

    def sel_terms(p):
        cosmo, log_dN = build(p)
        log1p_zs, J = cosmo.z_and_log_jacobian(log_dls_sel)
        opz = jnp.exp(log1p_zs)
        zs = opz - 1.0
        m1s = m1s_det_sel / opz
        log_m1s = log_m1s_det_sel - log1p_zs
        log_wts = (log_dN.call_from_logs(m1s, log_m1s, log_qs_sel, zs, log1p_zs)
                   - log_pdraw_sel + J)
        return fast._logsumexp_and_neff(log_wts[None, :], axis=1)

    def piece_sel(p):
        lse1, lse2, _ = sel_terms(p)
        log_mu_sel = jnp.squeeze(lse1) - jnp.log(Ndraw)
        log_mu2 = jnp.squeeze(lse2) - 2 * jnp.log(Ndraw)
        x = 2 * log_mu_sel - jnp.log(Ndraw) - log_mu2
        log_s2 = log_mu2 + jnp.log(-jnp.expm1(jnp.minimum(x, -1e-7)))
        neff_sel = jnp.exp(2 * log_mu_sel - log_s2)
        return -nobs * log_mu_sel + fast.log_smooth_neff_boundary(neff_sel, 4 * nobs)

    def piece_full(p):
        return piece_events(p) + piece_sel(p)

    return dict(grid=piece_grid, events=piece_events, sel=piece_sel, full=piece_full)


def make_tab_pieces(data, n_tab=8192):
    """Events term with the whole z-independent mass function folded into one
    fine 1-D table, and log_dndv folded into the dL-lookup table.  Valid only
    when mpisndot == 0 (statically), which is the production configuration.

    Per (event, sample) point this needs 4 gathers (2 shared-index dL-table
    lookups + 2 mass-table lookups) and ~10 flops -- no transcendentals.
    """
    m1s_det = jnp.asarray(data["m1s_det"], jnp.float32)
    log_pdraw = jnp.asarray(data["log_pdraw"], jnp.float32)
    log_m1s_det = jnp.log(m1s_det)
    qs = jnp.asarray(data["qs"], jnp.float32)
    log_qs = jnp.log(qs)
    log1p_qs = jnp.log1p(qs)
    log_dls = jnp.log(jnp.asarray(data["dls"], jnp.float32))
    nobs, nsamp = m1s_det.shape

    m_hi = float(np.max(data["m1s_det"])) * 1.01
    m_axis = fast._LogAxis(1.0, m_hi, n_tab)
    m_grid = m_axis.grid

    def piece_events_tab(p):
        s = full_sample(p)
        cosmo = fast.FlatwCDMCosmology(s["h"], s["Om"], s["w"], zmax=s["zmax"])
        log_dN = fast.build_population_model(s, use_low_bump=True)
        ld = log_dN.log_dndm

        # z-independent mass function on the fine grid; -inf floored so that
        # lerp between two dead nodes cannot form inf - inf = NaN.
        f_tab = jnp.maximum(ld(m_grid, 0.0), -1e30)

        log1p_tab = cosmo._log1p_z_table
        Jg_tab = (cosmo._J_table + 2 * jnp.log(cosmo.dH)
                  + log_dN.log_dndv.from_log1p(log1p_tab))

        t = jnp.clip((log_dls - jnp.log(cosmo.dH) - cosmo._u_lo) * cosmo._inv_du,
                     0.0, cosmo._n_dl - 1.0)
        log1p_zs = fast._lerp1d(log1p_tab, t, cosmo._n_dl)
        Jg = fast._lerp1d(Jg_tab, t, cosmo._n_dl)

        log_m1s = log_m1s_det - log1p_zs
        f1 = fast._lerp1d(f_tab, m_axis.frac_index(log_m1s), n_tab)
        f2 = fast._lerp1d(f_tab, m_axis.frac_index(log_m1s + log_qs), n_tab)

        log_wts = (
            f1 + f2
            + s["beta"] * (log_m1s + log1p_qs
                           - jnp.log(log_dN.mref * (1 + log_dN.qref)))
            + log_m1s + Jg - log_dN.log_norm - log_pdraw
        )
        lse1, _, neff = fast._logsumexp_and_neff(log_wts, axis=1)
        return jnp.sum(lse1 - jnp.log(nsamp)) + fast.log_smooth_neff_boundary(
            jnp.min(neff), nobs)

    return piece_events_tab


def sort_data(data):
    d = dict(data)
    order = np.argsort(d["m1s_det"], axis=1)
    for k in ("m1s_det", "qs", "dls", "log_pdraw"):
        d[k] = np.take_along_axis(d[k], order, axis=1)
    order_sel = np.argsort(d["m1s_det_sel"])
    for k in ("m1s_det_sel", "qs_sel", "dls_sel", "pdraw_sel"):
        d[k] = d[k][order_sel]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nobs", type=int, default=9000)
    ap.add_argument("--nsamp", type=int, default=4000)
    ap.add_argument("--nsel", type=int, default=1_700_000)
    ap.add_argument("--n_repeat", type=int, default=7)
    args = ap.parse_args()

    print(f"jax {jax.__version__} on {jax.devices()}")
    data = make_synthetic_data(args.nobs, args.nsamp, args.nsel)
    pieces = make_pieces(data)

    print("\n=== tabulated events term vs direct (value + grad agreement) ===")
    vg_direct = jax.jit(jax.value_and_grad(pieces["events"]))
    v0, g0 = block(vg_direct(P0))
    for n_tab in (4096, 8192, 16384):
        tab = make_tab_pieces(data, n_tab=n_tab)
        vg_tab = jax.jit(jax.value_and_grad(tab))
        v1, g1 = block(vg_tab(P0))
        dv = abs(float(v1) - float(v0))
        rel_v = dv / abs(float(v0))
        gerr = max(
            abs(float(g1[k]) - float(g0[k])) /
            max(abs(float(g0[k])), 1e-6)
            for k in g0
        )
        print(f"  n_tab={n_tab:6d}: value {float(v0):+.6e} -> {float(v1):+.6e} "
              f"(|dv|={dv:.3e}, rel={rel_v:.2e}), worst grad rel err={gerr:.2e}")
        timeit(vg_tab, P0, n_repeat=args.n_repeat,
               name=f"grad events TABULATED n_tab={n_tab}")

    print("\n=== reference timings ===")
    for name in ("grid", "events", "sel", "full"):
        timeit(jax.jit(jax.value_and_grad(pieces[name])), P0,
               n_repeat=args.n_repeat, name=f"grad {name} (direct)")

    print("\n=== jax.checkpoint on the direct events term ===")
    ck = jax.checkpoint(pieces["events"])
    timeit(jax.jit(jax.value_and_grad(ck)), P0, n_repeat=args.n_repeat,
           name="grad events (direct, remat)")


if __name__ == "__main__":
    main()

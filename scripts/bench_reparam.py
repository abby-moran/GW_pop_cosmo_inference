"""
Geometry benchmark: canonical vs reparameterized sampling coordinates.

Same data, same model (intensity_models_fast), same NUTS settings; the only
difference is the prior file:

  baseline  h, mpisn, sigma, mp_low sampled linearly (gwtc5_fullcosmo_evo
            style), mpisndot free
  reparam   log_h, log_mpisn_ref (mass scale at pivot z*), log_sigma
            (no hard floor), log_mp_low; mpisndot free

The reparameterization targets the two geometry pathologies seen in
endO5_fullcosmo_evo2/3 (notes/model-suggestions.md): the built-in
mpisn--mpisndot correlation (pivot) and the multiplicative spectral-siren
mpisn--h banana (log space, straightened so dense mass can absorb it).

Figure of merit: leapfrog gradients per effective sample of the *canonical*
parameters (h, mpisn, mpisndot, Omh2, sigma), which both runs record --
as sample sites in the baseline, as deterministics in the reparam run.

Usage:
    uv run python bench_reparam.py --nobs 2000 --nsamp 1000 --steps 150
"""
import argparse
import sys
import time

sys.path.append("../src/")

import numpy as np
import jax
import jax.numpy as jnp

from bench_model import make_synthetic_data, build_prior, TRUTH

# Mass-scale evolution in the model is mpisn(z) = mpisn + mpisndot*z/(1+z).
# The pivot should sit at the detection-weighted mean of x = z/(1+z); for
# make_synthetic_data (z = 1.5*power(3)) that is E[x] ~ 0.52, i.e. z* ~ 1.1.
DEFAULT_ZPIVOT = 1.1

COMMON_PRIOR_TEXT = """Omh2 = TruncatedNormal(0.143, 0.05, low=0.02, high=0.4)
w = -1
a = TruncatedNormal(2.35, 2, low=-1.65, high=6.35)
b = TruncatedNormal(1.9, 2, low=-2.1, high=5.9)
c = TruncatedNormal(4, 2, low=0, high=8)
dmbhmax = TruncatedNormal(3.0, 2.0, low=0.5, high=7.0)
beta = Normal(0, 2)
log_fpl = Uniform(np.log(1e-2), np.log(2))
lam = TruncatedNormal(2.7, 2.0, low=-1.3, high=6.7)
dkappa = TruncatedNormal(2.9, 2.0, low=1, high=6.9)
zp = TruncatedNormal(1.9, 1, low=0, high=3.9)
msigma_low = TruncatedNormal(4.0, 2.0, low=0.5, high=8.0)
log_flow = Uniform(np.log(1e-3), np.log(2))
mpisndot = Uniform(low=-2, high=8)
zmax = 6.5
mbh_min = 3.0
delta_m = 1.6
"""

# One line per reparameterizable parameter, per variant ingredient.
LINES = dict(
    h_lin="h = TruncatedNormal(0.7, 0.2, low=0.4, high=1.2)",
    h_log="log_h = TruncatedNormal(np.log(0.7), 0.29, low=np.log(0.4), high=np.log(1.2))",
    mpisn_lin="mpisn = TruncatedNormal(35.0, 5.0, low=20.0, high=50.0)",
    mpisn_pivot="mpisn_ref = TruncatedNormal(35.0, 5.0, low=20.0, high=50.0)",
    # Bounds tightened so mpisn = mpisn_ref - mpisndot*xpivot stays inside the
    # baseline's [20, 50] for every mpisndot in [-2, 8] (xpivot ~ 0.524).
    mpisn_pivot_tight="mpisn_ref = TruncatedNormal(35.0, 5.0, low=24.5, high=48.9)",
    mpisn_logpivot="log_mpisn_ref = TruncatedNormal(np.log(35.0), 0.145, low=np.log(20.0), high=np.log(50.0))",
    mpisn_log="log_mpisn = TruncatedNormal(np.log(35.0), 0.145, low=np.log(20.0), high=np.log(50.0))",
    sigma_lin="sigma = TruncatedNormal(0.1, 0.1, low=0.05)",
    sigma_log_floor="log_sigma = TruncatedNormal(np.log(0.1), 1.0, low=np.log(0.05))",
    sigma_log_free="log_sigma = Normal(np.log(0.1), 1.0)",
    mp_low_lin="mp_low = TruncatedNormal(9.0, 2.0, low=5.0, high=15.0)",
    mp_low_log="log_mp_low = TruncatedNormal(np.log(9.0), 0.22, low=np.log(5.0), high=np.log(15.0))",
)

# variant -> (h, mpisn, sigma, mp_low, needs_zpivot)
VARIANTS = dict(
    # everything, sigma floor removed (first attempt: diverged badly)
    full=("h_log", "mpisn_logpivot", "sigma_log_free", "mp_low_log", True),
    # everything, but keep the hard sigma >= 0.05 support of the baseline
    safe=("h_log", "mpisn_logpivot", "sigma_log_floor", "mp_low_log", True),
    # single ingredients, for isolating effects
    pivot=("h_lin", "mpisn_pivot", "sigma_lin", "mp_low_lin", True),
    pivot_tight=("h_lin", "mpisn_pivot_tight", "sigma_lin", "mp_low_lin", True),
    log=("h_log", "mpisn_log", "sigma_log_floor", "mp_low_log", False),
)


def variant_prior_text(variant, zpivot):
    h_k, m_k, s_k, mp_k, needs_zp = VARIANTS[variant]
    lines = [LINES[h_k], LINES[m_k], LINES[s_k], LINES[mp_k]]
    if needs_zp:
        lines.append(f"zpivot = {zpivot}")
    return COMMON_PRIOR_TEXT + "\n".join(lines) + "\n"

CANONICAL = ["h", "mpisn", "mpisndot", "Omh2", "sigma", "mp_low", "lam", "zp"]


def load_endO5(nobs, pe_path, sel_path):
    """First nobs events of the endO5_val2 mock, loaded exactly the way
    run_inf.py does (half the selection set, ndraw halved to match).  Unlike
    make_synthetic_data these carry real pdraw weights, so the MC-variance
    guard is inert (mc_var ~ 0.8 at 2000 events) and the benchmark probes
    geometry, not the penalty cliff."""
    import h5py
    import pandas as pd

    with h5py.File(pe_path, "r") as f:
        m1s = f["m1"][:nobs]
        qs = f["q"][:nobs]
        dls = f["dl"][:nobs]
        pdraws = f["pdraw"][:nobs]
    pdraws = np.nan_to_num(pdraws, neginf=-1e30, posinf=1e30)

    sel = pd.read_hdf(sel_path, key="true_parameters")
    half = int(np.round(len(sel) / 2))
    sel = sel.iloc[:half]
    ndraw = float(sel["ndraw"].iloc[0]) / 2

    return dict(
        m1s_det=np.asarray(m1s), qs=np.asarray(qs), dls=np.asarray(dls),
        log_pdraw=np.asarray(pdraws),
        m1s_det_sel=np.asarray(sel["m1d"]), qs_sel=np.asarray(sel["q"]),
        dls_sel=np.asarray(sel["dl"]), pdraw_sel=np.asarray(sel["pdraw_sel"]),
        Ndraw=ndraw,
    )


def reparam_truth(variant, zpivot):
    h_k, m_k, s_k, mp_k, _ = VARIANTS[variant]
    xpivot = zpivot / (1.0 + zpivot)
    mpisn_at_pivot = TRUTH["mpisn"] + TRUTH["mpisndot"] * xpivot
    t = {k: v for k, v in TRUTH.items()
         if k not in ("h", "mpisn", "sigma", "mp_low")}
    t.update({
        "h_lin": dict(h=TRUTH["h"]),
        "h_log": dict(log_h=float(np.log(TRUTH["h"]))),
    }[h_k])
    t.update({
        "mpisn_lin": dict(mpisn=TRUTH["mpisn"]),
        "mpisn_pivot": dict(mpisn_ref=float(mpisn_at_pivot)),
        "mpisn_pivot_tight": dict(mpisn_ref=float(mpisn_at_pivot)),
        "mpisn_logpivot": dict(log_mpisn_ref=float(np.log(mpisn_at_pivot))),
        "mpisn_log": dict(log_mpisn=float(np.log(TRUTH["mpisn"]))),
    }[m_k])
    t.update({
        "sigma_lin": dict(sigma=TRUTH["sigma"]),
        "sigma_log_floor": dict(log_sigma=float(np.log(TRUTH["sigma"]))),
        "sigma_log_free": dict(log_sigma=float(np.log(TRUTH["sigma"]))),
    }[s_k])
    t.update({
        "mp_low_lin": dict(mp_low=TRUTH["mp_low"]),
        "mp_low_log": dict(log_mp_low=float(np.log(TRUTH["mp_low"]))),
    }[mp_k])
    return t


def run(label, data, prior, truth, steps, max_tree_depth, dense_mass, seed=1):
    import intensity_models_fast as im
    from numpyro.infer import MCMC, NUTS, init_to_value
    import arviz as az

    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)

    truth = {k: jnp.asarray(v) for k, v in truth.items()
             if k in prior and not isinstance(prior[k], float)}
    kernel = NUTS(im.pop_cosmo_model, init_strategy=init_to_value(values=truth),
                  max_tree_depth=max_tree_depth, dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_warmup=steps, num_samples=steps, num_chains=1,
                progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), *model_args, use_low_bump=True,
             extra_fields=("num_steps", "accept_prob", "diverging",
                           "adapt_state.step_size"))
    wall = time.perf_counter() - t0

    ex = mcmc.get_extra_fields(group_by_chain=False)
    ns = np.asarray(ex["num_steps"])
    ap = np.asarray(ex["accept_prob"])
    dv = np.asarray(ex["diverging"])
    ss = np.asarray(ex["adapt_state.step_size"])
    samples = mcmc.get_samples()

    print(f"\n  --- {label} ---")
    print(f"    wall                    {wall:8.1f} s for {2*steps} iterations")
    print(f"    leapfrog steps/sample   mean {ns.mean():7.1f}  "
          f"median {np.median(ns):7.1f}  max {ns.max():5d}")
    print(f"    accept prob             mean {np.nanmean(ap):7.3f}")
    print(f"    divergences             {int(dv.sum())} / {steps}")
    print(f"    final step size         {ss[-1]:.4e}")

    # Where does the chain actually go, and where does it diverge?  The
    # mc_var_loglike deterministic is the Monte-Carlo noise guard; if
    # divergences cluster at low mpisn / high mc_var, the problem is the
    # enlarged support, not the coordinate change itself.
    dvb = dv.astype(bool)
    for k in ("mpisn", "mc_var_loglike", "min_neff", "sigma"):
        if k not in samples:
            continue
        x = np.asarray(samples[k], dtype=np.float64)
        line = f"    {k:14s} range [{x.min():9.4f}, {x.max():9.4f}]"
        if dvb.any():
            line += f"   at divergences [{x[dvb].min():9.4f}, {x[dvb].max():9.4f}]"
        print(line)

    total_grads = int(ns.sum())
    ess = {}
    for k in CANONICAL:
        if k not in samples:
            continue
        x = np.asarray(samples[k], dtype=np.float64)[None, :]
        ess[k] = float(az.ess(x))
    print(f"    total leapfrog grads    {total_grads}")
    print(f"    ESS (of {steps} draws) and grads/ESS:")
    for k, e in ess.items():
        print(f"      {k:10s} ESS {e:7.1f}   grads/eff.sample {total_grads/max(e,1e-9):10.0f}")
    min_ess = min(ess.values()) if ess else float("nan")
    print(f"    worst-case grads/eff.sample: {total_grads/max(min_ess,1e-9):.0f}")
    return dict(label=label, wall=wall, mean_steps=float(ns.mean()),
                total_grads=total_grads, ess=ess, min_ess=min_ess,
                divergences=int(dv.sum()), step_size=float(ss[-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nobs", type=int, default=2000)
    p.add_argument("--nsamp", type=int, default=1000)
    p.add_argument("--nsel", type=int, default=200000)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--max_tree_depth", type=int, default=10)
    p.add_argument("--no_dense_mass", action="store_true")
    p.add_argument("--zpivot", type=float, default=DEFAULT_ZPIVOT)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--variants", default="safe",
                   help=f"comma list from {sorted(VARIANTS)}; 'baseline' also allowed")
    p.add_argument("--real", action="store_true",
                   help="use the endO5_val2 mock (real pdraw weights) instead "
                        "of synthetic data; --nobs selects the event count. "
                        "Use --zpivot 0.75 (detection-weighted mean of "
                        "z/(1+z) for this catalog).")
    p.add_argument("--real_pe", default="/mnt/home/misi/src/GW_pop_cosmo_inference/runs/endO5_val2/PE_noevo_vlowres.h5")
    p.add_argument("--real_sel", default="/mnt/home/misi/src/GW_pop_cosmo_inference/runs/endO5_val2/sel_noevo.h5")
    args = p.parse_args()
    dense_mass = not args.no_dense_mass

    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"nobs={args.nobs} nsamp={args.nsamp} nsel={args.nsel} "
          f"steps={args.steps} depth={args.max_tree_depth} "
          f"dense_mass={dense_mass} zpivot={args.zpivot}")

    if args.real:
        data = load_endO5(args.nobs, args.real_pe, args.real_sel)
        print(f"endO5_val2 mock: PE {data['m1s_det'].shape}, "
              f"sel {data['m1s_det_sel'].shape[0]:,}, Ndraw {data['Ndraw']:.3e}")
    else:
        data = make_synthetic_data(args.nobs, args.nsamp, args.nsel, seed=11)

    from utils import get_priors_from_file

    out = []
    for variant in args.variants.split(","):
        variant = variant.strip()
        if variant == "baseline":
            prior = build_prior(False, "/tmp/bench_reparam_base.prior")
            truth = dict(TRUTH)
            label = "baseline (linear h/mpisn/sigma/mp_low)"
        else:
            path = f"/tmp/bench_reparam_{variant}.prior"
            with open(path, "w") as f:
                f.write(variant_prior_text(variant, args.zpivot))
            prior = get_priors_from_file(path)
            truth = reparam_truth(variant, args.zpivot)
            label = f"variant '{variant}'"
        out.append(run(label, data, prior, truth, args.steps,
                       args.max_tree_depth, dense_mass, seed=args.seed))

    if len(out) > 1:
        print("\n  === summary ===")
        for r in out:
            print(f"    {r['label']:44s} div {r['divergences']:3d}  "
                  f"grads {r['total_grads']:7d}  minESS {r['min_ess']:6.1f}  "
                  f"worst grads/ESS {r['total_grads']/max(r['min_ess'],1e-9):8.0f}")


if __name__ == "__main__":
    main()

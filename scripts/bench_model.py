"""
Benchmark the *gradient* of pop_cosmo_model's potential energy, which is what
NUTS actually pays for (up to 2**max_tree_depth of them per sample).

profile_mod.py times forward passes of sub-expressions; that undercounts by
~3x and misses the per-step PISN grid rebuild that dominates at small nsamp.

Usage:
    uv run python bench_model.py                      # synthetic, 9000 x 4000
    uv run python bench_model.py --nobs 2000 --nsamp 1000
    uv run python bench_model.py --config run_configs/mock_O5_noevo.ini   # real data
    uv run python bench_model.py --module intensity_models_fast           # compare impls
"""
import argparse
import json
import os
import sys
import time

sys.path.append("../src/")

import numpy as np
import jax
import jax.numpy as jnp
import numpyro


def block(x):
    return jax.tree_util.tree_map(
        lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x
    )


def timeit(fn, *args, n_repeat=7, name="", **kwargs):
    t0 = time.perf_counter()
    block(fn(*args, **kwargs))
    compile_time = time.perf_counter() - t0
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        block(fn(*args, **kwargs))
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    print(
        f"  [{name:52s}] compile+first: {compile_time:8.3f}s | "
        f"min: {times.min()*1000:9.3f}ms | median: {np.median(times)*1000:9.3f}ms"
    )
    return dict(
        name=name,
        compile_and_first_run=compile_time,
        min=float(times.min()),
        median=float(np.median(times)),
    )


# --------------------------------------------------------------------------
# Synthetic data drawn so that masses/redshifts land inside the model's grids
# (uniform junk would put most samples in the -inf tails and hide real cost).
# --------------------------------------------------------------------------
FIDUCIAL = dict(h=0.674, Om=0.315, w=-1.0)


def _dl_of_z(z, h=0.674, Om=0.315, w=-1.0, zmax=20.0, n=4096):
    zi = np.expm1(np.linspace(np.log(1), np.log(1 + zmax), n))
    E = np.sqrt(Om * (1 + zi) ** 3 + (1 - Om) * (1 + zi) ** (3 * (1 + w)))
    dH = 2.99792 / h
    dc = dH * np.concatenate(([0.0], np.cumsum(0.5 * np.diff(zi) * (1 / E[:-1] + 1 / E[1:]))))
    return np.interp(z, zi, dc * (1 + zi))


def make_synthetic_data(nobs, nsamp, nsel, zmax_data=1.5, seed=0):
    rng = np.random.default_rng(seed)

    # Per-event source-frame truths, then a narrow posterior blob around each.
    # Kept well inside the model's support (m2 > mbh_min, z < zmax) so that no
    # event is entirely dead at the truth point -- otherwise the *original*
    # model cannot even be initialized, which is a separate finding.
    m1_true = 12.0 + 55.0 * rng.beta(1.5, 3.0, size=(nobs, 1))
    q_true = rng.uniform(0.4, 1.0, size=(nobs, 1))
    z_true = zmax_data * rng.power(3.0, size=(nobs, 1))  # more events at high z

    m1_src = m1_true * np.exp(0.15 * rng.standard_normal((nobs, nsamp)))
    qs = np.clip(q_true + 0.05 * rng.standard_normal((nobs, nsamp)), 0.35, 1.0)
    zs = np.clip(z_true * np.exp(0.15 * rng.standard_normal((nobs, nsamp))), 1e-3, zmax_data)

    m1s_det = m1_src * (1 + zs)
    dls = _dl_of_z(zs, **FIDUCIAL)
    # run_inf.py passes the PE 'pdraw' column straight through to the model's
    # `log_pdraw` argument, so this array is in log space.
    log_pdraw = np.log(1e-6) + rng.standard_normal((nobs, nsamp))

    # Selection samples: 1-D, broader (they sample the full drawn population).
    m1_src_sel = 3.0 + 90.0 * rng.beta(1.2, 2.5, size=nsel)
    qs_sel = rng.uniform(0.05, 1.0, size=nsel)
    zs_sel = zmax_data * rng.power(2.0, size=nsel)
    m1s_det_sel = m1_src_sel * (1 + zs_sel)
    dls_sel = _dl_of_z(zs_sel, **FIDUCIAL)
    pdraw_sel = np.exp(np.log(1e-6) + rng.standard_normal(nsel))
    Ndraw = float(nsel) * 20.0

    return dict(
        m1s_det=np.asarray(m1s_det, np.float64),
        qs=np.asarray(qs, np.float64),
        dls=np.asarray(dls, np.float64),
        log_pdraw=np.asarray(log_pdraw, np.float64),
        m1s_det_sel=np.asarray(m1s_det_sel, np.float64),
        qs_sel=np.asarray(qs_sel, np.float64),
        dls_sel=np.asarray(dls_sel, np.float64),
        pdraw_sel=np.asarray(pdraw_sel, np.float64),
        Ndraw=Ndraw,
    )


def load_real_data(cfg_path, base_runs_dir="../runs"):
    import configparser
    import pandas as pd
    import h5py

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    run = cfg["run"]
    run_dir = os.path.join(base_runs_dir, run["run_dir"])
    pe_file = os.path.join(run_dir, run["output_file_pe"] if "output_file_pe" in run else run["output_file_PE"])
    sel_file = os.path.join(run_dir, run["output_sel_file"])
    evt_start = run.getint("evt_start")
    evt_end = run.get("evt_end")
    evt_end = None if (evt_end is None or evt_end.lower() == "none") else int(evt_end)

    with h5py.File(pe_file, "r") as f:
        m1s = f["m1"][evt_start:evt_end]
        qs = f["q"][evt_start:evt_end]
        dls = f["dl"][evt_start:evt_end]
        pdraws = f["pdraw"][evt_start:evt_end]
    pdraws = np.nan_to_num(pdraws, neginf=-1e30, posinf=1e30)

    sel = pd.read_hdf(sel_file, key="true_parameters")
    half = int(np.round(len(sel) / 2))
    sel = sel.iloc[:half]
    Ndraw = float(sel["ndraw"].iloc[0]) / 2

    return dict(
        m1s_det=np.asarray(m1s), qs=np.asarray(qs), dls=np.asarray(dls),
        log_pdraw=np.asarray(pdraws),
        m1s_det_sel=np.asarray(sel["m1d"]), qs_sel=np.asarray(sel["q"]),
        dls_sel=np.asarray(sel["dl"]), pdraw_sel=np.asarray(sel["pdraw_sel"]),
        Ndraw=Ndraw,
    ), cfg


# --------------------------------------------------------------------------
# Prior: mirrors the gwtc5-style cosmo prior (h sampled; Om, w fixed) with the
# low-mass bump parameters present.  Kept inline so the benchmark runs without
# the (gitignored) runs/priors directory.  OM_LINE / W_LINE are placeholders
# so --cosmo_free / --omh2 can switch the cosmology parameterization.
# --------------------------------------------------------------------------
DEFAULT_PRIOR_TEXT = """h = TruncatedNormal(0.7, 0.2, low=0.4, high=1.2)
OM_LINE
W_LINE
a = TruncatedNormal(2.35, 2, low=-1.65, high=6.35)
b = TruncatedNormal(1.9, 2, low=-2.1, high=5.9)
c = TruncatedNormal(4, 2, low=0, high=8)
mpisn = TruncatedNormal(35.0, 5.0, low=20.0, high=50.0)
dmbhmax = TruncatedNormal(3.0, 2.0, low=0.5, high=7.0)
sigma = TruncatedNormal(0.1, 0.1, low=0.05)
beta = Normal(0, 2)
log_fpl = Uniform(np.log(1e-2), np.log(2))
lam = TruncatedNormal(2.7, 2.0, low=-1.3, high=6.7)
dkappa = TruncatedNormal(2.9, 2.0, low=1, high=6.9)
zp = TruncatedNormal(1.9, 1, low=0, high=3.9)
mp_low = TruncatedNormal(9.0, 2.0, low=5.0, high=15.0)
msigma_low = TruncatedNormal(4.0, 2.0, low=0.5, high=8.0)
log_flow = Uniform(np.log(1e-3), np.log(2))
mpisndot = MPISNDOT
zmax = 6.5
mbh_min = 3.0
delta_m = 1.6
"""

# Truth values from scripts/pop_configs/mock_O5_noevo.txt, already mapped into
# the derived parameterisation the model samples in.  Om/w/Omh2 are only used
# when the corresponding prior line makes them sampled (--cosmo_free / --omh2);
# the `k in prior and not fixed` filter below drops them otherwise.
TRUTH = dict(
    h=0.674, a=-0.9426, b=0.237, c=2.360, mpisn=33.29, dmbhmax=36.7345 - 33.29,
    sigma=0.0539, log_fpl=float(np.log(0.63909)), lam=4.814, dkappa=8.3659 - 4.814,
    zp=0.954, beta=-2.43, msigma_low=4.0, mp_low=9.121,
    log_flow=float(np.log(0.6025)), mpisndot=0.0,
    Om=0.315, w=-1.0, Omh2=0.315 * 0.674 ** 2,
)


def build_prior(mpisndot_fixed, path, cosmo_free=False, omh2=False):
    """cosmo_free: sample Om and w instead of fixing them.
    omh2: sample the physical density Omh2 = Om*h^2 instead of Om (the model
    derives Om = Omh2/h^2; see get_deterministic_parameters).  Implies a
    sampled w as well when combined with cosmo_free; can also be used alone."""
    if omh2:
        om_line = "Omh2 = TruncatedNormal(0.143, 0.05, low=0.02, high=0.4)"
    elif cosmo_free:
        om_line = "Om = TruncatedNormal(0.315, 0.08, low=0.05, high=0.7)"
    else:
        om_line = "Om = 0.315"
    if cosmo_free:
        w_line = "w = TruncatedNormal(-1.0, 0.3, low=-2.0, high=-0.3)"
    else:
        w_line = "w = -1"
    txt = (DEFAULT_PRIOR_TEXT
           .replace("MPISNDOT", "0" if mpisndot_fixed else "Uniform(low=-2, high=8)")
           .replace("OM_LINE", om_line)
           .replace("W_LINE", w_line))
    with open(path, "w") as f:
        f.write(txt)
    from utils import get_priors_from_file

    return get_priors_from_file(path)


def diagnose(model, model_args, model_kwargs, truth):
    """Trace the model once at `truth` and report every site, flagging non-finite
    values.  This is what tells you *which* factor is killing initialization."""
    import numpyro.handlers as handlers

    print("\n=== model trace at truth point ===")
    with handlers.seed(rng_seed=0):
        with handlers.substitute(data=truth):
            tr = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    bad = []
    for name, site in list(tr.items()):
        if site["type"] not in ("sample", "deterministic", "factor"):
            continue
        # numpyro.factor sites are sample sites with a Unit distribution whose
        # value has shape (0,); the number we care about is log_factor.
        v = site["value"]
        fn = site.get("fn")
        if fn is not None and type(fn).__name__ == "Unit":
            v = fn.log_factor
        if v is None or (hasattr(v, "size") and v.size == 0):
            print(f"  {site['type']:14s} {name:24s} = <empty>")
            continue
        if isinstance(v, jax.core.Tracer):
            print(f"  {site['type']:14s} {name:24s} = LEAKED TRACER "
                  f"(site recorded inside a jax.jit -- will vanish on cache hit)")
            bad.append(name)
            continue
        v = np.asarray(v)
        finite = np.all(np.isfinite(v))
        tag = "" if finite else "   <-- NON-FINITE"
        if v.size == 1:
            print(f"  {site['type']:14s} {name:24s} = {float(v):+.6e}{tag}")
        else:
            print(f"  {site['type']:14s} {name:24s} shape {str(v.shape):12s} "
                  f"min {np.nanmin(v):+.4e} max {np.nanmax(v):+.4e} "
                  f"n_nonfinite {int((~np.isfinite(v)).sum())}{tag}")
        if not finite:
            bad.append(name)
    if bad:
        print(f"\n  NON-FINITE SITES: {bad}")
    return tr, bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="use real data from this .ini")
    p.add_argument("--module", default="intensity_models",
                   help="module providing pop_cosmo_model (to A/B implementations)")
    p.add_argument("--nobs", type=int, default=9000)
    p.add_argument("--nsamp", type=int, default=4000)
    p.add_argument("--nsel", type=int, default=1_700_000)
    p.add_argument("--n_repeat", type=int, default=7)
    p.add_argument("--mpisndot_free", action="store_true",
                   help="sample mpisndot (default: fixed to 0)")
    p.add_argument("--cosmo_free", action="store_true",
                   help="sample Om and w (default: fixed to 0.315, -1)")
    p.add_argument("--omh2", action="store_true",
                   help="sample Omh2 = Om*h^2 instead of Om (fast module only)")
    p.add_argument("--no_low_bump", action="store_true")
    p.add_argument("--mcmc", type=int, default=0,
                   help="if >0, also run this many warmup+sample steps of real NUTS")
    p.add_argument("--out", default=None)
    p.add_argument("--diagnose", action="store_true",
                   help="trace the model at the truth point and report every site")
    args = p.parse_args()

    import importlib

    im = importlib.import_module(args.module)
    print(f"model module: {im.__name__} ({im.__file__})")
    print(f"jax {jax.__version__} on {jax.devices()}  x64={jax.config.jax_enable_x64}")

    if args.config:
        data, cfg = load_real_data(args.config)
        print("loaded real data from", args.config)
    else:
        data = make_synthetic_data(args.nobs, args.nsamp, args.nsel)

    npe = data["m1s_det"].size
    print(f"PE array: {data['m1s_det'].shape} = {npe:,} points | "
          f"sel: {data['m1s_det_sel'].shape[0]:,}")

    scratch = os.environ.get("SCRATCH_PRIOR", "/tmp/bench_prior.prior")
    prior = build_prior(not args.mpisndot_free, scratch,
                        cosmo_free=args.cosmo_free, omh2=args.omh2)
    print("sampled params:", sorted(k for k, v in prior.items() if not isinstance(v, float)))

    use_low_bump = not args.no_low_bump
    model_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
        data["Ndraw"], prior,
    )
    model_kwargs = dict(use_low_bump=use_low_bump)

    truth = {k: jnp.asarray(v) for k, v in TRUTH.items() if k in prior
             and not isinstance(prior[k], float)}

    if args.diagnose:
        diagnose(im.pop_cosmo_model, model_args, model_kwargs, truth)

    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    t0 = time.perf_counter()
    mi = initialize_model(
        jax.random.PRNGKey(0), im.pop_cosmo_model,
        model_args=model_args, model_kwargs=model_kwargs, dynamic_args=False,
        init_strategy=init_to_value(values=truth),
    )
    print(f"initialize_model: {time.perf_counter()-t0:.2f}s")
    z0 = mi.param_info.z
    pe_fn = mi.potential_fn

    results = {}
    print("\n=== potential energy and its gradient (what NUTS pays per leapfrog step) ===")
    results["potential"] = timeit(jax.jit(pe_fn), z0, n_repeat=args.n_repeat,
                                 name="potential_fn(z)  [forward]")
    gfn = jax.jit(jax.grad(pe_fn))
    results["grad"] = timeit(gfn, z0, n_repeat=args.n_repeat,
                             name="grad(potential_fn)(z)  [fwd+bwd]")
    vg = jax.jit(jax.value_and_grad(pe_fn))
    results["value_and_grad"] = timeit(vg, z0, n_repeat=args.n_repeat,
                                       name="value_and_grad(potential_fn)(z)")

    g = results["grad"]["min"]
    print(f"\n  potential value at init: {float(jax.jit(pe_fn)(z0)):.6e}")
    print(f"  grad cost: {g*1000:.2f} ms/leapfrog step")
    for d in (5, 6, 7):
        per_sample = g * (2 ** d)
        print(f"    max_tree_depth={d}: worst case {2**d:4d} steps -> "
              f"{per_sample:7.2f}s/sample -> {per_sample*3600/3600:7.2f}h for 1800+1800")

    print("\n  peak GPU bytes in use: ", end="")
    try:
        st = jax.devices()[0].memory_stats()
        print(f"{st['peak_bytes_in_use']/2**30:.2f} GiB / {st['bytes_limit']/2**30:.2f} GiB")
    except Exception as e:
        print("unavailable", e)

    if args.mcmc:
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer import init_to_sample

        print(f"\n=== real NUTS: {args.mcmc} warmup + {args.mcmc} samples, 1 chain ===")
        kernel = NUTS(im.pop_cosmo_model, init_strategy=init_to_sample, max_tree_depth=7)
        mcmc = MCMC(kernel, num_warmup=args.mcmc, num_samples=args.mcmc,
                    num_chains=1, progress_bar=True)
        t0 = time.perf_counter()
        # num_steps must be requested: NUTS collects only ('z', 'diverging') by
        # default, so the `if "num_steps" in ex` check below was dead code and the
        # gradient-time x steps/sample x samples ~ wall-clock sanity check
        # documented in notes/2026-08-07-profiling-jax-numpyro-guide.md silently
        # never ran.
        mcmc.run(jax.random.PRNGKey(1), *model_args, **model_kwargs,
                 extra_fields=("num_steps", "accept_prob", "diverging"))
        dt = time.perf_counter() - t0
        results["mcmc"] = dict(steps=2 * args.mcmc, wall=dt, per_sample=dt / (2 * args.mcmc))
        print(f"  wall {dt:.1f}s -> {dt/(2*args.mcmc):.3f}s per sample")
        ex = mcmc.get_extra_fields(group_by_chain=False)
        if "num_steps" in ex:
            ns = np.asarray(ex["num_steps"])
            print(f"  leapfrog steps/sample: mean {ns.mean():.1f} max {ns.max()}")
            print(f"  accept_prob: mean {np.asarray(ex['accept_prob']).mean():.3f} | "
                  f"divergences: {int(np.asarray(ex['diverging']).sum())}")
            results["mcmc"]["mean_num_steps"] = float(ns.mean())
            # The check the profiling guide asks for, now that num_steps exists.
            # NB: `dt` includes the one-off trace+compile (~20s at production
            # scale), so the ratio only approaches 1 once the sample count is
            # large enough to amortize it -- don't read a short run's ratio as a
            # discrepancy.
            pred = results["grad"]["min"] * ns.sum()
            print(f"  predicted sampling time from gradient cost: {pred:.1f}s "
                  f"vs measured wall {dt:.1f}s (ratio {dt/max(pred,1e-9):.2f}; "
                  f"wall includes compile, so expect >1 for short runs)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

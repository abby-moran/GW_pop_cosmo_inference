"""
End-to-end NUTS comparison: intensity_models vs intensity_models_fast.

Per-leapfrog cost is only half the story.  The other half is how many leapfrog
steps NUTS takes per sample, which collapses if the gradient is ever NaN (NUTS
treats a NaN gradient as a divergence, warmup shrinks the step size, and every
subsequent sample runs the full 2**max_tree_depth trajectory).

Reports, for each module: wall time, mean/max leapfrog steps per sample,
accept probability, divergences, final step size.

Usage:
    uv run python bench_sampling.py --nobs 2000 --nsamp 1000 --steps 100
"""
import argparse
import os
import sys
import time

sys.path.append("../src/")

import numpy as np
import jax
import jax.numpy as jnp
import numpyro

from bench_model import make_synthetic_data, build_prior, TRUTH


def run(module_name, data, prior, truth, steps, max_tree_depth, dense_mass,
        model_kwargs, seed=1):
    import importlib

    im = importlib.import_module(module_name)
    from numpyro.infer import MCMC, NUTS, init_to_value

    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)

    kernel = NUTS(im.pop_cosmo_model, init_strategy=init_to_value(values=truth),
                  max_tree_depth=max_tree_depth, dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_warmup=steps, num_samples=steps, num_chains=1,
                progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), *model_args, **model_kwargs,
             extra_fields=("num_steps", "accept_prob", "diverging", "adapt_state.step_size"))
    wall = time.perf_counter() - t0

    ex = mcmc.get_extra_fields(group_by_chain=False)
    ns = np.asarray(ex["num_steps"])
    ap = np.asarray(ex["accept_prob"])
    dv = np.asarray(ex["diverging"])
    ss = np.asarray(ex["adapt_state.step_size"])
    keys = sorted(mcmc.get_samples().keys())

    print(f"\n  --- {module_name}  (dense_mass={dense_mass}, "
          f"max_tree_depth={max_tree_depth}) ---")
    print(f"    wall                    {wall:8.1f} s for {2*steps} iterations "
          f"({wall/(2*steps)*1000:.0f} ms/iter)")
    print(f"    leapfrog steps/sample   mean {ns.mean():7.1f}  median {np.median(ns):7.1f}"
          f"  max {ns.max():5d}")
    print(f"    accept prob             mean {np.nanmean(ap):7.3f}")
    print(f"    divergences             {int(dv.sum())} / {steps}")
    print(f"    final step size         {ss[-1]:.4e}")
    print(f"    ms per leapfrog step    {wall/max(ns.sum(),1)*1000:8.2f}")
    print(f"    derived params present  "
          f"{sorted(set(keys) & {'kappa','mbhmax','fpl','flow','R'})}")
    return dict(module=module_name, wall=wall, mean_steps=float(ns.mean()),
                max_steps=int(ns.max()), divergences=int(dv.sum()),
                step_size=float(ss[-1]), ms_per_leapfrog=wall / max(ns.sum(), 1) * 1000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nobs", type=int, default=2000)
    p.add_argument("--nsamp", type=int, default=1000)
    p.add_argument("--nsel", type=int, default=200000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max_tree_depth", type=int, default=7)
    p.add_argument("--modules", default="intensity_models,intensity_models_fast")
    p.add_argument("--dense_mass", action="store_true")
    p.add_argument("--mpisndot_free", action="store_true")
    p.add_argument("--dead_events", type=int, default=0,
                   help="force this many events entirely outside the model support")
    args = p.parse_args()

    print(f"jax {jax.__version__} on {jax.devices()}")
    data = make_synthetic_data(args.nobs, args.nsamp, args.nsel, seed=11)
    if args.dead_events:
        data["m1s_det"][: args.dead_events, :] = 0.5
        print(f"forced {args.dead_events} events out of support")

    prior = build_prior(not args.mpisndot_free,
                        os.environ.get("SCRATCH_PRIOR", "/tmp/bench_prior.prior"))
    truth = {k: jnp.asarray(v) for k, v in TRUTH.items()
             if k in prior and not isinstance(prior[k], float)}

    print(f"PE {data['m1s_det'].shape} = {data['m1s_det'].size:,} points, "
          f"sel {data['m1s_det_sel'].shape[0]:,}")

    out = []
    for mod in args.modules.split(","):
        try:
            out.append(run(mod, data, prior, truth, args.steps,
                           args.max_tree_depth, args.dense_mass, dict(use_low_bump=True)))
        except Exception as e:
            print(f"\n  --- {mod} ---\n    FAILED: {type(e).__name__}: {e}")

    if len(out) == 2:
        a, b = out
        print(f"\n  speedup {a['module']} -> {b['module']}: "
              f"{a['wall']/b['wall']:.2f}x wall, "
              f"{a['ms_per_leapfrog']/b['ms_per_leapfrog']:.2f}x per leapfrog step, "
              f"{a['mean_steps']/max(b['mean_steps'],1e-9):.2f}x fewer steps/sample")


if __name__ == "__main__":
    main()

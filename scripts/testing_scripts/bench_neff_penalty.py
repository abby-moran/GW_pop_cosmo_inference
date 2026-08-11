"""
A/B/C benchmark of the Monte-Carlo-accuracy guard in pop_cosmo_model:

  min_neff      -- original: penalize min-over-events n_eff below neff_criterion
  mc_variance   -- penalize sum_i 1/n_eff_i above mc_variance_budget
  none          -- no factor in the potential (diagnostics still recorded)

Times value_and_grad of the potential (the per-leapfrog cost) at production
size.  Reuses the synthetic data / prior machinery from bench_model.py.

Usage: uv run python bench_neff_penalty.py [--nobs 9000 --nsamp 4000 ...]
"""
import argparse
import sys
import time

sys.path.append("../src/")

import numpy as np
import jax
import jax.numpy as jnp

from bench_model import make_synthetic_data, build_prior, timeit, TRUTH

import intensity_models_fast as im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nobs", type=int, default=9000)
    p.add_argument("--nsamp", type=int, default=4000)
    p.add_argument("--nsel", type=int, default=1_700_000)
    p.add_argument("--n_repeat", type=int, default=20)
    args = p.parse_args()

    print(f"jax {jax.__version__} on {jax.devices()}  x64={jax.config.jax_enable_x64}")
    data = make_synthetic_data(args.nobs, args.nsamp, args.nsel)
    prior = build_prior(True, "/tmp/bench_neff_prior.prior")

    model_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
        data["Ndraw"], prior,
    )
    truth = {k: jnp.asarray(v) for k, v in TRUTH.items() if k in prior
             and not isinstance(prior[k], float)}

    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    variants = (
        ("min_neff (original guard)", dict(neff_penalty="min_neff")),
        ("mc_variance (default)", dict(neff_penalty="mc_variance", mc_variance_budget=5.0)),
        ("none (record only)", dict(neff_penalty="none")),
    )

    results = {}
    for label, kw in variants:
        mi = initialize_model(
            jax.random.PRNGKey(0), im.pop_cosmo_model,
            model_args=model_args, model_kwargs=dict(use_low_bump=True, **kw),
            dynamic_args=False, init_strategy=init_to_value(values=truth),
        )
        z0 = mi.param_info.z
        vg = jax.jit(jax.value_and_grad(mi.potential_fn))
        r = timeit(vg, z0, n_repeat=args.n_repeat, name=label)
        v, g = vg(z0)
        gnorm = float(jnp.linalg.norm(jnp.concatenate([jnp.ravel(x) for x in
                                                       jax.tree_util.tree_leaves(g)])))
        print(f"      potential={float(v):+.6e}  |grad|={gnorm:.6e}")
        results[label] = r

    base = results[variants[0][0]]["min"]
    print("\nrelative to min_neff:")
    for label, _ in variants:
        m = results[label]["min"]
        print(f"  {label:32s} {m*1000:8.3f} ms  ({100*(m-base)/base:+.1f}%)")

    # Sanity check the mc_variance factor itself at the truth point.
    import numpyro.handlers as handlers
    with handlers.seed(rng_seed=0), handlers.substitute(data=truth):
        tr = handlers.trace(im.pop_cosmo_model).get_trace(
            *model_args, use_low_bump=True, neff_penalty="mc_variance",
            mc_variance_budget=5.0)
    mc_var = float(tr["mc_var_loglike"]["value"])
    min_neff = float(tr["min_neff"]["value"])
    fac = float(tr["neff_criteria"]["fn"].log_factor)
    print(f"\nat truth: min_neff={min_neff:.2f}  sum(1/n_eff)={mc_var:.3f} "
          f"(MC sigma of logL = {np.sqrt(mc_var):.3f} nats)  factor={fac:+.4f}")


if __name__ == "__main__":
    main()

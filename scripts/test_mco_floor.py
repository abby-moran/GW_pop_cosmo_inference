"""
Check that mco_floor (the CO-IMF flattening scale, see log_dNdmCO) is piped
from the prior file, defaults to 6.0, and actually takes effect.

This guards against the failure mode that mco_min had before: a value set in
the prior file that build_population_model silently dropped, so the run used
the hardcoded default while appearing to honour the config.

Usage: uv run python test_mco_floor.py
"""
import os
import sys

sys.path.append("../src/")

import numpy as np
import jax
import jax.numpy as jnp

from bench_model import make_synthetic_data, DEFAULT_PRIOR_TEXT, TRUTH

import intensity_models_fast as fast
from utils import get_priors_from_file


def prior_with(extra_lines, path):
    txt = DEFAULT_PRIOR_TEXT.replace("MPISNDOT", "0")
    if extra_lines:
        txt = txt + extra_lines
    with open(path, "w") as f:
        f.write(txt)
    return get_priors_from_file(path)


def potential_at_truth(prior, data):
    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
                  data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
                  data["pdraw_sel"], data["Ndraw"], prior)
    truth = {k: jnp.asarray(v) for k, v in TRUTH.items()
             if k in prior and not isinstance(prior[k], float)}
    mi = initialize_model(
        jax.random.PRNGKey(0), fast.pop_cosmo_model,
        model_args=model_args, model_kwargs=dict(use_low_bump=True),
        dynamic_args=False, init_strategy=init_to_value(values=truth),
    )
    v, g = jax.jit(jax.value_and_grad(mi.potential_fn))(mi.param_info.z)
    gn = float(jnp.linalg.norm(jnp.concatenate(
        [jnp.ravel(x) for x in jax.tree_util.tree_leaves(g)])))
    return float(v), gn


def main():
    failures = []

    def check(ok, msg):
        print(f"  [{'OK ' if ok else 'FAIL'}] {msg}")
        if not ok:
            failures.append(msg)

    # ---- 1. the standalone function honours its argument -------------------
    print("\n=== 1. log_dNdmCO / _from_log respect mco_floor ===")
    mco = jnp.asarray([4.0, 5.0, 6.0, 8.0, 30.0])
    a, b = 2.35, 1.9
    for floor in (6.0, 10.0):
        direct = fast.log_dNdmCO(mco, a, b, mco_floor=floor)
        fromlog = fast.log_dNdmCO_from_log(jnp.log(mco), a, b, mco_floor=floor)
        check(np.allclose(direct, fromlog, atol=1e-5),
              f"floor={floor}: log_dNdmCO == log_dNdmCO_from_log "
              f"(max|d|={float(jnp.max(jnp.abs(direct - fromlog))):.2e})")
        flat = direct[mco < floor]
        check(bool(np.allclose(flat, direct[0], atol=1e-6)),
              f"floor={floor}: density is flat below the floor ({flat.size} nodes)")

    # a jnp scalar (what a prior-file float becomes via numpyro.deterministic)
    # must work, not raise on float()
    tr = fast.log_dNdmCO_from_log(jnp.log(mco), a, b, mco_floor=jnp.asarray(6.0))
    check(np.allclose(tr, fast.log_dNdmCO_from_log(jnp.log(mco), a, b, mco_floor=6.0),
                      atol=1e-6),
          "mco_floor as a jnp scalar matches the Python float path")

    # and under jit, where it is a tracer
    jitted = jax.jit(lambda f: fast.log_dNdmCO_from_log(jnp.log(mco), a, b, mco_floor=f))
    check(np.allclose(jitted(jnp.asarray(6.0)), tr, atol=1e-6),
          "mco_floor as a tracer (jit) matches")

    # ---- 2. the full model: default, explicit-default, and changed --------
    print("\n=== 2. pop_cosmo_model picks mco_floor up from the prior file ===")
    data = make_synthetic_data(300, 200, 20000, seed=5)
    scratch = "/tmp/test_mco_floor"
    os.makedirs(scratch, exist_ok=True)

    v_def = potential_at_truth(prior_with(None, f"{scratch}/a.prior"), data)
    v_6 = potential_at_truth(prior_with("mco_floor = 6\n", f"{scratch}/b.prior"), data)
    v_10 = potential_at_truth(prior_with("mco_floor = 10\n", f"{scratch}/c.prior"), data)
    v_min = potential_at_truth(
        prior_with("mco_floor = 4\n", f"{scratch}/d.prior"), data)
    for label, (v, gn) in (("default", v_def), ("mco_floor=6", v_6),
                           ("mco_floor=10", v_10), ("mco_floor=4", v_min)):
        print(f"  {label:14s} potential={v:+.8e}  |grad|={gn:.6e}")

    check(abs(v_6[0] - v_def[0]) <= 1e-4,
          "explicit mco_floor = 6 reproduces the default")
    check(abs(v_10[0] - v_def[0]) > 1e-3 * abs(v_def[0]),
          "mco_floor = 10 changes the potential (value is not ignored)")
    check(abs(v_min[0] - v_def[0]) > 1e-4 * abs(v_def[0]),
          "mco_floor = 4 (no flattening) changes the potential")
    check(all(np.isfinite(v) and np.isfinite(g)
              for v, g in (v_def, v_6, v_10, v_min)),
          "all variants give finite potential and gradient")

    # ---- 3. mco_floor reaches the grid through both class layers ----------
    print("\n=== 3. dataclass plumbing (LogDNDMDQDV -> LogDNDM -> LogDNDMPISN) ===")
    kw = dict(a=2.0, b=1.5, c=3.0, mpisn=33.0, mpisndot=0.0, mbhmax=37.0, sigma=0.06,
              fpl=0.5, beta=-2.0, lam=3.0, kappa=6.0, zp=1.9,
              mp_low=9.0, msigma_low=4.0, flow=0.5)
    m = jnp.asarray([5.0, 8.0, 20.0, 35.0])
    d6 = fast.LogDNDMDQDV(**kw)
    d10 = fast.LogDNDMDQDV(mco_floor=10.0, **kw)
    check(float(d6.log_dndm.mco_floor) == 6.0 and float(d10.log_dndm.mco_floor) == 10.0,
          "LogDNDM receives mco_floor from LogDNDMDQDV")
    check(float(d6.log_dndm.log_dndm_pisn.mco_floor) == 6.0
          and float(d10.log_dndm.log_dndm_pisn.mco_floor) == 10.0,
          "LogDNDMPISN receives mco_floor from LogDNDM")
    g6 = np.asarray(d6.log_dndm(m, 0.0))
    g10 = np.asarray(d10.log_dndm(m, 0.0))
    check(not np.allclose(g6, g10, atol=1e-4),
          f"the mass function changes with mco_floor (max|d|={np.max(np.abs(g6-g10)):.3e})")

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("all mco_floor plumbing checks passed")


if __name__ == "__main__":
    main()

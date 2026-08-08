"""Is the tab2d-vs-direct difference just z-grid resolution?

Compares the potential and every parameter gradient at the test-7 point for
the direct path and the consistent 2-D table at several n_z.  If the table is
merely a coarse-z approximation of the same model, both must converge to the
direct values as n_z grows.
"""
import sys
sys.path.append("../src/")

import numpy as np
import jax
import jax.numpy as jnp
import intensity_models_fast as fast
from test_fast_equivalence import make_synthetic_data, build_prior, TRUTH
from numpyro.infer.util import initialize_model
from numpyro.infer import init_to_value

nobs, nsamp, nsel = 400, 300, 40000
data = make_synthetic_data(nobs, nsamp, nsel, seed=19)
prior = build_prior(False, "/tmp/diag_nz.prior")
truth = {k: jnp.asarray(v) for k, v in TRUTH.items()
         if k in prior and not isinstance(prior[k], float)}
truth["mpisndot"] = jnp.asarray(3.0)
model_args = (data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
              data["m1s_det_sel"], data["qs_sel"], data["dls_sel"],
              data["pdraw_sel"], data["Ndraw"], prior)

CASES = [
    ("direct",      dict(tabulate_mass_function=False)),
    ("split n_z=30", dict(tabulate_selection=False)),
    ("tab  n_z=30", dict()),
    ("tab  n_z=60", dict(n_z=60)),
    ("tab  n_z=120", dict(n_z=120)),
    ("tab  n_z=240", dict(n_z=240)),
]
# n_z also sets the PISN grid resolution, so "direct" at the same n_z is the
# honest reference for each: compare tab(n_z) against direct(n_z).
res = {}
for label, kw in CASES:
    nz = kw.get("n_z", 30)
    for sub, extra in ((label, kw), (f"direct n_z={nz}",
                                     dict(tabulate_mass_function=False, n_z=nz))):
        if sub in res:
            continue
        mi = initialize_model(
            jax.random.PRNGKey(0), fast.pop_cosmo_model, model_args=model_args,
            model_kwargs=dict(use_low_bump=True, neff_penalty="none",
                              smooth_tail_edge=True, **extra),
            dynamic_args=False, init_strategy=init_to_value(values=truth),
        )
        v, g = jax.jit(jax.value_and_grad(mi.potential_fn))(mi.param_info.z)
        res[sub] = (float(v), {k: float(x) for k, x in g.items()})

keys = sorted(res["direct"][1])
print(f"\n{'case':16s} {'potential':>14s} {'d-vs-direct(nz)':>16s} "
      f"{'max|dgrad|':>11s} {'max rel':>9s}  worst")
for label, kw in CASES:
    nz = kw.get("n_z", 30)
    ref = res[f"direct n_z={nz}"]
    v, g = res[label]
    d = {k: g[k] - ref[1][k] for k in keys}
    worst = max(keys, key=lambda k: abs(d[k]) / max(abs(ref[1][k]), 1.0))
    print(f"{label:16s} {v:14.6f} {v - ref[0]:16.6f} "
          f"{max(abs(x) for x in d.values()):11.4f} "
          f"{abs(d[worst]) / max(abs(ref[1][worst]), 1.0):9.4f}  "
          f"{worst} ({ref[1][worst]:+.3f} -> {g[worst]:+.3f})")

print("\nper-parameter gradients (direct n_z=30 reference)")
print(f"{'param':12s} " + " ".join(f"{l:>14s}" for l, _ in CASES))
for k in keys:
    print(f"{k:12s} " + " ".join(f"{res[l][1][k]:14.4f}" for l, _ in CASES))

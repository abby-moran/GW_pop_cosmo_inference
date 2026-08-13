"""Diagnose the mpisndot-free pathology.

Compares the model potential (and its individual factors) at the truth point
and at the failed run's posterior mode, for the tabulated (default) and direct
mass-function paths.  Also profiles the potential along mpisndot.

Run from scripts/.
"""
import argparse
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--x64", action="store_true")
p.add_argument("--config", default="run_configs/mock_O5_evo.ini")
p.add_argument("--prior", default="../runs/priors/gwtc5_evo.prior")
p.add_argument("--nobs", type=int, default=None, help="truncate events")
p.add_argument("--no_smooth_edge", action="store_true")
p.add_argument("--module", default="intensity_models_fast")
p.add_argument("--profile", action="store_true")
args = p.parse_args()

if args.x64:
    os.environ["JAX_ENABLE_X64"] = "1"

sys.path.append("../src/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.handlers as handlers
import importlib

im = importlib.import_module(args.module)
from bench_model import load_real_data
from utils import get_priors_from_file

print(f"module={im.__file__}  x64={jax.config.jax_enable_x64}  {jax.devices()}")

data, cfg = load_real_data(args.config)
if args.nobs:
    for k in ("m1s_det", "qs", "dls", "log_pdraw"):
        data[k] = data[k][: args.nobs]
nobs = data["m1s_det"].shape[0]
print(f"nobs={nobs} nsamp={data['m1s_det'].shape[1]} nsel={len(data['m1s_det_sel'])}")

prior = get_priors_from_file(args.prior)
free = sorted(k for k, v in prior.items() if not isinstance(v, float))
print("free:", free)

model_args = (
    data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
    data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
    data["Ndraw"], prior,
)

# ---- parameter points ----------------------------------------------------
TRUTH = dict(
    h=0.674, Om=0.315, w=-1.0, a=-0.9426, b=0.237, c=2.360,
    mpisn=33.29, dmbhmax=36.7345 - 33.29, sigma=0.0539,
    log_fpl=float(np.log(0.63909)), lam=4.814, dkappa=8.3659 - 4.814,
    zp=0.954, beta=-2.43, msigma_low=4.0, mp_low=9.121,
    log_flow=float(np.log(0.6025)), mpisndot=0.0, Omh2=0.315 * 0.674 ** 2,
)
# posterior mode of runs/endO5_evo/O5_evo.nc (all walls)
MODE = dict(
    TRUTH,
    a=-1.63622, b=-2.08626, c=1.50021, mpisn=38.50631, dmbhmax=0.50785,
    sigma=0.05012, log_fpl=0.57725, lam=4.21618, dkappa=3.83612, zp=1.00816,
    beta=-3.77565, mp_low=10.71271, msigma_low=3.69533, log_flow=-1.39055,
    mpisndot=-1.99017,
)


def factors(params, tabulate, smooth=None, tab_sel=None):
    kw = dict(use_low_bump=True)
    if tabulate is not None:
        kw["tabulate_mass_function"] = tabulate
    if tab_sel is not None:
        kw["tabulate_selection"] = tab_sel
    if smooth is not None:
        kw["smooth_tail_edge"] = smooth
    vals = {k: jnp.asarray(v) for k, v in params.items()
            if k in prior and not isinstance(prior[k], float)}
    with handlers.seed(rng_seed=0), handlers.substitute(data=vals):
        tr = handlers.trace(im.pop_cosmo_model).get_trace(*model_args, **kw)
    out = {}
    lp = 0.0
    for name, site in tr.items():
        if site["type"] != "sample":
            continue
        fn = site.get("fn")
        if fn is not None and type(fn).__name__ == "Unit":
            v = float(np.asarray(fn.log_factor))
            out[name] = v
            lp += v
        elif not site.get("is_observed", False) and site.get("value") is not None:
            try:
                v = float(np.sum(np.asarray(fn.log_prob(site["value"]))))
            except Exception:
                continue
            lp += v
    out["log_mu_sel"] = float(np.asarray(tr["log_mu_sel"]["value"]))
    out["min_neff"] = float(np.asarray(tr["min_neff"]["value"]))
    out["logpost"] = lp
    return out


smooth = False if args.no_smooth_edge else None
# (label, tabulate_mass_function, tabulate_selection)
CONFIGS = (
    ("split", None, False),    # the broken production path: PE table, sel direct
    ("tab_both", None, True),  # consistent, both tabulated
    ("direct", False, None),   # consistent, both direct
)
rows = []
for label, params in (("truth", TRUTH), ("mode", MODE)):
    for cl, tab, tsel in CONFIGS:
        f = factors(params, tab, smooth, tsel)
        rows.append((label, cl, f))
        print(f"{label:6s} {cl:9s} loglike={f['loglike']:+14.4f} "
              f"selfactor={f['selfactor']:+14.4f} log_mu_sel={f['log_mu_sel']:+9.5f} "
              f"logpost={f['logpost']:+14.4f} min_neff={f['min_neff']:.2f}")

print()
d = {(a, b): f for a, b, f in rows}
for cl, _, _ in CONFIGS:
    dl = d[("mode", cl)]["logpost"] - d[("truth", cl)]["logpost"]
    print(f"  [{cl:9s}] logpost(mode) - logpost(truth) = {dl:+.3f} "
          f"({'MODE WINS (pathological)' if dl > 0 else 'truth wins'})")
for label in ("truth", "mode"):
    for cl in ("split", "tab_both"):
        print(f"  [{label:5s}] {cl:9s} - direct: "
              f"loglike {d[(label,cl)]['loglike'] - d[(label,'direct')]['loglike']:+9.4f}  "
              f"selfactor {d[(label,cl)]['selfactor'] - d[(label,'direct')]['selfactor']:+9.4f}  "
              f"total {d[(label,cl)]['logpost'] - d[(label,'direct')]['logpost']:+9.4f}")

if args.profile:
    print("\n=== potential profile along mpisndot (other params at truth) ===")
    grid = [-1.99, -1.5, -1.0, -0.5, -0.2, 1e-6, 0.2, 0.5, 1.0, 2.0, 4.0, 7.9]
    print(f"{'mpisndot':>9s} {'tab loglike':>14s} {'tab self':>14s} {'tab tot':>14s} "
          f"{'dir loglike':>14s} {'dir self':>14s} {'dir tot':>14s} {'tab-dir':>10s}")
    for md in grid:
        pr = dict(TRUTH, mpisndot=md)
        ft = factors(pr, None, smooth)
        fd = factors(pr, False, smooth)
        print(f"{md:9.3f} {ft['loglike']:14.4f} {ft['selfactor']:14.4f} {ft['logpost']:14.4f} "
              f"{fd['loglike']:14.4f} {fd['selfactor']:14.4f} {fd['logpost']:14.4f} "
              f"{ft['logpost'] - fd['logpost']:10.4f}")

"""Trace plot (per-parameter trace + marginal density) of a finished run.

One row per FREE (sampled) parameter: the marginal density on the left, the
chain traces on the right, chains distinguished by color and divergences
ticked along the bottom of the trace panels.  The plotted set comes from
diagnose_run.free_parameters() with the prior run_inf.py embeds in the .nc
posterior attrs (or an explicit --prior), so derived deterministics that
shadow a sampled coordinate (mpisn when mpisn_ref is sampled, r when log_r
is sampled, log_flow when log_fpeak is sampled, ...) are excluded.  Without
a prior the sampled/derived split degrades to the site-name heuristic and a
warning is printed.

For mock runs, red truth lines are drawn in the density panels from the pop
config embedded in the .nc (same mapping as plot_corner.py); real-data runs
have no truths and get none.

Usage:
    uv run python plot_trace.py --run realGWTC5_noevo_259ev_perz
    uv run python plot_trace.py --run endO5_evo2 --params mpisn_ref dmbhmax
"""
import argparse
import atexit
import os
import sys
import tempfile

sys.path.append("../src")

import numpy as np
import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_run import free_parameters, ini_from_attrs
from plot_corner import parse_truths


def resolve_prior(prior_path, attrs):
    """Parsed prior dict (utils.get_priors_from_file): an explicit --prior
    wins, else the prior_file_contents embedded in the .nc posterior attrs
    (spilled to a temp file, since the parser wants a path).  Returns
    (prior_dict_or_None, source_note)."""
    src = prior_path
    if prior_path is None and "prior_file_contents" in attrs:
        fd, prior_path = tempfile.mkstemp(prefix="trace_prior_", suffix=".prior")
        with os.fdopen(fd, "w") as f:
            f.write(attrs["prior_file_contents"])
        atexit.register(os.unlink, prior_path)
        src = ("embedded prior_file_contents (%s)"
               % attrs.get("run_config_prior", "?"))
    if prior_path is None:
        return None, None
    from utils import get_priors_from_file
    return get_priors_from_file(prior_path), src


def resolve_truths(args, post):
    """Truths for mock runs, same precedence as plot_corner: an explicit
    --pop_config, else the pop config embedded in the .nc attrs; embedded
    pop_config_file 'none'/absent means a real-data run, no truths."""
    if args.pop_config:
        with open(args.pop_config) as f:
            return parse_truths(f.read()), args.pop_config
    attrs = post.attrs
    name = attrs.get("run_config_pop_config_file", "none")
    if "pop_config_file_contents" in attrs and str(name).lower() != "none":
        return (parse_truths(attrs["pop_config_file_contents"]),
                f"embedded pop_config_file_contents ({name})")
    return {}, "none (real-data run, no truths)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run_dir under ../runs")
    p.add_argument("--nc", default=None,
                   help="NetCDF name (default: the only .nc in run_dir)")
    p.add_argument("--params", nargs="+", default=None,
                   help="restrict to these parameters (default: all free "
                        "parameters)")
    p.add_argument("--prior", default=None,
                   help="prior file (default: the prior embedded in the .nc's "
                        "posterior attrs)")
    p.add_argument("--pop_config", default=None,
                   help="truth pop config path (default: the pop config "
                        "embedded in the .nc, if any; real-data runs have none)")
    p.add_argument("--out", default=None)
    p.add_argument("--runs_dir", default="../runs")
    args = p.parse_args()

    run_dir = os.path.join(args.runs_dir, args.run)
    if args.nc is None:
        ncs = sorted(f for f in os.listdir(run_dir) if f.endswith(".nc"))
        if len(ncs) != 1:
            sys.exit(f"expected exactly one .nc in {run_dir}, found {ncs}")
        args.nc = ncs[0]
    nc_path = os.path.join(run_dir, args.nc)
    out = args.out or os.path.join(run_dir, args.nc[:-3] + "_trace.png")

    idata = az.from_netcdf(nc_path)
    post = idata.posterior
    ini = ini_from_attrs(post.attrs) or {}

    prior, prior_src = resolve_prior(args.prior, post.attrs)
    if prior is None:
        print("WARNING: no prior (--prior not given, none embedded in the "
              ".nc); the sampled/derived split falls back to site names and "
              "derived aliases like mpisn/r may appear", file=sys.stderr)
    else:
        print(f"prior: {prior_src}")

    names = free_parameters(post, prior)
    if args.params:
        bad = [k for k in args.params
               if k not in post or set(post[k].dims) != {"chain", "draw"}
               or np.ptp(np.asarray(post[k].values)) == 0]
        if bad:
            sys.exit(f"not scalar varying posterior sites: {bad}; "
                     f"free parameters are: {', '.join(names)}")
        extra = [k for k in args.params if k not in names]
        if extra:
            print(f"note: {', '.join(extra)} are not free (sampled) "
                  "parameters; plotting them anyway")
        names = list(args.params)

    truths_all, truth_src = resolve_truths(args, post)
    print(f"truths: {truth_src}")
    nchain, ndraw = post.sizes["chain"], post.sizes["draw"]
    print(f"{nc_path}: {nchain} chains x {ndraw} draws, "
          f"{len(names)} parameters")
    print("  " + ", ".join(names))

    az.rcParams["plot.max_subplots"] = max(2 * len(names), 40)
    axes = az.plot_trace(
        idata,
        var_names=names,
        compact=False,
        combined=False,
        divergences="bottom",
        figsize=(12, 1.6 * len(names)),
    )
    for row, name in zip(axes, names):
        t = truths_all.get(name)
        if t is not None:
            row[0].axvline(t, color="red", lw=1.5)
    fig = axes.ravel()[0].figure
    fig.suptitle(args.run, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main()

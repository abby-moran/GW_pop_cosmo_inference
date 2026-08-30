"""Corner plot of a finished inference run, in the house style.

Matches runs/endO5_{val,val2,fullcosmo_evo2}/*_corner.png: corner.corner with
red truth crosshairs, dashed 16/50/84% quantiles and median +/- titles.

Parameters are drawn from ORDER below, skipping any that the run held fixed
(zero posterior variance), so the same call works whether cosmology and/or
mpisndot were sampled.  Truths come from the pop config run_inf.py embeds in
the .nc posterior attrs (or, for older .nc's, a pop_config file -- pass
--pop_config, or --config with the run .ini), mapped into the derived
parameterisation the model samples in (kappa -> dkappa, etc.), the same way
run_inf.load_true_vals does.

Usage:
    uv run python plot_corner.py --run endO5_fullcosmo_evo3
    uv run python plot_corner.py --run endO5_evo2 --nc O5_evo2.nc
"""
import argparse
import os
import sys

import numpy as np
import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

# Plot order and labels, matching the existing figures.
ORDER = [
    ("h", r"$h$"),
    ("Omh2", r"$\Omega_M h^2$"),
    ("w", r"$w$"),
    ("mpisndot", r"$\dot{m}_{\rm PISN}$"),
    ("a", r"$a$"),
    ("b", r"$b$"),
    ("c", r"$c$"),
    ("mpisn", r"$m_{\rm PISN}$"),
    ("dmbhmax", r"$\Delta m^{\rm max}_{\rm BH}$"),
    ("sigma", r"$\sigma$"),
    ("log_fpl", r"$\log f_{\rm pl}$"),
    ("log_r", r"$\log r$"),
    ("lam", r"$\lambda$"),
    ("dkappa", r"$\Delta \kappa$"),
    ("zp", r"$z_p$"),
    ("beta", r"$\beta$"),
    ("msigma_low", r"$\sigma_{\rm low}$"),
    ("mp_low", r"$m_{p,\rm low}$"),
    ("log_flow", r"$\log f_{\rm low}$"),
    ("log_fpeak", r"$\log f_{\rm peak}$"),
]

# Bump-amplitude coordinates that are alternatives for the same degree of
# freedom.  Sampling log_fpeak makes the model record log_flow as a *varying*
# deterministic (intensity_models_fast.py:1181), so both are present and
# non-constant in the posterior; plotting both would duplicate one parameter
# and invent a perfect correlation.  When the sampled coordinate is present,
# drop its aliases.  Same table as scripts/diagnose_run.py.
AMPLITUDE_ALIASES = [
    ("log_fpeak", ("log_flow", "logit_flow")),
    ("logit_flow", ("log_flow",)),
]


def parse_truths(text):
    """Same mapping as run_inf.load_true_vals: pop configs store the physical
    parameters, the model samples the derived ones."""
    tv = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            tv[k.strip()] = float(v.strip())
    tv["dkappa"] = tv["kappa"] - tv["lam"]
    tv["dmbhmax"] = tv["mbhmax"] - tv["mpisn"]
    tv["log_fpl"] = np.log(tv["fpl"])
    tv["log_flow"] = np.log(tv["flow"]) if "flow" in tv else np.log(1e-5)
    # Peak-height parametrization: log_fpeak = log_flow - log(msigma_low).
    if "msigma_low" in tv:
        tv["log_fpeak"] = tv["log_flow"] - np.log(tv["msigma_low"])
    if "Omh2" not in tv and "Om" in tv and "h" in tv:
        tv["Omh2"] = tv["Om"] * tv["h"] ** 2
    return tv


def load_truths(path):
    with open(path) as f:
        return parse_truths(f.read())


def resolve_truths(args, post):
    """Truth values, by precedence: an explicit --pop_config; the pop config
    text run_inf.py embeds in the posterior attrs (run_config_pop_config_file
    = 'none' or missing contents = real-data run, no truths); the
    pop_config_file named by an explicit --config .ini; else the historical
    default pop config.  Returns (truths_dict, source_note)."""
    import configparser
    if args.pop_config:
        return load_truths(args.pop_config), args.pop_config
    attrs = post.attrs
    if any(k.startswith("run_config_") for k in attrs):
        name = attrs.get("run_config_pop_config_file", "none")
        if "pop_config_file_contents" in attrs and str(name).lower() != "none":
            return (parse_truths(attrs["pop_config_file_contents"]),
                    f"embedded pop_config_file_contents ({name})")
        return {}, "embedded attrs: real-data run, no truths"
    if args.config:
        cfg = configparser.ConfigParser()
        cfg.read(args.config)
        name = cfg["run"].get("pop_config_file") if "run" in cfg else None
        if name is None or name.lower() == "none":
            return {}, f"{args.config}: pop_config_file none/absent, no truths"
        for d in ("pop_configs", os.path.join("pop_configs", "archive")):
            path = os.path.join(d, name)
            if os.path.exists(path):
                return load_truths(path), path
        sys.exit(f"pop_config_file = {name} (from {args.config}) not found "
                 f"under pop_configs/")
    return load_truths(DEFAULT_POP_CONFIG), DEFAULT_POP_CONFIG


DEFAULT_POP_CONFIG = "pop_configs/mock_O5_noevo.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run_dir under ../runs")
    p.add_argument("--nc", default=None, help="NetCDF name (default: the only .nc in run_dir)")
    p.add_argument("--pop_config", default=None,
                   help="truth pop config path (default: the pop config "
                        "embedded in the .nc's posterior attrs, else "
                        f"{DEFAULT_POP_CONFIG})")
    p.add_argument("--config", default=None,
                   help="run .ini used as fallback metadata source when the "
                        ".nc carries no embedded run_config_* attrs")
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
    out = args.out or os.path.join(run_dir, args.nc[:-3] + "_corner.png")

    post = az.from_netcdf(nc_path).posterior
    truths_all, truth_src = resolve_truths(args, post)
    print(f"truths: {truth_src}")

    # Resolve alternative parametrizations of the bump amplitude before
    # selecting columns: whichever coordinate was sampled wins, its aliases are
    # derived and would duplicate the same degree of freedom.
    skip = set()
    for sampled, aliases in AMPLITUDE_ALIASES:
        if sampled in post and np.ptp(np.asarray(post[sampled].values)) > 0:
            for a in aliases:
                if a in post:
                    skip.add(a)

    names, labels, cols, truths = [], [], [], []
    for k, lab in ORDER:
        if k not in post or k in skip:
            continue
        v = np.asarray(post[k].values).ravel()
        # Exact range test, not a std threshold: a parameter pinned to a single
        # float32 value can still return std ~1e-7 from roundoff in the mean
        # (h = 0.674 does; Om = 0.315 happens to return exactly 0), which let
        # a constant column reach corner.corner and abort the whole figure
        # with "column(s) have no dynamic range".
        if np.ptp(v) == 0:           # held fixed in this run
            continue
        names.append(k)
        labels.append(lab)
        cols.append(v)
        truths.append(truths_all.get(k, None))

    samples = np.column_stack(cols)
    print(f"{nc_path}: {samples.shape[0]} draws x {samples.shape[1]} free parameters")
    print("  " + ", ".join(names))

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        truth_color="red",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs=dict(fontsize=9),
        label_kwargs=dict(fontsize=11),
    )
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

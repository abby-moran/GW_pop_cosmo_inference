"""Posterior predictive distributions of a finished inference run.

Three panels: rate density vs m1 (log-log), vs q, and vs z ("R(z)"), each
with the pointwise posterior median and a 90%-credible band (5th-95th
percentile), plus the true population in red.  Run metadata (the truth pop
config and smooth_tail_edge) is read from the run_config_* / *_file_contents
attrs run_inf.py embeds in the .nc posterior; older .nc's fall back to the
run .ini (an explicit --config path, else the .ini in run_dir), whose
`pop_config_file` follows run_inf.py's own convention ('none'/absent =
real-data run, so no truth overlay).  Override with --pop_config or suppress
with --no_truth.

Truth-curve normalization: pop configs carry no rate parameter, so the
truth shape is anchored at its self-consistent rate on this run's data,
R_true = nobs / mu_sel(truth).  run_inf recenters at truth and stores the
physical log_mu_sel(truth) as the posterior attr `log_pdraw_sel_scale`;
nobs is recovered exactly from the recorded R/R_unit/log_mu_sel draws.
Do NOT anchor at the posterior-median R: R is the density at the single
reference point (m1=30, q=1, z~0), so any local shape misfit there (e.g.
an overestimated sigma widening the PISN peak across m=30) transfers to
the truth curve as a spurious global offset.  When the attr is absent
(no truth recentering) the script falls back to median R with a warning.

Two modes:

* default: read the fixed-slice deterministics the fast model stores in the
  posterior (intensity_models_fast.py, `coords` grids) -- conditional rate
  densities with the *other* parameters held at reference (q=1, z~0, m1=30):
      mdNdmdVdt_fixed_qz   m1 dN/dm1 dq dV dt at q=1,  z=zref
      dNdqdVdt_fixed_mz    mref dN/dm1 dq dV dt at m1=30, z=zref
      dNdVdt_fixed_mq      mref dN/dm1 dq dV dt at m1=30, q=1
* --marginal: rebuild LogDNDMDQDV per (thinned) posterior draw and integrate
  out the other dimensions -- true marginal PPDs:
      m1 dN/dm1 dV dt at z=zref    (integrated over q)
      dN/dq dV dt at z=zref        (integrated over m1)
      R(z) = dN/dV dt              (integrated over m1 and q)

Usage:
    uv run python plot_ppd.py --run endO5_fullcosmo_evo7-redo
    uv run python plot_ppd.py --run endO5_narrowbump_d10 --marginal --ndraw 300
"""
import argparse
import os
import sys

sys.path.append('../src/')

import numpy as np
import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Stored-slice variable names and the grids they were evaluated on
# (intensity_models_fast.coords).
SLICE_VARS = ("mdNdmdVdt_fixed_qz", "dNdqdVdt_fixed_mz", "dNdVdt_fixed_mq")

# Parameters build_population_model reads directly from the sample dict.
# Missing entries fall back to these values (with a printed note); the
# bump parameters (mp_low, msigma_low, flow) are handled via use_low_bump.
PARAM_DEFAULTS = {
    "a": None, "b": None, "c": None, "mpisn": None, "mbhmax": None,
    "sigma": None, "fpl": None, "beta": None, "lam": None, "kappa": None,
    "zp": None,
    "mpisndot": 0.0, "zmax": 20.0, "mbh_min": 3.0, "delta_m": 2.5,
    "mco_min": 4.0, "mco_floor": 6.0,
}
BUMP_KEYS = ("mp_low", "msigma_low", "flow")


def parse_truths(text):
    """Pop configs store the *physical* parameters (kappa, fpl, mbhmax, flow),
    which is what build_population_model wants -- no coordinate mapping."""
    tv = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            tv[k.strip()] = float(v.strip())
    return tv


def load_truths(path):
    with open(path) as f:
        return parse_truths(f.read())


def ini_getboolean(value, fallback):
    """configparser-style boolean for a raw ini string ('true', 'yes', ...)."""
    if value is None:
        return fallback
    import configparser
    return configparser.ConfigParser.BOOLEAN_STATES[str(value).strip().lower()]


def truth_anchor_R(post):
    """Self-consistent rate for the truth shape: R_true = nobs / mu_sel(truth).

    The physical log_mu_sel at the truth point is the posterior attr
    `log_pdraw_sel_scale` (run_inf.py recenters at truth and stores it).
    nobs is recovered exactly from the model's R definition,
    R = (nobs + sqrt(nobs) R_unit) / mu_sel, using the recorded draws.
    Returns (R_anchor, label); falls back to the posterior-median R when
    the attr is missing (run not recentered at truth)."""
    R_med = float(np.median(np.asarray(post["R"].values)))
    log_mu_sel_true = post.attrs.get("log_pdraw_sel_scale")
    if log_mu_sel_true is None or "R_unit" not in post or "log_mu_sel" not in post:
        print("warning: no truth-recentering attr in posterior; anchoring the "
              "truth curve at the posterior-median R (level not meaningful)")
        return R_med, "median R"
    take = 32  # any handful of draws gives the same nobs up to float32 noise
    x = (np.asarray(post["R"].values).reshape(-1)[:take].astype(np.float64)
         * np.exp(np.asarray(post["log_mu_sel"].values).reshape(-1)[:take].astype(np.float64)))
    u = np.asarray(post["R_unit"].values).reshape(-1)[:take].astype(np.float64)
    sqrt_n = 0.5 * (-u + np.sqrt(u * u + 4 * x))   # positive root of n + sqrt(n) u = x
    nobs = round(float(np.median(sqrt_n ** 2)))
    R_true = nobs / np.exp(float(log_mu_sel_true))
    print(f"truth anchor: R_true = nobs/mu_sel(truth) = {R_true:.3g} "
          f"(nobs={nobs}, median posterior R = {R_med:.3g})")
    return R_true, "R_true"


def quantile_band(arr, level=0.90):
    """Pointwise median and equal-tailed credible band over draws (axis 0)."""
    lo, hi = 0.5 - level / 2, 0.5 + level / 2
    return (np.median(arr, axis=0),
            np.quantile(arr, lo, axis=0),
            np.quantile(arr, hi, axis=0))


def build_truth_model(tv, smooth_tail_edge):
    """LogDNDMDQDV at the true parameters (import deferred: jax is only
    needed for the truth overlay and --marginal)."""
    from intensity_models_fast import build_population_model
    use_low_bump = tv.get("flow", 0.0) > 0 and "mp_low" in tv
    sample = {k: tv.get(k, d) for k, d in PARAM_DEFAULTS.items()}
    missing = [k for k, v in sample.items() if v is None]
    if missing:
        sys.exit(f"pop_config is missing required parameters: {missing}")
    for k in BUMP_KEYS:
        if k in tv:
            sample[k] = tv[k]
    return build_population_model(sample, use_low_bump=use_low_bump,
                                  smooth_tail_edge=smooth_tail_edge)


def read_run_ini(run_dir, config_path=None):
    """Parse the .ini run_inf.py copies into run_dir (or, when given, an
    explicit --config path, which wins).  Returns the [run] section and its
    filename, or (None, None) when no parseable .ini is found."""
    import configparser
    if config_path is not None:
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        if "run" not in cfg:
            sys.exit(f"--config {config_path} has no [run] section")
        return cfg["run"], config_path
    for f in sorted(os.listdir(run_dir)):
        if not f.endswith(".ini"):
            continue
        cfg = configparser.ConfigParser()
        try:
            cfg.read(os.path.join(run_dir, f))
            if "run" in cfg:
                return cfg["run"], f
        except (configparser.Error, ValueError):
            continue
    return None, None


def resolve_pop_config(run_ini, ini_name):
    """Mirror run_inf.py's convention: `pop_config_file` in the run's .ini
    names the truth config; absent or 'none' means a real-data run with no
    truth.  Searches pop_configs/ and pop_configs/archive/."""
    if run_ini is None:
        return None
    name = run_ini.get("pop_config_file")
    if name is None or name.lower() == "none":
        return None
    for d in ("pop_configs", os.path.join("pop_configs", "archive")):
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
    print(f"warning: pop_config_file = {name} (from {ini_name}) not found "
          f"under pop_configs/; skipping the truth overlay")
    return None


def marginal_grids(zmax_plot, coords):
    """Evaluation grids for --marginal: reuse the stored m/q grids so both
    modes plot on the same abscissae; z re-gridded to the plotted range."""
    mg = np.asarray(coords["m_grid"])
    qg = np.asarray(coords["q_grid"])
    zg = np.expm1(np.linspace(0.0, np.log1p(zmax_plot), 96))
    zg[0] = 1e-3  # zref: avoid z=0 exactly, matches the model's reference
    # If zmax_plot coincides with the model's zmax the endpoint sits exactly on
    # the hard z < zmax cutoff and evaluates to zero; nudge it just inside.
    # Relative nudge so it survives float32 rounding at any magnitude
    # (an absolute 1e-6 is below half a ULP for float32 values >~ 17).
    zg[-1] *= 1 - 1e-6
    return mg, qg, zg


def make_marginal_fn(mg, qg, zg, zref):
    """Jittable map draw-params -> (m1 marginal, q marginal, R(z)).

    Any approximation here is plotting-only (the likelihood is untouched), so
    plain trapezoids on the stored grids are fine.
    """
    import jax
    import jax.numpy as jnp
    from intensity_models_fast import build_population_model

    def _one(params, use_low_bump, smooth_tail_edge):
        ld = build_population_model(params, use_low_bump=use_low_bump,
                                    smooth_tail_edge=smooth_tail_edge)
        R = params["R"]
        # z ~ 0 plane for the mass/mass-ratio marginals: (nm, nq)
        plane = jnp.exp(ld(mg[:, None], qg[None, :], zref))
        m_marg = R * mg * jnp.trapezoid(plane, qg, axis=1)
        q_marg = R * jnp.trapezoid(plane, mg, axis=0)
        # full (nz, nm, nq) cube for the volumetric rate R(z)
        cube = jnp.exp(ld(mg[None, :, None], qg[None, None, :],
                          zg[:, None, None]))
        Rz = R * jnp.trapezoid(jnp.trapezoid(cube, qg, axis=2), mg, axis=1)
        return m_marg, q_marg, Rz

    return _one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run_dir under ../runs")
    p.add_argument("--nc", default=None, help="NetCDF name (default: the only .nc in run_dir)")
    p.add_argument("--pop_config", default=None,
                   help="truth pop config path (default: the pop config "
                        "embedded in the .nc's posterior attrs, else "
                        "auto-resolve the run .ini's pop_config_file; "
                        "'none'/absent there means a real-data run, so no "
                        "truth overlay)")
    p.add_argument("--config", default=None,
                   help="run .ini used as fallback metadata source when the "
                        ".nc carries no embedded run_config_* attrs (wins "
                        "over any auto-discovered .ini in run_dir)")
    p.add_argument("--out", default=None)
    p.add_argument("--runs_dir", default="../runs")
    p.add_argument("--marginal", action="store_true",
                   help="integrate out the other dimensions per draw instead of "
                        "reading the stored fixed-slice deterministics")
    p.add_argument("--ndraw", type=int, default=500,
                   help="posterior draws used in --marginal mode (thinned at random)")
    p.add_argument("--zmax_plot", type=float, default=6.5,
                   help="upper edge of the R(z) panel (mock catalogs: zmax=6.5)")
    p.add_argument("--level", type=float, default=0.90, help="credible-band level")
    p.add_argument("--no_truth", action="store_true",
                   help="skip the truth overlay (real-catalog runs have no "
                        "true population; --pop_config is then ignored)")
    p.add_argument("--hard_tail_edge", action="store_true",
                   help="force smooth_tail_edge=False for the truth/marginal "
                        "models (default: auto-detect from the .nc's embedded "
                        "attrs, else the .ini copied into run_dir, falling "
                        "back to run_inf's True)")
    p.add_argument("--seed", type=int, default=42, help="thinning RNG seed")
    args = p.parse_args()

    run_dir = os.path.join(args.runs_dir, args.run)
    if not os.path.isdir(run_dir):
        sys.exit(f"no such run directory: {run_dir}")
    if args.nc is None:
        ncs = sorted(f for f in os.listdir(run_dir) if f.endswith(".nc"))
        if len(ncs) != 1:
            sys.exit(f"expected exactly one .nc in {run_dir}, found {ncs}")
        args.nc = ncs[0]
    nc_path = os.path.join(run_dir, args.nc)
    suffix = "_ppd_marginal.png" if args.marginal else "_ppd.png"
    out = args.out or os.path.join(run_dir, args.nc[:-3] + suffix)

    post = az.from_netcdf(nc_path).posterior
    attrs = post.attrs
    # run_inf.py >= 1383bbc embeds the whole [run] section as run_config_*
    # attrs (plus the verbatim prior / pop config text); prefer those, and
    # only fall back to loose config files for older .nc's.
    embedded = any(k.startswith("run_config_") for k in attrs)
    if embedded:
        print("run metadata: embedded posterior attrs (run_config_*, from "
              f"{attrs.get('run_config_file', '?')})")
        run_ini, ini_name = None, None
    else:
        print("run metadata: no embedded attrs; using config-file fallback")
        run_ini, ini_name = read_run_ini(run_dir, args.config)
        if run_ini is None:
            print("note: no .ini found in run_dir")

    if args.hard_tail_edge:
        smooth_tail_edge = False
    elif embedded:
        smooth_tail_edge = ini_getboolean(
            attrs.get("run_config_smooth_tail_edge"), True)
        print(f"smooth_tail_edge={smooth_tail_edge} (from embedded attrs)")
    elif run_ini is None:
        smooth_tail_edge = True  # run_inf's fallback
        print("note: assuming smooth_tail_edge=True")
    else:
        smooth_tail_edge = run_ini.getboolean("smooth_tail_edge", fallback=True)
        print(f"smooth_tail_edge={smooth_tail_edge} (from {ini_name})")

    # Truth source, in precedence order: --no_truth / --pop_config, then the
    # embedded pop config text (run_config_pop_config_file = 'none' or a
    # missing pop_config_file_contents means a real-data run with no truth),
    # then the .ini-based resolution.
    truths, truth_src = None, None
    if not args.no_truth:
        if args.pop_config:
            truths, truth_src = load_truths(args.pop_config), args.pop_config
        elif embedded:
            name = attrs.get("run_config_pop_config_file", "none")
            if ("pop_config_file_contents" in attrs
                    and str(name).lower() != "none"):
                truths = parse_truths(attrs["pop_config_file_contents"])
                truth_src = f"embedded pop_config_file_contents ({name})"
        else:
            pop_config = resolve_pop_config(run_ini, ini_name)
            if pop_config is not None:
                truths, truth_src = load_truths(pop_config), pop_config
    if truths is None and not args.no_truth:
        print("no truth pop config for this run; skipping the truth overlay")
    R_anchor = truth_anchor_R(post)[0] if truths is not None else None

    from intensity_models_fast import coords, LogDNDMDQDV  # noqa: E402  (needs sys.path)
    if truths is None:
        ld_true = None
        zref = float(LogDNDMDQDV.zref)
        mref = float(LogDNDMDQDV.mref)
        qref = float(LogDNDMDQDV.qref)
    else:
        print(f"truth pop config: {truth_src}")
        ld_true = build_truth_model(truths, smooth_tail_edge)
        zref = float(ld_true.zref)
        mref = float(ld_true.mref)
        qref = float(ld_true.qref)

    if args.marginal:
        import jax
        import jax.numpy as jnp

        mg, qg, zg = marginal_grids(args.zmax_plot, coords)
        use_low_bump = "flow" in post  # only recorded when the bump is on
        param_keys = [k for k in PARAM_DEFAULTS if k in post]
        defaulted = {k: d for k, d in PARAM_DEFAULTS.items() if k not in post}
        if any(v is None for v in defaulted.values()):
            sys.exit(f"posterior is missing required parameters: "
                     f"{[k for k, v in defaulted.items() if v is None]}")
        if defaulted:
            print(f"note: not in posterior, using defaults: {defaulted}")

        nsamp = post["R"].values.size
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(nsamp, size=min(args.ndraw, nsamp), replace=False)
        stack = {k: jnp.asarray(np.asarray(post[k].values).reshape(-1)[idx])
                 for k in param_keys + (list(BUMP_KEYS) if use_low_bump else []) + ["R"]}
        for k, d in defaulted.items():
            stack[k] = jnp.full(idx.size, d)

        one = make_marginal_fn(jnp.asarray(mg), jnp.asarray(qg), jnp.asarray(zg), zref)
        fn = jax.jit(lambda s: one(s, use_low_bump, smooth_tail_edge))
        print(f"evaluating marginal PPDs for {idx.size} draws "
              f"(use_low_bump={use_low_bump}, smooth_tail_edge={smooth_tail_edge}) ...")
        m_ppd, q_ppd, z_ppd = (np.asarray(a) for a in jax.lax.map(fn, stack))

        if ld_true is None:
            tm = tq = tz = None
        else:
            # truth marginals through the identical code path
            tm, tq, tz = (np.asarray(a) for a in one(
                {**{k: truths.get(k, PARAM_DEFAULTS[k]) for k in PARAM_DEFAULTS},
                 **{k: truths[k] for k in BUMP_KEYS if k in truths},
                 "R": R_anchor},
                truths.get("flow", 0.0) > 0, smooth_tail_edge))
        xz = zg
        ylab_m = r"$m_1\,\mathrm{d}N/\mathrm{d}m_1\,\mathrm{d}V\,\mathrm{d}t$" + f"  ($z={zref:g}$)"
        ylab_q = r"$\mathrm{d}N/\mathrm{d}q\,\mathrm{d}V\,\mathrm{d}t$" + f"  ($z={zref:g}$)"
        ylab_z = r"$R(z) = \mathrm{d}N/\mathrm{d}V\,\mathrm{d}t$"
    else:
        missing = [v for v in SLICE_VARS if v not in post]
        if missing:
            sys.exit(f"{nc_path} lacks {missing} (older run?); use --marginal instead")
        m_ppd, q_ppd, z_ppd = (
            np.asarray(post[v].values).reshape(-1, post[v].shape[-1])
            for v in SLICE_VARS)
        zmask = np.asarray(coords["z_grid"]) <= args.zmax_plot
        z_ppd = z_ppd[:, zmask]
        xz = np.asarray(coords["z_grid"])[zmask]

        if ld_true is None:
            tm = tq = tz = None
        else:
            # truth slices: same fixed-reference quantities as the deterministics
            mg_t, qg_t = np.asarray(coords["m_grid"]), np.asarray(coords["q_grid"])
            tm = mg_t * R_anchor * np.asarray(np.exp(ld_true(mg_t, qref, zref)))
            tq = mref * R_anchor * np.asarray(np.exp(ld_true(mref, qg_t, zref)))
            tz = mref * R_anchor * np.asarray(np.exp(ld_true(mref, qref, xz)))
        ylab_m = r"$m_1\,\mathrm{d}N/\mathrm{d}m_1\,\mathrm{d}q\,\mathrm{d}V\,\mathrm{d}t$" + f"  ($q=1$, $z={zref:g}$)"
        ylab_q = r"$m_\mathrm{ref}\,\mathrm{d}N/\mathrm{d}m_1\,\mathrm{d}q\,\mathrm{d}V\,\mathrm{d}t$" + f"  ($m_1={mref:g}$, $z={zref:g}$)"
        ylab_z = r"$m_\mathrm{ref}\,\mathrm{d}N/\mathrm{d}m_1\,\mathrm{d}q\,\mathrm{d}V\,\mathrm{d}t$" + f"  ($m_1={mref:g}$, $q=1$)"

    xm = np.asarray(coords["m_grid"])
    xq = np.asarray(coords["q_grid"])
    anchor_note = ("no truth overlay" if R_anchor is None
                   else f"truth anchor R = {R_anchor:.3g} Gpc^-3 yr^-1")
    print(f"{nc_path}: {m_ppd.shape[0]} draws, {anchor_note}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        (axes[0], xm, m_ppd, tm, r"$m_1\ [M_\odot]$", ylab_m, "log"),
        (axes[1], xq, q_ppd, tq, r"$q$", ylab_q, "linear"),
        (axes[2], xz, z_ppd, tz, r"$z$", ylab_z, "linear"),
    ]
    pct = 100 * args.level
    for ax, x, ppd, truth, xlab, ylab, xscale in panels:
        med, lo, hi = quantile_band(ppd, args.level)
        ax.fill_between(x, lo, hi, color="C0", alpha=0.25, lw=0,
                        label=f"{pct:.0f}% credible")
        ax.plot(x, med, color="C0", lw=1.5, label="median")
        if truth is not None:
            ax.plot(x, truth, color="red", ls="--", lw=1.2, label="truth")
        ax.set_xscale(xscale)
        ax.set_yscale("log")
        # trim the decades of zero-density floor without hiding structure
        ymax = np.nanmax(hi)
        ax.set_ylim(ymax * 1e-7, ymax * 3)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab + r"  $[\mathrm{Gpc^{-3}\,yr^{-1}}]$", fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].set_xlim(2, 300)
    axes[0].legend(loc="best", fontsize=8, framealpha=0.5)
    fig.suptitle(args.run)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

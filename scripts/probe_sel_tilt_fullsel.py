"""Probe whether using the full selection set would clear narrow-feature tilt.

Reuses the diagnose_run selection-tilt machinery (same weights, bootstrap,
thresholds) but evaluates it at several selection fractions of the HDF5 that
run_inf currently halves.  Stage-0 of the nsel-growth plan: if fraction=1.0
already brings every free narrow-feature param below FAIL, growing the pool
may be unnecessary for this run.

Usage (from ``scripts/``)::

    ../.venv/bin/python probe_sel_tilt_fullsel.py --run endO5_narrowbump_d10
    ../.venv/bin/python probe_sel_tilt_fullsel.py \\
        --nc ../runs/endO5_narrowbump_d10/O5_narrowbump_d10.nc \\
        --sel ../runs/endO5_narrowbump_d10/sel_narrowbump.h5 \\
        --pop_config pop_configs/mock_O5_narrowbump.txt \\
        --fractions 0.5,1.0

Exit 0 if full-sel (largest fraction, default includes 1.0) has worst severity
OK/NOTE/WARN; exit 1 if it still FAILs; exit 2 on usage errors.
"""
from __future__ import annotations

import argparse
import os
import sys

import arviz as az
import numpy as np
import pandas as pd

# Same directory as diagnose_run.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../src/")

import diagnose_run as dr


def _free_narrow(post, prior):
    free = dr.free_parameters(post, prior)
    return [p for p in dr.NARROW_FEATURE_PARAMS if p in free]


def _tilt_rows(sel, base, post, targets, nobs, lp_std, use_low_bump, nboot, seed):
    """One row per target: same math as diagnose_run.section_selection_tilt."""
    rows = []
    for name in targets:
        vals = np.asarray(post[name].values).ravel()
        lo, hi = float(np.percentile(vals, 16)), float(np.percentile(vals, 84))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        sample_lo = dict(base)
        sample_lo[name] = lo
        sample_hi = dict(base)
        sample_hi[name] = hi
        for s in (sample_lo, sample_hi):
            if "mpisn" in s and "dmbhmax" in s:
                s["mbhmax"] = float(s["mpisn"]) + float(s["dmbhmax"])
        lw_lo = dr._sel_log_wts(sel, sample_lo, use_low_bump=use_low_bump)
        lw_hi = dr._sel_log_wts(sel, sample_hi, use_low_bump=use_low_bump)
        # Fresh RNG stream per (fraction, param) but deterministic given seed.
        delta, sd = dr._bootstrap_delta_log_mu(
            lw_lo, lw_hi, nboot=nboot, seed=seed + hash(name) % 10007)
        noise = nobs * sd
        tilt = nobs * abs(delta)
        ratio = (noise / lp_std) if (lp_std and lp_std > 0) else None
        rows.append(dict(name=name, lo=lo, hi=hi, noise=noise, tilt=tilt,
                         ratio=ratio, nsel=len(sel),
                         sev=dr._sel_tilt_severity(noise, ratio)))
    return rows


def _worst(rows):
    worst = "OK"
    for r in rows:
        if dr.SEVERITY_ORDER[r["sev"]] > dr.SEVERITY_ORDER[worst]:
            worst = r["sev"]
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None,
                   help="run dir name under --runs_dir (e.g. endO5_narrowbump_d10)")
    p.add_argument("--nc", default=None, help="explicit .nc path")
    p.add_argument("--sel", default=None, help="explicit selection HDF5")
    p.add_argument("--pop_config", default=None)
    p.add_argument("--prior", default=None)
    p.add_argument("--runs_dir", default="../runs")
    p.add_argument("--priors_dir", default="../runs/priors")
    p.add_argument("--pop_configs_dir", default="pop_configs")
    p.add_argument("--run_configs_dir", default="run_configs")
    p.add_argument("--fractions", default="0.5,1.0",
                   help="comma-separated fractions of the sel HDF5 to use "
                        "(0.5 = run_inf default; 1.0 = Stage 0 full sel)")
    p.add_argument("--nboot", type=int, default=dr.SEL_TILT_NBOOT)
    p.add_argument("--seed", type=int, default=dr.SEL_TILT_SEED)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    # Fake a namespace so resolve_nc works with --run / --nc.
    class NS:
        pass
    ns = NS()
    ns.run, ns.nc, ns.runs_dir = args.run, args.nc, args.runs_dir
    try:
        nc_path, run_dir, run_name = dr.resolve_nc(ns)
    except SystemExit as e:
        return 2 if e.code is None else int(e.code)

    ini_path = dr.find_ini(run_dir, run_name, args.run_configs_dir)
    ini = dr.read_ini(ini_path) if ini_path else {}

    sel_file = args.sel or dr.resolve_sel_file(run_dir, ini)
    if not sel_file or not os.path.exists(sel_file):
        print("ERROR: selection HDF5 not found (pass --sel)", file=sys.stderr)
        return 2

    pop_path = args.pop_config
    if pop_path is None and ini.get("pop_config_file"):
        for cand in (os.path.join(args.pop_configs_dir, ini["pop_config_file"]),
                     os.path.join(run_dir, ini["pop_config_file"])):
            if os.path.exists(cand):
                pop_path = cand
                break
    truths = dr.load_truths(pop_path) if pop_path else None

    prior = None
    prior_path = args.prior
    if prior_path is None and ini.get("prior"):
        # basename or relative as stored in ini
        for cand in (os.path.join(args.priors_dir, os.path.basename(ini["prior"])),
                     os.path.join(args.priors_dir, ini["prior"]),
                     ini["prior"]):
            if os.path.exists(cand):
                prior_path = cand
                break
    if prior_path and os.path.exists(prior_path):
        try:
            from utils import get_priors_from_file
            prior = get_priors_from_file(prior_path)
        except Exception as exc:
            print("WARNING: prior parse failed (%s)" % exc, file=sys.stderr)

    nobs = None
    if ini.get("evt_end") is not None:
        nobs = int(ini["evt_end"]) - int(ini.get("evt_start") or 0)
    if nobs is None:
        print("ERROR: nobs unknown (need ini evt_end or pass a matching --run)",
              file=sys.stderr)
        return 2

    fractions = []
    for tok in args.fractions.split(","):
        tok = tok.strip()
        if not tok:
            continue
        f = float(tok)
        if not (0.0 < f <= 1.0):
            print("ERROR: fraction %s not in (0, 1]" % tok, file=sys.stderr)
            return 2
        fractions.append(f)
    if not fractions:
        print("ERROR: empty --fractions", file=sys.stderr)
        return 2

    idata = az.from_netcdf(nc_path)
    post = idata.posterior
    targets = _free_narrow(post, prior)
    if not targets:
        print("no free narrow-feature params among %s; nothing to probe"
              % ", ".join(dr.NARROW_FEATURE_PARAMS))
        return 0

    lp = None
    if hasattr(idata, "sample_stats") and "lp" in idata.sample_stats:
        lp = np.asarray(idata.sample_stats["lp"].values).ravel()
    lp_std = float(np.std(lp)) if lp is not None and lp.size else None

    post_med = dr._posterior_median_dict(post)
    base = dr._canonical_pop_sample(truths, post_med)
    use_low_bump = ini.get("use_low_bump", True)

    sel_all = pd.read_hdf(sel_file, key="true_parameters")
    n_all = len(sel_all)

    # Validate weights once on a tiny slice.
    try:
        dr._sel_log_wts(sel_all.iloc[:2], base, use_low_bump=use_low_bump)
    except Exception as exc:
        print("ERROR: cannot rebuild selection weights (%s)" % exc, file=sys.stderr)
        return 2

    print("=" * 72)
    print("SEL-TILT FULLSEL PROBE: %s" % run_name)
    print("=" * 72)
    print("nc         : %s" % nc_path)
    print("sel        : %s (%d rows)" % (sel_file, n_all))
    print("nobs       : %d" % nobs)
    print("lp_std     : %s" % ("%.3f nats" % lp_std if lp_std else "n/a"))
    print("targets    : %s" % ", ".join(targets))
    print("thresholds : FAIL if noise>=%.1f or noise/lp_std>=%.2f; "
          "WARN if noise>=%.1f or ratio>=%.2f"
          % (dr.SEL_TILT_NOISE_FAIL, dr.SEL_TILT_RATIO_FAIL,
             dr.SEL_TILT_NOISE_WARN, dr.SEL_TILT_RATIO_WARN))
    print("nboot      : %d (seed %d)" % (args.nboot, args.seed))
    print()

    results = {}
    half_rows = None
    for frac in fractions:
        n_use = max(1, int(np.round(n_all * frac)))
        sel = sel_all.iloc[:n_use]
        # Distinct seed offset per fraction so bootstraps aren't identical
        # when n_use changes only by padding (they're nested prefixes).
        rows = _tilt_rows(sel, base, post, targets, nobs, lp_std, use_low_bump,
                          args.nboot, args.seed + int(frac * 1000))
        results[frac] = rows
        if abs(frac - 0.5) < 1e-9:
            half_rows = {r["name"]: r for r in rows}

        worst = _worst(rows)
        tag = "  <- run_inf default" if abs(frac - 0.5) < 1e-9 else ""
        if abs(frac - 1.0) < 1e-9:
            tag = "  <- Stage 0 full sel"
        print("-" * 72)
        print("fraction=%.2f  nsel=%d  worst=%s%s"
              % (frac, n_use, worst, tag))
        print("%-12s %10s %10s %12s %8s" % (
            "param", "noise", "tilt", "noise/lp", "sev"))
        for r in rows:
            ratio_s = ("%.2f" % r["ratio"]) if r["ratio"] is not None else "n/a"
            print("%-12s %10.2f %10.2f %12s %8s" % (
                r["name"], r["noise"], r["tilt"], ratio_s, r["sev"]))

        if half_rows and abs(frac - 0.5) > 1e-9:
            print("  vs fraction=0.5 (noise ratio half/this; ~sqrt(N) => 0.71 "
                  "for 2x N):")
            for r in rows:
                h = half_rows.get(r["name"])
                if h and r["noise"] > 0:
                    print("    %s: %.2f -> %.2f  (half/full=%.2f, expect~%.2f)"
                          % (r["name"], h["noise"], r["noise"],
                             h["noise"] / r["noise"],
                             np.sqrt(n_use / max(1, int(np.round(n_all * 0.5))))))

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    full = results.get(1.0) or results[max(fractions)]
    full_frac = 1.0 if 1.0 in results else max(fractions)
    full_worst = _worst(full)
    half = results.get(0.5)

    print("At fraction=%.2f (nsel=%d): worst severity = %s"
          % (full_frac, full[0]["nsel"], full_worst))
    if half:
        print("At fraction=0.50 (nsel=%d): worst severity = %s"
              % (half[0]["nsel"], _worst(half)))

    if full_worst == "FAIL":
        print("Stage 0 alone would NOT clear the diagnose FAIL.  Proceed to "
              "grow the injection pool / reweight (Stage 1+).")
        rc = 1
    elif full_worst == "WARN":
        print("Stage 0 would clear FAIL but still WARN.  Full-sel rerun is "
              "worth doing; pool growth may still be needed to reach OK.")
        rc = 0
    else:
        print("Stage 0 would clear the tilt FAIL (severity %s).  A full-sel "
              "inference rerun is the efficient next step before building a "
              "larger pool." % full_worst)
        rc = 0

    # Idealized scaling: how many times more sel than half to hit ratio FAIL/WARN.
    if half and lp_std and lp_std > 0:
        print()
        print("Rough N multiplier vs half-sel to reach ratio thresholds "
              "(assuming noise ~ 1/sqrt(N)):")
        for r in half:
            if r["ratio"] is None or r["ratio"] <= 0:
                continue
            for label, thr in (("WARN(<0.5)", dr.SEL_TILT_RATIO_WARN),
                               ("OK-ish(<0.35)", 0.35)):
                if r["ratio"] > thr:
                    mult = (r["ratio"] / thr) ** 2
                    print("  %s -> %s: ~%.1fx half-sel (~%d rows)"
                          % (r["name"], label, mult,
                             int(np.round(r["nsel"] * mult))))

    if args.json:
        import json
        print(json.dumps(dict(
            run=run_name, nc=nc_path, sel=sel_file, nobs=nobs, lp_std=lp_std,
            fractions={str(f): results[f] for f in fractions},
            full_worst=full_worst,
        ), default=float, indent=2))

    return rc


if __name__ == "__main__":
    sys.exit(main())

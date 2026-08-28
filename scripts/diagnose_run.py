"""Health check for a finished inference run, from its saved .nc
(and, for a couple of checks, the run's PE / selection HDF5).

Reads the posterior + sample_stats an ``scripts/run_inf.py`` run wrote and turns
them into a severity-tagged report and a prioritized list of config changes.

Usage (run from ``scripts/``)::

    uv run python diagnose_run.py --run endO5_broadbump
    uv run python diagnose_run.py --nc ../runs/endO5_evo/O5_evo.nc \
                                  --prior ../runs/priors/gwtc5_evo.prior
    uv run python diagnose_run.py --run endO5_evo2 --json
    uv run python diagnose_run.py --run endO5_evo2 --test-float32

``--run`` resolves the single ``.nc`` in ``../runs/<name>/`` the same way
``plot_corner.py`` does.  ``--prior`` / ``--pop_config`` are optional; when
omitted they are read from an ``.ini`` found in the run directory, or from the
``scripts/run_configs/*.ini`` whose ``[run] run_dir`` matches.  Every check that
needs a file we could not find degrades to an explicit "insufficient
information" note rather than a guess.

EXIT CODE: 0 if the worst finding is OK/NOTE/WARN, 1 if any finding is FAIL,
2 on a usage error (no such run, ambiguous .nc).
A FAIL means the run's numbers should not be quoted as they stand.

--------------------------------------------------------------------------
Checks and thresholds (all sourced, not invented)
--------------------------------------------------------------------------
1. Sampler convergence
   r-hat            <=1.01 OK | <=1.05 NOTE | <=1.10 WARN | >1.10 FAIL
   bulk/tail ESS    >=200 OK | >=100 NOTE | >=50 WARN | <50 FAIL
                    (calibrated on runs/endO5_evo2, min bulk ESS 118, which is
                    the reference "healthy" run)
   divergences      0 OK | <0.2% NOTE | <1% WARN | >=1% FAIL
   BFMI             <0.3 WARN (Betancourt's standard threshold)
   lp r-hat         >1.05 WARN: chains sitting in different modes
   depth cap        inferred from max(n_steps) = 2^depth - 1, cross-checked
                    against the config's max_tree_depth (run_inf default 7)

2. Monte-Carlo adequacy, in both directions
   mc_var_loglike   = sum_i 1/n_eff_i, the MC variance of the *total* log
                    likelihood; guard budget mc_variance_budget = 5.0 (default
                    in intensity_models_fast.pop_cosmo_model).  MC sd of the
                    total log likelihood is sqrt(mc_var).
                    >=1.0*budget FAIL | >=0.8 WARN | <0.2*budget (i.e. <1.0)
                    NOTE that n_pe is stronger than needed.
                    Cost scales ~linearly in n_pe, variance ~1/n_pe, so the
                    script reports the largest safe n_pe reduction.
                    Source: notes/2026-08-07-neff-penalty-redesign.md
   min_neff         <2 WARN, <5 NOTE (single events carried by a few PE samples)
   neff_sel         model guard is neff_sel >= 4*nobs (Farr 2019).
                    <1x FAIL | <1.5x WARN | >3x NOTE (oversized selection set).
                    neff_sel can never exceed nsel, so nsel < 4*nobs makes the
                    guard permanently active: the 17,624-rows-vs-36,000-hinge
                    case from notes/2026-08-09-low-mass-bump-width-
                    identifiability.md is unusable by construction.
   sel-tilt noise   for narrow mass-feature parameters (sigma, msigma_low,
                    dmbhmax): bootstrap nobs*sd(Delta log_mu_sel) across the
                    parameter's posterior 16-84% range, with other params held
                    at truth (else posterior median).  The Farr hinge only
                    controls the global normalization error; a sharp feature's
                    identity is a ~99.5% cancellation between the PE numerator
                    and the selection term, so a few nats of parameter-
                    dependent selection MC noise can fake or erase the
                    measurement (endO5_fullcosmo_evo7: hinge 2.0x OK, but
                    nobs*sd ~ 3 nats on sigma wiped the truth preference).
                    Compare that noise to the posterior lp std (the log-
                    likelihood variation the sampler actually saw):
                    noise/lp_std >=1.0 FAIL | >=0.5 WARN | noise>=2 nats NOTE.
                    Absolute floors (same scale as mc_variance_budget):
                    noise>=5 FAIL | >=2 WARN.
                    Requires the run's selection HDF5 + a pop_config (or a
                    complete posterior) so it can rebuild selection weights;
                    otherwise degrades to "insufficient information".
   float32 safety   evaluate the numpyro potential (and its gradient) at the
                    truth (else posterior median) in float32 and float64, on
                    the run's own PE + selection data.  The truth is near the
                    worst case: float32 error tracks |loglike|, and NUTS lives
                    at high likelihood (notes/2026-08-07-float32-accuracy-
                    audit.md).  Two traps make a naive comparison lie:
                    (1) each precision's init_to_value z0 differs by ~1e-7,
                    and |dPE/dz|~1e5 fabricates ~1e-2 nats -- the real
                    effect's size -- so the float64 leg is forced onto the
                    float32 z0; (2) inputs are rounded to float32 first, so
                    the measurement is arithmetic precision not input
                    precision.  JAX x64 must be set before import, so the two
                    legs are subprocesses.  A constant potential offset is
                    absorbed into the target; a wrong gradient misdirects
                    trajectories.  Thresholds vs the ~0.2 nat leapfrog energy
                    scale at 80% accept (same note):
                    |dPE|>=0.2 or 1-cos(grad)>=1e-6 FAIL |
                    |dPE|>=0.1 or 1-cos>=1e-8 WARN.
                    Production-scale measured |dPE| is ~0.02 nats (OK).
                    Needs PE HDF5 + selection HDF5 + prior; else "insufficient
                    information".  Off by default; pass --test-float32.

3. Sampler geometry -> max_tree_depth / dense_mass / target_accept_prob / nmcmc
   Recommendations are conditioned on the joint evidence (saturation fraction,
   ESS, whether depth/dense_mass were *already* raised), never on saturation
   alone.  See notes/2026-08-07-optimization-changelog.md and the comments in
   run_inf.py.

4. Model choice / identifiability
   prior-dominated  posterior sd / prior sd >0.9 WARN, >0.6 NOTE
   prior walls      fraction of draws within 1% of the prior's effective range
                    of a truncation bound: >0.5 WARN, >0.1 NOTE.  >=3 params
                    above 0.3 simultaneously is the wall-mode signature of the
                    selection/event inconsistency bug ->  FAIL, and the script
                    names that note.
                    Source: notes/2026-08-08-tabulated-selection-consistency.md
   degeneracies     Pearson r and Spearman rho for the strongest pairs
                    (|r|>0.9 WARN, >0.7 NOTE); |rho|-|r| > 0.05 flags a curved
                    (banana) degeneracy rather than a linear one.
                    Eigen-decomposition of the free-parameter correlation
                    matrix; directions with eigenvalue >2 are reported.
   low-mass bump    msigma_low > utils.BUMP_MSIGMA_LOW_MAX (2.5 Msun), or
                    msigma_low/mp_low > 0.3: the Gaussian stops being a bump
                    and becomes a second continuum across 6-20 Msun, the only
                    window where the CO-IMF index `a` has leverage.  Fisher
                    sigma(a) = 2.55 / 0.73 / 0.29 / 0.15 at true width
                    4 / 3 / 2 / 1.
                    Source: mass-model-audit.md, and
                    notes/2026-08-09-low-mass-bump-width-identifiability.md

5. Truth recovery (only with a pop_config).  Quantile of each truth in its
   marginal.  Reported with an explicit warning that truth quantiles are
   meaningless at low ESS: runs/endO5_evo (bulk ESS ~3) appeared to recover
   18/19 truths and runs/endO5_fullcosmo_evo3 (ESS 22) appeared to badly miss
   one; both readings were artifacts of the chain, not of the model.
"""
import os

# Diagnosis is pure post-processing; keep it off any GPU a running job may want.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import configparser
import glob
import json
import subprocess
import sys
import tempfile
import warnings

sys.path.append("../src")

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import arviz as az

# ---------------------------------------------------------------- constants

SEVERITY_ORDER = {"OK": 0, "NOTE": 1, "WARN": 2, "FAIL": 3}

MC_VARIANCE_BUDGET = 5.0        # intensity_models_fast.pop_cosmo_model default
NEFF_SEL_MULTIPLE = 4           # the model's 4*nobs selection hinge
BUMP_RATIO_MAX = 0.3            # msigma_low/mp_low, mass-model-audit.md sec 9.4

# Narrow mass-feature parameters whose identity is a PE/selection cancellation.
# The Farr neff_sel hinge does not protect these; see the sel-tilt check.
NARROW_FEATURE_PARAMS = ("sigma", "msigma_low", "dmbhmax")
SEL_TILT_NBOOT = 400
SEL_TILT_SEED = 7
# Absolute noise floors (nats on the R-marginalized loglike).  Same scale as
# mc_variance_budget: sqrt(5)~2.2 is the PE-side MC sd the guard tolerates.
SEL_TILT_NOISE_FAIL = 5.0
SEL_TILT_NOISE_WARN = 2.0
# Noise relative to the posterior's lp std (the loglike variation the sampler
# actually explored).  >=1 means selection MC alone can reshape the posterior.
SEL_TILT_RATIO_FAIL = 1.0
SEL_TILT_RATIO_WARN = 0.5

# Float32-vs-float64 potential at one parameter point.  NUTS dual-averaging
# targets 80% accept, i.e. a typical |Delta H| of a few tenths of a nat;
# numpyro only flags a divergence above max_delta_energy = 1000.
# Source: notes/2026-08-07-float32-accuracy-audit.md.
HMC_ENERGY_SCALE = 0.2
F32_DPE_FAIL = 0.2          # 1x the leapfrog energy scale
F32_DPE_WARN = 0.1          # 0.5x
F32_GRAD_COS_FAIL = 1e-6    # ~1 mrad; audit measured 4e-11 at nobs=9000
F32_GRAD_COS_WARN = 1e-8

# Deterministic sites that are diagnostics, not model parameters.
DIAGNOSTIC_SITES = {
    "min_neff", "mc_var_loglike", "neff_sel", "log_mu_sel", "neff",
    "R", "R_unit", "loglik_array_dim",
}
# Deterministic sites derived from sampled ones (run_inf / get_deterministic_parameters).
DERIVED_SITES = {"Om", "kappa", "mbhmax", "flow", "fpl"}
# The bump / tail amplitudes have several interchangeable parametrizations, and
# get_deterministic_parameters records the *other* ones as deterministics that
# still vary (e.g. sampling log_fpeak = log_flow - log(msigma_low) records a
# varying log_flow).  Without a prior file to say which was sampled, the
# sampled coordinate wins and its aliases are treated as derived.
AMPLITUDE_ALIASES = [
    ("log_fpeak", ("log_flow", "logit_flow")),
    ("logit_flow", ("log_flow",)),
    ("logit_fpl", ("log_fpl",)),
]

# Fisher sigma(a) vs the TRUE low-mass bump width, mass-model-audit.md sec 6.
BUMP_WIDTH_SIGMA_A = {4.0: 2.55, 3.0: 0.73, 2.0: 0.29, 1.0: 0.15}

# The a - bump degeneracy block (mass-model-audit.md sec 7.2).
BUMP_BLOCK = {"a", "log_flow", "msigma_low", "mp_low"}


class Finding(object):
    def __init__(self, section, severity, title, detail=None, metrics=None):
        self.section = section
        self.severity = severity
        self.title = title
        self.detail = detail or []
        if isinstance(self.detail, str):
            self.detail = [self.detail]
        self.metrics = metrics or {}

    def as_dict(self):
        return dict(section=self.section, severity=self.severity,
                    title=self.title, detail=self.detail, metrics=self.metrics)


class Report(object):
    def __init__(self):
        self.findings = []
        self.recs = []      # (priority, key, direction, rationale)
        self.metrics = {}

    def add(self, *a, **kw):
        f = Finding(*a, **kw)
        self.findings.append(f)
        return f

    def rec(self, priority, key, direction, rationale):
        self.recs.append(dict(priority=priority, key=key, direction=direction,
                              rationale=rationale))

    def worst(self):
        if not self.findings:
            return "OK"
        return max((f.severity for f in self.findings), key=lambda s: SEVERITY_ORDER[s])

    def deduped_recs(self):
        seen, out = {}, []
        for r in sorted(self.recs, key=lambda r: r["priority"]):
            k = (r["key"], r["direction"])
            if k in seen:
                # keep the highest-priority instance, merge rationales
                if r["rationale"] not in seen[k]["rationale"]:
                    seen[k]["rationale"] += "; " + r["rationale"]
                continue
            seen[k] = dict(r)
            out.append(seen[k])
        return out


# ---------------------------------------------------------------- resolution

def usage_error(msg):
    """Exit 2, not 1: exit 1 is reserved for FAIL-level findings."""
    print("ERROR: " + msg, file=sys.stderr)
    raise SystemExit(2)


def resolve_nc(args):
    """Mirrors plot_corner.py: --run picks the single .nc under ../runs/<run>."""
    if args.nc:
        nc_path = args.nc
        run_dir = os.path.dirname(os.path.abspath(nc_path))
        run_name = args.run or os.path.basename(run_dir)
        return nc_path, run_dir, run_name
    if not args.run:
        usage_error("give --run or --nc")
    run_dir = os.path.join(args.runs_dir, args.run)
    if not os.path.isdir(run_dir):
        usage_error("no such run directory: " + run_dir)
    ncs = sorted(f for f in os.listdir(run_dir) if f.endswith(".nc"))
    if len(ncs) != 1:
        usage_error("expected exactly one .nc in %s, found %s -- pass "
                    "--nc explicitly" % (run_dir, ncs))
    return os.path.join(run_dir, ncs[0]), run_dir, args.run


def find_ini(run_dir, run_name, run_configs_dir):
    """An .ini copied into the run dir wins; otherwise the run_configs .ini
    whose [run] run_dir matches this run."""
    local = sorted(glob.glob(os.path.join(run_dir, "*.ini")))
    if local:
        return local[0]
    for path in sorted(glob.glob(os.path.join(run_configs_dir, "*.ini"))):
        cfg = configparser.ConfigParser()
        try:
            cfg.read(path)
            if cfg.has_section("run") and cfg["run"].get("run_dir") == run_name:
                return path
        except configparser.Error:
            continue
    return None


def read_ini(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if not cfg.has_section("run"):
        return {}
    run = cfg["run"]

    def _int(k, d=None):
        v = run.get(k)
        if v is None or str(v).lower() == "none":
            return d
        try:
            return int(v)
        except ValueError:
            return d

    out = dict(
        prior=run.get("prior"),
        pop_config_file=run.get("pop_config_file"),
        output_sel_file=run.get("output_sel_file"),
        output_file_PE=run.get("output_file_PE"),
        nmcmc=_int("nmcmc"), nchain=_int("nchain"),
        n_pe=_int("n_pe"),
        evt_start=_int("evt_start", 0), evt_end=_int("evt_end"),
        max_tree_depth=_int("max_tree_depth", 7),
        max_tree_depth_explicit=run.get("max_tree_depth") is not None,
        dense_mass=run.getboolean("dense_mass", fallback=False),
        target_accept_prob=run.getfloat("target_accept_prob", fallback=0.8),
        use_low_bump=run.getboolean("use_low_bump", fallback=True),
        smooth_tail_edge=run.getboolean("smooth_tail_edge", fallback=True),
    )
    if out["pop_config_file"] and str(out["pop_config_file"]).lower() == "none":
        out["pop_config_file"] = None
    return out


def load_truths(path):
    """Same derived-parameter mapping as plot_corner.load_truths /
    run_inf.load_true_vals."""
    tv = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                try:
                    tv[k.strip()] = float(v.strip())
                except ValueError:
                    continue
    if "kappa" in tv and "lam" in tv:
        tv["dkappa"] = tv["kappa"] - tv["lam"]
    if "mbhmax" in tv and "mpisn" in tv:
        tv["dmbhmax"] = tv["mbhmax"] - tv["mpisn"]
    if "fpl" in tv:
        tv["log_fpl"] = np.log(tv["fpl"])
    tv["log_flow"] = np.log(tv["flow"]) if "flow" in tv else np.log(1e-5)
    if "Omh2" not in tv and "Om" in tv and "h" in tv:
        tv["Omh2"] = tv["Om"] * tv["h"] ** 2
    return tv


# ---------------------------------------------------------------- prior info

def prior_summary(prior, nsample=60000, seed=0):
    """sd / bounds for every distribution in a parsed prior dict.

    numpyro distributions have no uniform analytic .variance across the
    TruncatedNormal / Uniform / Normal mix used here, so sample them.  60k
    draws give the sd to <1%, far finer than any threshold below.
    """
    import jax
    key = jax.random.PRNGKey(seed)
    out = {}
    for name, dd in prior.items():
        if isinstance(dd, float):
            out[name] = dict(fixed=True, value=float(dd))
            continue
        try:
            key, sub = jax.random.split(key)
            s = np.asarray(dd.sample(sub, (nsample,)), dtype=float)
        except Exception:
            out[name] = dict(fixed=False, sd=None, low=None, high=None)
            continue
        sup = getattr(dd, "support", None)
        low = getattr(sup, "lower_bound", None)
        high = getattr(sup, "upper_bound", None)
        low = float(low) if low is not None and np.isfinite(low) else None
        high = float(high) if high is not None and np.isfinite(high) else None
        sd = float(np.std(s))
        # Effective range used for the wall epsilon: the true support when it
        # is two-sided, else +-2 sd around the mean of the prior.
        if low is not None and high is not None:
            rng = high - low
        else:
            rng = 4.0 * sd
        out[name] = dict(fixed=False, sd=sd, low=low, high=high, range=rng,
                         mean=float(np.mean(s)))
    return out


# ---------------------------------------------------------------- sections

def section_inventory(rep, nc_path, idata, ini_path, ini, prior_path, pop_path, nobs):
    post = idata.posterior
    nchain = post.sizes["chain"]
    ndraw = post.sizes["draw"]
    rep.metrics.update(nc=nc_path, nchain=int(nchain), ndraw=int(ndraw),
                       ndraw_total=int(nchain * ndraw))
    lines = ["netcdf      : %s" % nc_path,
             "chains x draws: %d x %d = %d draws" % (nchain, ndraw, nchain * ndraw),
             "config      : %s" % (ini_path or "NOT FOUND"),
             "prior       : %s" % (prior_path or "NOT FOUND"),
             "pop_config  : %s" % (pop_path or "NOT FOUND")]
    if nobs is None:
        lines.append("nobs        : UNKNOWN (no evt_start/evt_end); the 4*nobs "
                     "selection criterion cannot be evaluated")
    else:
        lines.append("nobs        : %d (evt_end - evt_start)" % nobs)
    if ini:
        lines.append("sampler     : max_tree_depth=%s%s dense_mass=%s "
                     "target_accept_prob=%s nmcmc=%s nchain=%s n_pe=%s"
                     % (ini.get("max_tree_depth"),
                        "" if ini.get("max_tree_depth_explicit") else " (default)",
                        ini.get("dense_mass"), ini.get("target_accept_prob"),
                        ini.get("nmcmc"), ini.get("nchain"), ini.get("n_pe")))
    for k in ("recentering_offset", "log_pdraw_sel_scale"):
        if k in post.attrs:
            lines.append("attr %-18s= %.6g" % (k, float(post.attrs[k])))
    if "recentering_offset" in post.attrs:
        lines.append("float32 recentering was active: recorded lp/energy are "
                     "shifted by recentering_offset (add it back for absolute "
                     "log-posterior values).  R and log_mu_sel are already in "
                     "the physical convention.")
    else:
        lines.append("no recentering_offset attr: this run predates the float32 "
                     "recentering, or used the unoptimized module.")
    pe_name = (ini or {}).get("output_file_PE")
    if pe_name:
        pe_path = os.path.join(os.path.dirname(nc_path), pe_name)
        lines.append("PE hdf5     : %s%s" % (
            pe_name, "" if os.path.exists(pe_path) else " (NOT FOUND)"))
    rep.add("inventory", "OK", "run inventory", lines)


def free_parameters(post, prior):
    """Sampled scalar parameters with genuine variation.

    The constant test is an exact range test, never a std threshold: a
    parameter pinned to a single float32 value can still report std ~1e-7 from
    roundoff (plot_corner.py learned this the hard way when a constant column
    aborted a whole corner figure).
    """
    derived = set(DERIVED_SITES)
    if prior is None:
        for sampled, aliases in AMPLITUDE_ALIASES:
            if sampled in post and np.ptp(np.asarray(post[sampled].values)) > 0:
                derived.update(aliases)
    names = []
    for k in sorted(post.data_vars):
        da = post[k]
        if set(da.dims) != {"chain", "draw"}:
            continue                      # vector site (hz, dNdVdt_..., neff)
        if k in DIAGNOSTIC_SITES or k in derived:
            continue
        v = np.asarray(da.values)
        if np.ptp(v) == 0:                # held fixed in this run
            continue
        if prior is not None:
            if k not in prior or isinstance(prior[k], float):
                continue                  # prior says fixed -> not a free param
        names.append(k)
    return names


def section_convergence(rep, idata, free, ini):
    post, ss = idata.posterior, idata.sample_stats
    summ = az.summary(post, var_names=free, round_to=None)
    worst_ess = summ.sort_values("ess_bulk")
    min_ess = float(worst_ess["ess_bulk"].iloc[0])
    min_ess_tail = float(summ["ess_tail"].min())
    max_rhat = float(summ["r_hat"].max())
    max_rhat_p = summ["r_hat"].idxmax()
    rep.metrics.update(min_ess_bulk=min_ess, min_ess_tail=min_ess_tail,
                       max_rhat=max_rhat, max_rhat_param=str(max_rhat_p))

    if max_rhat <= 1.01:
        sev = "OK"
    elif max_rhat <= 1.05:
        sev = "NOTE"
    elif max_rhat <= 1.10:
        sev = "WARN"
    else:
        sev = "FAIL"
    bad = summ[summ["r_hat"] > 1.01].sort_values("r_hat", ascending=False)
    detail = ["max r-hat = %.3f (%s)" % (max_rhat, max_rhat_p)]
    if len(bad):
        detail.append("above 1.01: " + ", ".join(
            "%s %.3f" % (i, r.r_hat) for i, r in bad.head(8).iterrows()))
    else:
        detail.append("every free parameter has r-hat <= 1.01")
    rep.add("convergence", sev, "between-chain agreement (r-hat)", detail,
            dict(max_rhat=max_rhat, param=str(max_rhat_p)))

    if min_ess >= 200:
        sev = "OK"
    elif min_ess >= 100:
        sev = "NOTE"
    elif min_ess >= 50:
        sev = "WARN"
    else:
        sev = "FAIL"
    detail = ["min bulk ESS = %.0f, min tail ESS = %.0f (of %d draws)"
              % (min_ess, min_ess_tail, post.sizes["chain"] * post.sizes["draw"])]
    detail.append("worst offenders (bulk ESS, r-hat): " + ", ".join(
        "%s %.0f/%.3f" % (i, r.ess_bulk, r.r_hat)
        for i, r in worst_ess.head(6).iterrows()))
    if min_ess < 100:
        detail.append("At bulk ESS %.0f the Monte-Carlo error on any quoted "
                      "interval is >= %.0f%% of its width; medians and 68%% "
                      "intervals for the worst parameters should not be "
                      "quoted." % (min_ess, 100.0 / np.sqrt(max(min_ess, 1))))
    rep.add("convergence", sev, "effective sample size", detail,
            dict(min_ess_bulk=min_ess, min_ess_tail=min_ess_tail))

    # divergences
    ndiv = int(np.asarray(ss["diverging"].values).sum()) if "diverging" in ss else None
    ntot = int(post.sizes["chain"] * post.sizes["draw"])
    if ndiv is None:
        rep.add("convergence", "NOTE", "divergences",
                "insufficient information: no 'diverging' in sample_stats")
    else:
        frac = ndiv / float(ntot)
        rep.metrics["divergences"] = ndiv
        if ndiv == 0:
            sev = "OK"
        elif frac < 0.002:
            sev = "NOTE"
        elif frac < 0.01:
            sev = "WARN"
        else:
            sev = "FAIL"
        d = ["%d divergent transitions out of %d (%.2f%%)" % (ndiv, ntot, 100 * frac)]
        if ndiv:
            d.append("Divergences mean the trajectories hit a region the "
                     "integrator could not resolve; the posterior is biased "
                     "away from it.  Non-smooth gradients are one cause "
                     "(the legacy neff_penalty='min_neff' guard had kinked "
                     "boundaries; the mc_variance default does not), a "
                     "mismatch between numerator and denominator densities is "
                     "another (notes/2026-08-08-tabulated-selection-"
                     "consistency.md: 14 divergences on runs/endO5_evo).")
        rep.add("convergence", sev, "divergences", d, dict(divergences=ndiv))

    # tree depth / step size
    depth = None
    if "tree_depth" in ss:
        depth = np.asarray(ss["tree_depth"].values)
    elif "n_steps" in ss:
        n_steps = np.asarray(ss["n_steps"].values)
        depth = np.floor(np.log2(np.maximum(n_steps, 1))).astype(int) + 1
    if depth is None:
        rep.add("convergence", "NOTE", "tree depth / step size",
                "insufficient information: this .nc has no n_steps/tree_depth. "
                "The run predates run_inf.py's extra_fields=(...), so tree "
                "depth, energy, acceptance and step size were never recorded. "
                "Rerun with the current run_inf.py to get them.")
        rep.metrics["depth_cap_frac"] = None
    else:
        cap_obs = int(depth.max())
        cap_cfg = ini.get("max_tree_depth") if ini else None
        cap = cap_cfg if cap_cfg is not None else cap_obs
        frac = float((depth >= cap).mean())
        rep.metrics.update(depth_cap=int(cap), depth_cap_observed=cap_obs,
                           depth_cap_frac=frac)
        d = ["max_tree_depth in use = %d (%s); max n_steps = %d = 2^%d - 1"
             % (cap, "from config" if cap_cfg is not None else
                "inferred from max(n_steps)", int(2 ** cap_obs - 1), cap_obs)]
        d.append("%.1f%% of iterations reached the depth cap" % (100 * frac))
        if cap_cfg is not None and cap_obs != cap_cfg:
            d.append("the cap was never reached (deepest observed %d < config %d)"
                     % (cap_obs, cap_cfg))
        if frac > 0.9:
            sev = "NOTE"
        elif frac > 0.5:
            sev = "NOTE"
        else:
            sev = "OK"
        rep.add("convergence", sev, "tree-depth saturation", d,
                dict(depth_cap=int(cap), frac_at_cap=frac))

    if "step_size" in ss:
        sscol = np.asarray(ss["step_size"].values)
        adapted = sscol[:, -1]
        rep.metrics["step_size"] = [float(x) for x in adapted]
        rep.add("convergence", "OK", "adapted step size",
                "per chain: " + ", ".join("%.4g" % x for x in adapted),
                dict(step_size=[float(x) for x in adapted]))

    if "acceptance_rate" in ss:
        acc = float(np.asarray(ss["acceptance_rate"].values).mean())
        tgt = (ini or {}).get("target_accept_prob", 0.8)
        rep.metrics["acceptance_rate"] = acc
        sev = "NOTE" if abs(acc - tgt) > 0.1 else "OK"
        rep.add("convergence", sev, "acceptance rate",
                "mean acceptance %.3f against target_accept_prob %.2f" % (acc, tgt),
                dict(acceptance_rate=acc, target=tgt))

    if "energy" in ss:
        try:
            bfmi = np.asarray(az.bfmi(idata), dtype=float)
            rep.metrics["bfmi"] = [float(x) for x in bfmi]
            sev = "WARN" if np.any(bfmi < 0.3) else "OK"
            d = ["BFMI per chain: " + ", ".join("%.3f" % x for x in bfmi)]
            if sev == "WARN":
                d.append("BFMI below 0.3 means the momentum resampling is not "
                         "exploring the energy distribution: heavy tails or a "
                         "funnel.  More depth will not fix it; reparameterize.")
            rep.add("convergence", sev, "energy / BFMI", d, dict(bfmi=list(map(float, bfmi))))
        except Exception as exc:
            rep.add("convergence", "NOTE", "energy / BFMI",
                    "could not compute BFMI: %s" % exc)
    else:
        rep.add("convergence", "NOTE", "energy / BFMI",
                "insufficient information: no 'energy' in sample_stats")

    if "lp" in ss:
        lp = np.asarray(ss["lp"].values)
        means = lp.mean(axis=1)
        pooled = float(lp.std())
        spread = float(means.max() - means.min())
        try:
            lp_rhat = float(az.rhat(ss[["lp"]])["lp"])
        except Exception:
            lp_rhat = float("nan")
        rep.metrics.update(lp_chain_means=[float(x) for x in means],
                           lp_rhat=lp_rhat)
        sev = "WARN" if (lp_rhat == lp_rhat and lp_rhat > 1.05) else "OK"
        d = ["per-chain lp means: " + ", ".join("%.2f" % x for x in means)
             + "  (pooled sd %.2f, spread %.2f = %.2f sd)"
             % (pooled, spread, spread / pooled if pooled else float("nan")),
             "lp r-hat = %.3f" % lp_rhat]
        if sev == "WARN":
            d.append("The chains are not sampling the same log-posterior level: "
                     "they are in different modes, or at least one has not "
                     "reached the typical set.  Parameter medians pooled across "
                     "chains are not meaningful in this state.")
        if "recentering_offset" in idata.posterior.attrs:
            d.append("(lp is shifted by recentering_offset; only differences "
                     "and the r-hat are meaningful here.)")
        rep.add("convergence", sev, "per-chain log-posterior", d,
                dict(lp_rhat=lp_rhat, lp_chain_means=[float(x) for x in means]))
    else:
        rep.add("convergence", "NOTE", "per-chain log-posterior",
                "insufficient information: no 'lp' in sample_stats")

    return summ, min_ess, max_rhat


def section_monte_carlo(rep, idata, nobs, ini):
    post = idata.posterior
    n_pe = (ini or {}).get("n_pe")

    # ---- per-event PE samples
    if "mc_var_loglike" in post:
        mc = np.asarray(post["mc_var_loglike"].values).ravel()
        med, hi = float(np.median(mc)), float(np.max(mc))
        ratio = med / MC_VARIANCE_BUDGET
        rep.metrics.update(mc_var_median=med, mc_var_max=hi,
                           mc_variance_budget=MC_VARIANCE_BUDGET)
        d = ["mc_var_loglike (= sum_i 1/n_eff_i) median %.2f, max %.2f, "
             "against mc_variance_budget %.1f" % (med, hi, MC_VARIANCE_BUDGET),
             "MC sd of the total log likelihood = sqrt(mc_var) = %.2f nats "
             "(median)" % np.sqrt(med)]
        if ratio >= 1.0:
            sev = "FAIL"
            d.append("The posterior sits at or above the budget: the softplus "
                     "guard is SHAPING the posterior, not merely railing it.")
            rep.rec(1, "n_pe", "increase",
                    "mc_var_loglike %.2f is at/above the %.1f budget; variance "
                    "falls ~1/n_pe, so n_pe needs about %.1fx"
                    % (med, MC_VARIANCE_BUDGET, med / (0.6 * MC_VARIANCE_BUDGET)))
        elif ratio >= 0.8:
            sev = "WARN"
            d.append("Within 20% of the budget; the guard is close to active "
                     "and any sharpening of the population would push it over.")
            rep.rec(2, "n_pe", "increase",
                    "mc_var_loglike %.2f is within 20%% of the %.1f budget"
                    % (med, MC_VARIANCE_BUDGET))
        elif med < 1.0:
            sev = "NOTE"
            d.append("Far below the budget: the PE sample count was stronger "
                     "than needed.")
            if n_pe:
                safe = int(n_pe * med / (0.6 * MC_VARIANCE_BUDGET))
                d.append("n_pe could drop to roughly %d (cost is ~linear in "
                         "n_pe, variance ~1/n_pe) and still leave 40%% margin."
                         % max(safe, 1))
                rep.rec(4, "n_pe", "decrease",
                        "mc_var_loglike %.2f uses only %.0f%% of the budget; "
                        "n_pe ~%d would still leave margin"
                        % (med, 100 * ratio, max(safe, 1)))
            else:
                rep.rec(4, "n_pe", "decrease",
                        "mc_var_loglike %.2f uses only %.0f%% of the budget"
                        % (med, 100 * ratio))
        else:
            sev = "OK"
            d.append("The guard was inactive over the whole posterior: this is "
                     "the posterior you would have obtained with no penalty. "
                     "Reducing n_pe is not free -- variance scales ~1/n_pe, so "
                     "halving n_pe would put mc_var at ~%.1f." % (2 * med))
        rep.add("monte-carlo", sev, "per-event PE Monte-Carlo variance", d,
                dict(mc_var_median=med, mc_var_max=hi, budget=MC_VARIANCE_BUDGET))
    else:
        rep.add("monte-carlo", "NOTE", "per-event PE Monte-Carlo variance",
                "insufficient information: no mc_var_loglike site (run predates "
                "the mc_variance guard)")

    if "min_neff" in post:
        mn = np.asarray(post["min_neff"].values).ravel()
        lo, med = float(mn.min()), float(np.median(mn))
        rep.metrics.update(min_neff_min=lo, min_neff_median=med)
        if lo < 2:
            sev = "WARN"
        elif lo < 5:
            sev = "NOTE"
        else:
            sev = "OK"
        d = ["min_neff over the posterior: min %.2f, median %.2f" % (lo, med)]
        if lo < 5:
            d.append("Some single events are estimated from only a handful of "
                     "PE samples.  That is tolerable while the total variance "
                     "(mc_var_loglike) is in budget, but worth a look.")
        d.append("The legacy neff_penalty='min_neff' guard demanded "
                 "min_neff >= nobs, which at these values would have been "
                 "permanently active; the mc_variance default replaces it.")
        rep.add("monte-carlo", sev, "worst per-event n_eff", d,
                dict(min_neff_min=lo, min_neff_median=med))

    # ---- selection samples
    if "neff_sel" not in post:
        rep.add("monte-carlo", "NOTE", "selection Monte-Carlo",
                "insufficient information: no neff_sel site")
        return
    ns = np.asarray(post["neff_sel"].values).ravel()
    lo, med = float(ns.min()), float(np.median(ns))
    rep.metrics.update(neff_sel_min=lo, neff_sel_median=med)
    if nobs is None:
        rep.add("monte-carlo", "NOTE", "selection Monte-Carlo",
                ["neff_sel median %.4g, min %.4g" % (med, lo),
                 "nobs is unknown (no config), so the model's 4*nobs criterion "
                 "cannot be evaluated.  Supply --pop_config's run .ini or state "
                 "nobs to complete this check."],
                dict(neff_sel_min=lo, neff_sel_median=med))
        return
    hinge = NEFF_SEL_MULTIPLE * nobs
    ratio = lo / float(hinge)
    rep.metrics.update(neff_sel_hinge=hinge, neff_sel_ratio=ratio)
    d = ["neff_sel median %.4g, min %.4g against the model hinge 4*nobs = %d"
         % (med, lo, hinge),
         "headroom ratio min(neff_sel)/(4*nobs) = %.2fx" % ratio]
    if ratio < 1.0:
        sev = "FAIL"
        d.append("The selection guard is BELOW threshold, so the softplus "
                 "penalty is active everywhere and the posterior is being "
                 "penalized over its whole support -- not railed, shaped.")
        d.append("neff_sel can never exceed nsel, so this configuration needs "
                 "at least %d selection rows before the guard is even "
                 "satisfiable.  (Precedent: a 17,624-row selection set against "
                 "a 36,000 hinge was unusable by construction and that run was "
                 "cancelled; see notes/2026-08-09-low-mass-bump-width-"
                 "identifiability.md.)" % hinge)
        rep.rec(1, "selection samples (nsel) / injections", "increase",
                "neff_sel %.4g is below the 4*nobs = %d hinge, so the selection "
                "guard is permanently active" % (lo, hinge))
        rep.rec(2, "evt_end", "decrease",
                "alternatively lower nobs: the hinge is 4*nobs, so nobs <= "
                "%d would be satisfiable with this selection set" % int(lo / 4))
    elif ratio < 1.5:
        sev = "WARN"
        d.append("Less than 50% margin: a slightly sharper population, or more "
                 "events, would activate the guard.")
        rep.rec(3, "selection samples (nsel) / injections", "increase",
                "only %.2fx margin on the 4*nobs selection hinge" % ratio)
    elif ratio > 3.0:
        sev = "NOTE"
        d.append("More than 3x margin: the selection set is larger than this "
                 "run needs.  It could be reduced (or, better, the same "
                 "injections could support a larger nobs, since the hinge is "
                 "4*nobs -- up to nobs ~ %d at this neff_sel)." % int(lo / 4 / 1.5))
    else:
        sev = "OK"
        d.append("Comfortable margin on the Farr-2019 criterion.")
    rep.add("monte-carlo", sev, "selection-integral n_eff", d,
            dict(neff_sel_min=lo, neff_sel_median=med, hinge=hinge, ratio=ratio))


# ---------------------------------------------------------------- selection tilt

def _posterior_median_dict(post):
    """Median of every scalar posterior variable, as plain Python floats.

    Grid-valued deterministics (hz, mdNdmdVdt_..., shape (chain, draw, n_grid))
    are skipped; only sites whose trailing shape is () or (1,) are kept.
    """
    out = {}
    for name in post.data_vars:
        v = np.asarray(post[name].values)
        # Expect at least (chain, draw); anything with extra size > 1 is a grid.
        if v.ndim > 2 and int(np.prod(v.shape[2:])) > 1:
            continue
        flat = v.ravel()
        if flat.size == 0:
            continue
        out[name] = float(np.median(flat))
    return out


def _canonical_pop_sample(base, post_med):
    """Build the canonical dict expected by build_population_model / FlatwCDM.

    Prefer ``base`` (truths) for every key it has; fill gaps from posterior
    medians; derive the usual transformed names.
    """
    s = dict(base or {})
    for k, v in (post_med or {}).items():
        s.setdefault(k, v)

    # Pivot reparam: mpisn_ref at zpivot -> mpisn at z=0.
    if "mpisn" not in s and "mpisn_ref" in s:
        zp = float(s.get("zpivot", 0.75))
        mdot = float(s.get("mpisndot", 0.0))
        s["mpisn"] = float(s["mpisn_ref"]) - mdot * zp / (1.0 + zp)
    if "mbhmax" not in s and "mpisn" in s and "dmbhmax" in s:
        s["mbhmax"] = float(s["mpisn"]) + float(s["dmbhmax"])
    if "kappa" not in s and "lam" in s and "dkappa" in s:
        s["kappa"] = float(s["lam"]) + float(s["dkappa"])
    if "fpl" not in s and "log_fpl" in s:
        s["fpl"] = float(np.exp(s["log_fpl"]))
    if "flow" not in s:
        if "log_flow" in s:
            s["flow"] = float(np.exp(s["log_flow"]))
        elif "log_fpeak" in s and "msigma_low" in s:
            s["flow"] = float(np.exp(s["log_fpeak"]) * s["msigma_low"])
    if "Om" not in s and "Omh2" in s and "h" in s:
        s["Om"] = float(s["Omh2"]) / float(s["h"]) ** 2
    return s


def _sel_log_wts(sel, sample, use_low_bump=True):
    """Per-row selection log-weights matching pop_cosmo_model's selection path."""
    import jax.numpy as jnp
    import intensity_models_fast as im

    required = ("a", "b", "c", "mpisn", "mpisndot", "mbhmax", "sigma", "fpl",
                "beta", "lam", "kappa", "zp", "zmax", "mbh_min", "delta_m",
                "h", "Om", "w")
    missing = [k for k in required if k not in sample]
    if missing:
        raise KeyError("incomplete population sample, missing %s" % missing)

    pop = {k: sample[k] for k in (
        "a", "b", "c", "mpisn", "mpisndot", "mbhmax", "sigma", "fpl", "beta",
        "lam", "kappa", "zp", "zmax", "mbh_min", "delta_m")}
    pop["mp_low"] = sample.get("mp_low", 1.0)
    pop["msigma_low"] = sample.get("msigma_low", 1.0)
    pop["flow"] = sample.get("flow", 0.0)
    pop["mco_min"] = sample.get("mco_min", 4.0)
    pop["mco_floor"] = sample.get("mco_floor", 6.0)

    cosmo = im.FlatwCDMCosmology(sample["h"], sample["Om"], sample["w"],
                                 zmax=sample["zmax"])
    log_dN = im.build_population_model(pop, use_low_bump=use_low_bump, n_z=30,
                                       smooth_tail_edge=True)
    m1d = jnp.asarray(sel["m1d"].values)
    q = jnp.asarray(sel["q"].values)
    dl = jnp.asarray(sel["dl"].values)
    log1p_z, J = cosmo.z_and_log_jacobian(jnp.log(dl))
    opz = jnp.exp(log1p_z)
    m1 = m1d / opz
    lw = (log_dN.call_from_logs(m1, jnp.log(m1), jnp.log(q), opz - 1.0, log1p_z)
          - jnp.log(jnp.asarray(sel["pdraw_sel"].values)) + J)
    return np.asarray(lw, dtype=np.float64)


def _bootstrap_delta_log_mu(lw_lo, lw_hi, nboot=SEL_TILT_NBOOT, seed=SEL_TILT_SEED):
    """Delta log_mu_sel = logsumexp(lw_hi) - logsumexp(lw_lo), plus bootstrap sd."""
    m = max(float(np.max(lw_lo)), float(np.max(lw_hi)))
    w_lo = np.exp(lw_lo - m)
    w_hi = np.exp(lw_hi - m)
    delta = float(np.log(w_hi.sum()) - np.log(w_lo.sum()))
    n = w_lo.size
    rng = np.random.default_rng(seed)
    boots = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        boots[i] = np.log(w_hi[idx].sum()) - np.log(w_lo[idx].sum())
    return delta, float(boots.std(ddof=1))


def resolve_sel_file(run_dir, ini):
    """Locate the selection HDF5 used by the run, if it is still on disk."""
    name = (ini or {}).get("output_sel_file")
    if not name or not run_dir:
        return None
    cand = os.path.join(run_dir, name)
    return cand if os.path.exists(cand) else None


def resolve_pe_file(run_dir, ini):
    """Locate the PE HDF5 used by the run, if it is still on disk."""
    name = (ini or {}).get("output_file_PE")
    if not name or not run_dir:
        return None
    cand = os.path.join(run_dir, name)
    return cand if os.path.exists(cand) else None


def _sel_tilt_severity(noise, ratio):
    if noise >= SEL_TILT_NOISE_FAIL or (
            ratio is not None and ratio >= SEL_TILT_RATIO_FAIL):
        return "FAIL"
    if noise >= SEL_TILT_NOISE_WARN or (
            ratio is not None and ratio >= SEL_TILT_RATIO_WARN):
        return "WARN"
    if noise >= 1.0:
        return "NOTE"
    return "OK"


def section_selection_tilt(rep, idata, free, truths, nobs, sel_file, ini):
    """Bootstrap selection-tilt MC noise for narrow mass-feature parameters.

    The Farr neff_sel >= 4*nobs hinge only bounds the *global* normalization
    error of log_mu_sel.  For a sharp feature the R-marginalized likelihood is
    a near-cancellation between the PE numerator and the selection term, so a
    few nats of *parameter-dependent* selection noise can move the posterior
    even when the hinge has comfortable headroom.  Calibrated on
    endO5_fullcosmo_evo7 (sigma truth below the posterior; hinge 2.0x OK).
    """
    targets = [p for p in NARROW_FEATURE_PARAMS if p in free]
    if not targets:
        rep.add("monte-carlo", "OK", "narrow-feature selection tilt",
                "no free narrow-feature parameters (%s); check skipped"
                % ", ".join(NARROW_FEATURE_PARAMS))
        return
    if nobs is None:
        rep.add("monte-carlo", "NOTE", "narrow-feature selection tilt",
                "insufficient information: nobs unknown (no config), so "
                "nobs*sd(Delta log_mu_sel) cannot be formed")
        return
    if not sel_file:
        rep.add("monte-carlo", "NOTE", "narrow-feature selection tilt",
                ["insufficient information: selection HDF5 not found",
                 "Need the run's output_sel_file next to the .nc (or a matching "
                 "run_configs/*.ini that names it).  Without it the Farr "
                 "neff_sel hinge is the only selection check, and that hinge "
                 "missed the sigma failure on endO5_fullcosmo_evo7."])
        return

    try:
        import pandas as pd
        sel_all = pd.read_hdf(sel_file, key="true_parameters")
    except Exception as exc:
        rep.add("monte-carlo", "NOTE", "narrow-feature selection tilt",
                "insufficient information: could not read %s (%s)"
                % (sel_file, exc))
        return

    sel = sel_all
    n_use = len(sel)
    use_low_bump = (ini or {}).get("use_low_bump", True)

    post = idata.posterior
    post_med = _posterior_median_dict(post)
    base = _canonical_pop_sample(truths, post_med)
    try:
        # Probe once so a missing key fails with a clean NOTE, not mid-loop.
        _ = _sel_log_wts(sel.iloc[:2], base, use_low_bump=use_low_bump)
    except Exception as exc:
        rep.add("monte-carlo", "NOTE", "narrow-feature selection tilt",
                "insufficient information: could not rebuild selection "
                "weights from truths/posterior (%s)" % exc)
        return

    lp = None
    if hasattr(idata, "sample_stats") and "lp" in idata.sample_stats:
        lp = np.asarray(idata.sample_stats["lp"].values).ravel()
    lp_std = float(np.std(lp)) if lp is not None and lp.size else None

    rows = []
    worst_sev = "OK"
    for name in targets:
        vals = np.asarray(post[name].values).ravel()
        lo, hi = float(np.percentile(vals, 16)), float(np.percentile(vals, 84))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        sample_lo = dict(base); sample_lo[name] = lo
        sample_hi = dict(base); sample_hi[name] = hi
        # Keep the hard edge consistent with whichever of mpisn / dmbhmax moved.
        for s in (sample_lo, sample_hi):
            if "mpisn" in s and "dmbhmax" in s:
                s["mbhmax"] = float(s["mpisn"]) + float(s["dmbhmax"])
        try:
            lw_lo = _sel_log_wts(sel, sample_lo, use_low_bump=use_low_bump)
            lw_hi = _sel_log_wts(sel, sample_hi, use_low_bump=use_low_bump)
            delta, sd = _bootstrap_delta_log_mu(lw_lo, lw_hi)
        except Exception as exc:
            rep.add("monte-carlo", "NOTE", "narrow-feature selection tilt",
                    "failed evaluating %s tilt (%s); remaining params skipped"
                    % (name, exc))
            return
        noise = nobs * sd
        tilt = nobs * abs(delta)          # |selection contribution| to loglike
        ratio = (noise / lp_std) if (lp_std and lp_std > 0) else None
        rows.append(dict(name=name, lo=lo, hi=hi, delta=delta, sd=sd,
                         noise=noise, tilt=tilt, ratio=ratio))

    if not rows:
        rep.add("monte-carlo", "NOTE", "narrow-feature selection tilt",
                "no usable posterior range for %s" % ", ".join(targets))
        return

    d = ["selection set: %d rows from %s"
         % (n_use, os.path.basename(sel_file)),
         "for each free narrow-feature param, bootstrap nobs*sd(Delta "
         "log_mu_sel) across the posterior 16-84%% range with other params "
         "at %s; compare that noise to the posterior lp std "
         "(log-likelihood variation the sampler saw)"
         % ("truth" if truths else "posterior median")]
    if lp_std is not None:
        d.append("posterior lp std = %.2f nats" % lp_std)
    d.append("per-parameter results (noise = nobs*bootstrap_sd, "
             "tilt = |nobs*Delta log_mu_sel|):")

    worst_row = None
    for r in rows:
        ratio_s = ("%.2f" % r["ratio"]) if r["ratio"] is not None else "n/a"
        d.append("  %s in [%.4g, %.4g]: noise %.2f nats, tilt %.2f nats, "
                 "noise/lp_std %s"
                 % (r["name"], r["lo"], r["hi"], r["noise"], r["tilt"], ratio_s))
        sev = _sel_tilt_severity(r["noise"], r["ratio"])
        if (SEVERITY_ORDER[sev] > SEVERITY_ORDER[worst_sev] or
                (sev == worst_sev and
                 (worst_row is None or r["noise"] > worst_row["noise"]))):
            worst_sev = sev
            worst_row = r

    flagged = [r for r in rows
               if _sel_tilt_severity(r["noise"], r["ratio"]) in ("FAIL", "WARN")]

    if worst_sev == "FAIL":
        d.append("Selection MC noise across the posterior range is as large as "
                 "(or larger than) the log-likelihood variation the sampler "
                 "explored.  The Farr neff_sel hinge can still look fine: it "
                 "bounds only the global normalization, not a sharp-feature "
                 "tilt.  Precedent: endO5_fullcosmo_evo7 recovered a "
                 "prior-dominated sigma with truth at quantile 0.001 while "
                 "neff_sel sat at 2.0x the hinge.")
        rep.rec(1, "selection samples (nsel) / injection pool", "increase",
                "narrow-feature selection tilt noise %.2f nats on %s "
                "(noise/lp_std %s); grow the selection set or draw it from "
                "the proposal pool without truth-rejection so the PE/selection "
                "cancellation is not noise-dominated"
                % (worst_row["noise"], worst_row["name"],
                   ("%.2f" % worst_row["ratio"]) if worst_row["ratio"] is not None
                   else "n/a"))
    elif worst_sev == "WARN":
        d.append("Selection MC noise is a substantial fraction of the "
                 "log-likelihood variation.  Narrow-feature medians "
                 "(%s) may be shifted; quote with caution or grow nsel."
                 % ", ".join(r["name"] for r in flagged))
        rep.rec(2, "selection samples (nsel) / injection pool", "increase",
                "narrow-feature selection tilt noise up to %.2f nats"
                % max(r["noise"] for r in rows))
    elif worst_sev == "NOTE":
        d.append("Detectable selection-tilt noise, but below the level that "
                 "reshaped endO5_fullcosmo_evo7.  Worth watching if a "
                 "narrow-feature truth sits in the posterior tail.")
    else:
        d.append("Selection-tilt MC noise is small compared with the "
                 "log-likelihood variation; the Farr hinge is not hiding a "
                 "sharp-feature failure of the evo7 kind.")

    # Name every WARN/FAIL param, not only the worst -- on evo7 sigma is the
    # scientifically interesting failure even when dmbhmax has larger noise.
    for r in flagged:
        rep.rec(2, r["name"], "do not quote",
                "selection-tilt MC noise %.2f nats (noise/lp_std %s) can move "
                "this narrow-feature marginal; treat as prior- or noise-"
                "dominated until nsel is raised"
                % (r["noise"],
                   ("%.2f" % r["ratio"]) if r["ratio"] is not None else "n/a"))

    rep.metrics["sel_tilt"] = rows
    if lp_std is not None:
        rep.metrics["lp_std"] = lp_std
    rep.add("monte-carlo", worst_sev, "narrow-feature selection tilt", d,
            dict(n_sel_use=n_use, lp_std=lp_std,
                 worst_noise=max(r["noise"] for r in rows),
                 params={r["name"]: r for r in rows}))


# ---------------------------------------------------------------- float32 vs float64

def _plain_floats(d):
    """JSON-safe {name: float} from a mixed dict of numpy / python scalars."""
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out[k] = fv
    return out


def _round_to_f32(data):
    """Round every input array to float32 and store it back as float64.

    Both precision legs then see numerically identical inputs, so the
    comparison isolates arithmetic precision rather than input precision.
    Same helper as scripts/testing_scripts/test_float32_accuracy.py.
    """
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = np.asarray(np.asarray(v, np.float32), np.float64)
        else:
            out[k] = float(np.float32(v))
    return out


def _load_pe_sel_arrays(pe_file, sel_file, evt_start, evt_end):
    """Load the same PE + selection arrays run_inf.py feeds the model."""
    import h5py
    import pandas as pd

    evt_start = 0 if evt_start is None else int(evt_start)
    try:
        with h5py.File(pe_file, "r") as f:
            m1s = np.asarray(f["m1"][evt_start:evt_end])
            qs = np.asarray(f["q"][evt_start:evt_end])
            dls = np.asarray(f["dl"][evt_start:evt_end])
            pdraws = np.asarray(f["pdraw"][evt_start:evt_end])
    except (KeyError, OSError):
        pe = pd.read_hdf(pe_file, key="samples").iloc[evt_start:evt_end]
        m1s = np.asarray(pe["m1"].to_list())
        qs = np.asarray(pe["q"].to_list())
        dls = np.asarray(pe["dl"].to_list())
        pdraws = np.asarray(pe["pdraw"].to_list())
    pdraws = np.nan_to_num(pdraws, neginf=-1e30, posinf=1e30)

    sel = pd.read_hdf(sel_file, key="true_parameters")
    ndraw = float(sel["ndraw"].iloc[0])
    return dict(
        m1s_det=m1s, qs=qs, dls=dls, log_pdraw=pdraws,
        m1s_det_sel=np.asarray(sel["m1d"].values),
        qs_sel=np.asarray(sel["q"].values),
        dls_sel=np.asarray(sel["dl"].values),
        pdraw_sel=np.asarray(sel["pdraw_sel"].values),
        Ndraw=ndraw,
    )


def _eval_point_from_spec(spec, prior, im):
    """Constrained parameter dict for init_to_value, in the prior's coordinates."""
    truths = spec.get("truths") or {}
    post_med = spec.get("posterior_median") or {}
    source = "truth" if truths else "posterior median"
    sample = _canonical_pop_sample(truths, post_med)
    if "log_fpeak" not in sample and "msigma_low" in sample:
        if "log_flow" in sample:
            sample["log_fpeak"] = float(sample["log_flow"]) - float(
                np.log(sample["msigma_low"]))
        elif "flow" in sample:
            sample["log_fpeak"] = (float(np.log(sample["flow"]))
                                   - float(np.log(sample["msigma_low"])))
    sample = im.map_truths_to_prior_coords(sample, prior)
    point = {}
    for k, v in sample.items():
        if k not in prior or isinstance(prior[k], float):
            continue
        try:
            point[k] = float(v)
        except (TypeError, ValueError):
            continue
    if not point:
        raise KeyError("no sampled-site values for init_to_value (source=%s)"
                       % source)
    return point, source


def _factor_values(im, model_args, model_kwargs, point):
    """Per-factor log contributions ('loglike', 'selfactor', guards, ...)."""
    import jax.numpy as jnp
    import numpyro.handlers as handlers

    truth_j = {k: jnp.asarray(v) for k, v in point.items()}
    with handlers.seed(rng_seed=0), handlers.substitute(data=truth_j):
        tr = handlers.trace(im.pop_cosmo_model).get_trace(
            *model_args, **model_kwargs)
    out = {}
    for name, site in tr.items():
        fn = site.get("fn")
        if fn is not None and type(fn).__name__ == "Unit":
            out[name] = float(np.asarray(fn.log_factor))
        elif site["type"] == "deterministic" and np.asarray(site["value"]).size == 1:
            out[name] = float(np.asarray(site["value"]))
    return out


def run_float32_leg(spec_path):
    """One precision leg, invoked as a subprocess so JAX_ENABLE_X64 is set
    before jax is imported.  Writes potential / gradient / factors to spec['out'].
    """
    with open(spec_path) as f:
        spec = json.load(f)

    import jax
    import jax.numpy as jnp
    import intensity_models_fast as im
    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value
    from utils import get_priors_from_file

    x64 = bool(jax.config.jax_enable_x64)
    print("  float32-leg: x64=%s  jax %s on %s" % (
        x64, jax.__version__, jax.devices()), file=sys.stderr, flush=True)

    prior = get_priors_from_file(spec["prior_path"])
    data = _load_pe_sel_arrays(
        spec["pe_file"], spec["sel_file"],
        spec.get("evt_start"), spec.get("evt_end"))
    data = _round_to_f32(data)
    point, source = _eval_point_from_spec(spec, prior, im)

    model_args = (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
        data["Ndraw"], prior,
    )
    model_kwargs = dict(
        use_low_bump=bool(spec.get("use_low_bump", True)),
        smooth_tail_edge=bool(spec.get("smooth_tail_edge", True)),
    )

    baselines = spec.get("baselines")
    if spec.get("recenter") and baselines is None:
        baselines = im.recentering_baselines(
            model_args, point, use_low_bump=model_kwargs["use_low_bump"],
            smooth_tail_edge=model_kwargs["smooth_tail_edge"])
        baselines = dict(
            loglike_ref=np.asarray(baselines["loglike_ref"]).tolist(),
            log_mu_sel_ref=float(baselines["log_mu_sel_ref"]),
            log_pdraw_sel_scale=float(baselines["log_pdraw_sel_scale"]),
            offset=float(baselines["offset"]),
        )
    if baselines is not None:
        model_kwargs.update(
            loglike_ref=np.asarray(baselines["loglike_ref"], np.float64),
            log_mu_sel_ref=float(baselines["log_mu_sel_ref"]),
            log_pdraw_sel_scale=float(baselines["log_pdraw_sel_scale"]),
        )

    truth_j = {k: jnp.asarray(v) for k, v in point.items()}
    mi = initialize_model(
        jax.random.PRNGKey(0), im.pop_cosmo_model,
        model_args=model_args, model_kwargs=model_kwargs, dynamic_args=False,
        init_strategy=init_to_value(values=truth_j),
    )
    z0 = mi.param_info.z
    z0_override = spec.get("z0")
    if z0_override is not None:
        # Same unconstrained point as the other precision leg.  Without this
        # the two z0s sit ~1e-7 apart and |dPE/dz|~1e5 fabricates ~1e-2 nats.
        z0 = {k: jnp.asarray(np.asarray(z0_override[k], np.float64),
                             dtype=z0[k].dtype)
              for k in z0 if k in z0_override}

    factors = _factor_values(im, model_args, model_kwargs, point)
    vg = jax.jit(jax.value_and_grad(mi.potential_fn))
    try:
        v, g = vg(z0)
        grad = {k: float(x) for k, x in g.items()}
    except Exception as exc:
        print("  value_and_grad failed (%s); falling back to value only"
              % exc, file=sys.stderr, flush=True)
        v = jax.jit(mi.potential_fn)(z0)
        grad = None
    v = float(v)
    print("  potential = %.10e" % v, file=sys.stderr, flush=True)

    result = dict(
        x64=x64, potential=v, grad=grad, factors=factors,
        z0={k: float(x) for k, x in z0.items()},
        baselines=baselines,
        point_source=source, point=point,
        nobs=int(np.asarray(data["m1s_det"]).shape[0]),
        nsamp=int(np.asarray(data["m1s_det"]).shape[1])
        if np.asarray(data["m1s_det"]).ndim > 1 else 1,
        nsel=int(np.asarray(data["m1s_det_sel"]).shape[0]),
        recentered=bool(baselines),
    )
    with open(spec["out"], "w") as f:
        json.dump(result, f)
    return 0


def _run_float32_leg_subprocess(spec, scratch, x64):
    spec = dict(spec)
    tag = "f64" if x64 else "f32"
    spec["out"] = os.path.join(scratch, tag + ".json")
    spec_path = os.path.join(scratch, "spec_%s.json" % tag)
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1" if x64 else "0"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("JAX_PLATFORMS", "cpu")
    cmd = [sys.executable, os.path.abspath(__file__), "--float32-leg", spec_path]
    proc = subprocess.run(
        cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-3000:]
        raise RuntimeError("float%s leg failed (exit %d):\n%s"
                           % ("64" if x64 else "32", proc.returncode, tail))
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    with open(spec["out"]) as f:
        return json.load(f)


def _f32_severity(abs_dv, one_minus_cos):
    if abs_dv >= F32_DPE_FAIL or (
            one_minus_cos is not None and one_minus_cos >= F32_GRAD_COS_FAIL):
        return "FAIL"
    if abs_dv >= F32_DPE_WARN or (
            one_minus_cos is not None and one_minus_cos >= F32_GRAD_COS_WARN):
        return "WARN"
    return "OK"


def section_float32(rep, idata, truths, prior_path, pe_file, sel_file, ini, nobs):
    """float32 vs float64 potential (and gradient) at the truth point.

    This is the run-specific version of scripts/testing_scripts/test_float32_accuracy.py.
    A single evaluation is a bound, not a lucky one: the audit found the truth
    is near the worst case because float32 error tracks |loglike|.
    """
    if not prior_path:
        rep.add("monte-carlo", "NOTE", "float32 safety",
                "insufficient information: no prior file, so the model cannot "
                "be rebuilt")
        return
    if not pe_file:
        rep.add("monte-carlo", "NOTE", "float32 safety",
                ["insufficient information: PE HDF5 not found",
                 "Need the run's output_file_PE next to the .nc (or a matching "
                 "run_configs/*.ini that names it)."])
        return
    if not sel_file:
        rep.add("monte-carlo", "NOTE", "float32 safety",
                "insufficient information: selection HDF5 not found")
        return

    post_med = _posterior_median_dict(idata.posterior)
    if not truths and not post_med:
        rep.add("monte-carlo", "NOTE", "float32 safety",
                "insufficient information: no pop_config truths and no "
                "posterior medians to evaluate at")
        return

    recenter = "recentering_offset" in idata.posterior.attrs
    spec = dict(
        pe_file=os.path.abspath(pe_file),
        sel_file=os.path.abspath(sel_file),
        prior_path=os.path.abspath(prior_path),
        truths=_plain_floats(truths),
        posterior_median=_plain_floats(post_med),
        evt_start=(ini or {}).get("evt_start", 0),
        evt_end=(ini or {}).get("evt_end"),
        use_low_bump=(ini or {}).get("use_low_bump", True),
        smooth_tail_edge=(ini or {}).get("smooth_tail_edge", True),
        recenter=recenter,
    )
    print("float32 safety: compiling the model at %s in float32 then float64 "
          "(CPU; can take a few minutes at production scale) ..."
          % ("truth" if truths else "posterior median"),
          file=sys.stderr, flush=True)

    try:
        with tempfile.TemporaryDirectory(prefix="diag_f32_") as scratch:
            r32 = _run_float32_leg_subprocess(spec, scratch, x64=False)
            spec64 = dict(spec, z0=r32["z0"], baselines=r32.get("baselines"))
            r64 = _run_float32_leg_subprocess(spec64, scratch, x64=True)
    except Exception as exc:
        rep.add("monte-carlo", "NOTE", "float32 safety",
                "insufficient information: could not evaluate the potential "
                "(%s)" % exc)
        return

    if r32.get("x64") or not r64.get("x64"):
        rep.add("monte-carlo", "NOTE", "float32 safety",
                "insufficient information: JAX x64 did not toggle between "
                "legs (float32 x64=%s, float64 x64=%s); comparison is not "
                "a precision test" % (r32.get("x64"), r64.get("x64")))
        return

    v32, v64 = float(r32["potential"]), float(r64["potential"])
    dv = v32 - v64
    abs_dv = abs(dv)
    if not np.isfinite(v32) or not np.isfinite(v64):
        rep.add("monte-carlo", "FAIL", "float32 safety",
                "potential is non-finite (float32=%s, float64=%s); the "
                "evaluation point is outside support or the model dumped NaNs"
                % (v32, v64))
        return
    g32, g64 = r32.get("grad"), r64.get("grad")
    one_minus_cos = None
    worst_rel = None
    if g32 and g64:
        keys = sorted(set(g32) & set(g64))
        coss_num = n2 = n4 = 0.0
        rels = []
        for k in keys:
            a, b = float(g32[k]), float(g64[k])
            rels.append(abs(a - b) / max(abs(b), 1e-30))
            coss_num += a * b
            n2 += a * a
            n4 += b * b
        if rels:
            worst_rel = float(max(rels))
        denom = (n2 * n4) ** 0.5
        if denom > 0:
            one_minus_cos = float(1.0 - coss_num / denom)

    sev = _f32_severity(abs_dv, one_minus_cos)
    nobs_used = r32.get("nobs") or nobs
    ratio_hmc = abs_dv / HMC_ENERGY_SCALE

    d = ["evaluated at %s; recentering %s (matches this run); "
         "nobs=%s nsamp=%s nsel=%s"
         % (r32.get("point_source", "?"),
            "on" if r32.get("recentered") else "off",
            r32.get("nobs"), r32.get("nsamp"), r32.get("nsel")),
         "same unconstrained z0 (from the float32 leg) and float32-rounded "
         "inputs on both legs, so this is arithmetic precision not a "
         "1e-7-in-z0 artifact",
         "potential float64 = %.6e, float32 = %.6e" % (v64, v32),
         "|dPE| = %.4e nats = %.3fx the ~%.2f nat HMC energy scale"
         % (abs_dv, ratio_hmc, HMC_ENERGY_SCALE)]

    f32_fac, f64_fac = r32.get("factors") or {}, r64.get("factors") or {}
    for name in ("loglike", "selfactor"):
        if name in f32_fac and name in f64_fac:
            d.append("  %s: f64=%+.6e  d=%+.3e"
                     % (name, f64_fac[name], f32_fac[name] - f64_fac[name]))
    if one_minus_cos is not None:
        d.append("gradient 1-cos = %.3e, worst-component rel = %s"
                 % (one_minus_cos,
                    ("%.2e" % worst_rel) if worst_rel is not None else "n/a"))
    else:
        d.append("gradient not available (value_and_grad failed; potential "
                 "comparison only)")

    if sev == "FAIL":
        d.append("Float32 roundoff of the potential is comparable to (or "
                 "larger than) the leapfrog integrator's own energy error, "
                 "so it can move the Metropolis accept/reject.  Precedent "
                 "scale: nobs=9000 measured |dPE|~0.02 nats (0.1x) and was "
                 "safe; this run is past that margin.  See "
                 "notes/2026-08-07-float32-accuracy-audit.md.")
        rep.rec(1, "float32 arithmetic", "verify with float64",
                "|dPE|=%.3e nats (%.2fx HMC energy scale) at nobs=%s; "
                "a float64 verification leg is needed before quoting "
                "this catalogue" % (abs_dv, ratio_hmc, nobs_used))
    elif sev == "WARN":
        d.append("Float32 error is a substantial fraction of the HMC energy "
                 "scale.  Fine for quoting if divergences are zero, but there "
                 "is little headroom for a larger catalogue.")
        rep.rec(2, "float32 arithmetic", "watch at next catalogue size",
                "|dPE|=%.3e nats (%.2fx HMC energy scale)" % (abs_dv, ratio_hmc))
    else:
        d.append("Float32 error is well below the leapfrog energy scale "
                 "(production-scale audit: ~0.02 nats / 0.1x at nobs=9000).  "
                 "The truth point is near the worst case, so this is a bound "
                 "over the posterior, not a lucky evaluation.")

    metrics = dict(dpe=dv, abs_dpe=abs_dv, ratio_hmc=ratio_hmc,
                   one_minus_cos=one_minus_cos, worst_grad_rel=worst_rel,
                   nobs=r32.get("nobs"), nsamp=r32.get("nsamp"),
                   nsel=r32.get("nsel"), recentered=r32.get("recentered"),
                   point_source=r32.get("point_source"),
                   potential_f32=v32, potential_f64=v64)
    rep.metrics["float32"] = metrics
    rep.add("monte-carlo", sev, "float32 safety", d, metrics)


def section_geometry(rep, idata, ini, min_ess, corr_info):
    frac = rep.metrics.get("depth_cap_frac")
    cap = rep.metrics.get("depth_cap")
    dense = (ini or {}).get("dense_mass", False)
    cap_cfg = (ini or {}).get("max_tree_depth")
    max_abs_r = corr_info.get("max_abs_r")
    cond = corr_info.get("cond")

    if frac is None:
        rep.add("geometry", "NOTE", "sampler geometry",
                "insufficient information: without n_steps/tree_depth the "
                "depth-vs-ESS diagnosis cannot be made.  ESS alone (min bulk "
                "%.0f) says %s." % (min_ess,
                                    "the sampler is fine" if min_ess >= 100
                                    else "the sampler is not mixing, cause "
                                         "undetermined"))
        return

    d = ["max_tree_depth = %s, saturated in %.1f%% of iterations, min bulk ESS "
         "%.0f, dense_mass = %s" % (cap, 100 * frac, min_ess, dense)]
    if cond is not None:
        d.append("posterior correlation matrix: condition number %.0f, "
                 "strongest |r| = %.3f" % (cond, max_abs_r))

    saturated = frac > 0.5
    low_ess = min_ess < 100
    already_tuned = (cap_cfg is not None and cap_cfg > 7) or dense

    if saturated and low_ess and already_tuned:
        sev = "FAIL"
        d.append("Depth %s and dense_mass=%s were ALREADY applied and the cap "
                 "is still saturated with ESS %.0f.  This is a CONDITIONING "
                 "problem, not a trajectory-length problem: each extra depth "
                 "level doubles the cost per iteration and buys only a factor "
                 "2 in trajectory length, so chasing it is expensive and "
                 "unlikely to close a %.0fx ESS gap." % (cap, dense, min_ess,
                                                         200.0 / max(min_ess, 1)))
        d.append("The better lever is reparameterization: sample the "
                 "combinations the posterior is actually curved in.  See the "
                 "degeneracy directions in the identifiability section, and "
                 "mass-model-audit.md sec 9.2 for the specific proposal on the "
                 "CO-IMF/bump ridge.")
        rep.rec(1, "reparameterization (model)", "change",
                "depth %s + dense_mass=%s still leaves ESS %.0f at 100%% "
                "saturation; the posterior is badly conditioned"
                % (cap, dense, min_ess))
        if not dense:
            rep.rec(2, "dense_mass", "true",
                    "strong correlations (max |r| %.2f) with a diagonal metric"
                    % (max_abs_r if max_abs_r is not None else float("nan")))
    elif saturated and low_ess:
        sev = "FAIL" if min_ess < 50 else "WARN"
        d.append("The cap binds in %.1f%% of iterations AND the chains are not "
                 "mixing: NUTS is being cut off before its U-turn.  Raising "
                 "max_tree_depth is indicated (cost is 2x per level)."
                 % (100 * frac))
        rep.rec(1, "max_tree_depth", "increase (%s -> %s)" % (cap, int(cap) + 3),
                "cap saturated in %.1f%% of iterations with min bulk ESS %.0f"
                % (100 * frac, min_ess))
        if not dense and max_abs_r is not None and max_abs_r > 0.7:
            d.append("The posterior also carries strong linear correlations "
                     "(max |r| = %.3f, condition number %.0f) that a diagonal "
                     "mass matrix cannot absorb; dense_mass lets NUTS learn "
                     "them instead of paying for them in trajectory length."
                     % (max_abs_r, cond))
            d.append("Caveat: a dense metric estimates d(d+1)/2 = %d entries "
                     "from the final warmup window, so nmcmc (warmup = samples "
                     "in run_inf.py) may need raising as well."
                     % (corr_info["d"] * (corr_info["d"] + 1) // 2))
            rep.rec(1, "dense_mass", "true",
                    "max |r| = %.3f, condition number %.0f with a diagonal "
                    "metric" % (max_abs_r, cond))
            rep.rec(3, "nmcmc", "increase",
                    "a dense metric estimates %d entries from the final warmup "
                    "window" % (corr_info["d"] * (corr_info["d"] + 1) // 2))
    elif saturated and not low_ess:
        sev = "NOTE"
        d.append("The cap binds but the chains still reached bulk ESS %.0f: "
                 "depth is the run's cost driver, not its failure mode.  "
                 "Raising max_tree_depth would buy more ESS per iteration at "
                 "2x wall time per level; it is an efficiency choice, not a "
                 "correctness one." % min_ess)
    elif not saturated and low_ess:
        sev = "FAIL" if min_ess < 50 else "WARN"
        d.append("The chains mix badly (%.0f) but the depth cap is reached in "
                 "only %.1f%% of iterations, so max_tree_depth is NOT the "
                 "binding constraint.  Look at the identifiability section: "
                 "low ESS with short trajectories points at the posterior "
                 "surface (walls, multimodality, divergences), not at the "
                 "sampler's budget." % (min_ess, 100 * frac))
    else:
        sev = "OK"
        d.append("Depth is rarely hit and ESS is comfortable: the sampler "
                 "settings were stronger than this posterior needed.  "
                 "max_tree_depth could be lowered, or nmcmc reduced, to buy "
                 "wall time on reruns.")
        rep.rec(5, "max_tree_depth / nmcmc", "decrease",
                "cap reached in only %.1f%% of iterations at min bulk ESS %.0f"
                % (100 * frac, min_ess))
    rep.add("geometry", sev, "sampler geometry", d,
            dict(frac_at_cap=frac, depth_cap=cap, dense_mass=bool(dense),
                 min_ess_bulk=min_ess))


def section_identifiability(rep, idata, free, prior, pinfo):
    post = idata.posterior
    X = np.column_stack([np.asarray(post[k].values).ravel() for k in free])
    corr_info = dict(d=len(free))

    # ---- prior-dominated marginals
    if pinfo:
        rows = []
        for k in free:
            info = pinfo.get(k)
            if not info or info.get("sd") in (None, 0):
                continue
            psd = info["sd"]
            osd = float(np.std(np.asarray(post[k].values)))
            rows.append((k, osd / psd, osd, psd))
        rows.sort(key=lambda r: -r[1])
        dominated = [r for r in rows if r[1] > 0.9]
        weak = [r for r in rows if 0.6 < r[1] <= 0.9]
        d = ["posterior sd / prior sd, worst first: " + ", ".join(
            "%s %.2f" % (k, r) for k, r, _, _ in rows[:8])]
        if dominated:
            sev = "WARN"
            d.append("PRIOR-DOMINATED (ratio > 0.9): " + ", ".join(
                "%s (%.2f)" % (k, r) for k, r, _, _ in dominated))
            d.append("For these the data said essentially nothing.  Their "
                     "medians are prior medians and must not be reported as "
                     "measurements.")
        elif weak:
            sev = "NOTE"
            d.append("Weakly informed (0.6 < ratio <= 0.9): " + ", ".join(
                "%s (%.2f)" % (k, r) for k, r, _, _ in weak))
        else:
            sev = "OK"
            d.append("Every free parameter is narrowed to < 0.6 of its prior "
                     "width; the likelihood is doing the work.")
        d.append("Caveat: a small ratio is necessary, not sufficient.  A "
                 "parameter can be narrowed by a degeneracy with another "
                 "parameter and still be unidentified; see the degeneracy "
                 "block below.")
        rep.add("identifiability", sev, "prior-dominated parameters", d,
                dict(sd_ratio={k: r for k, r, _, _ in rows}))
        rep.metrics["sd_ratio"] = {k: r for k, r, _, _ in rows}
        for k, r, _, _ in dominated:
            rep.rec(3, "prior for %s" % k, "report as prior-dominated",
                    "posterior sd is %.2f of the prior sd" % r)
    else:
        rep.add("identifiability", "NOTE", "prior-dominated parameters",
                "insufficient information: no prior file, so posterior widths "
                "cannot be compared to prior widths")

    # ---- prior walls
    if pinfo:
        walls = []
        for k in free:
            info = pinfo.get(k)
            if not info or info.get("sd") is None:
                continue
            v = np.asarray(post[k].values).ravel()
            eps = 0.01 * info["range"]
            for side, bound in (("low", info["low"]), ("high", info["high"])):
                if bound is None:
                    continue
                f = float(np.mean(np.abs(v - bound) <= eps))
                if f > 0.02:
                    walls.append((k, side, bound, f))
        walls.sort(key=lambda w: -w[3])
        pinned = [w for w in walls if w[3] > 0.5]
        touching = [w for w in walls if 0.1 < w[3] <= 0.5]
        many = [w for w in walls if w[3] > 0.3]
        d = []
        if walls:
            d.append("fraction of draws within 1% of the prior's effective "
                     "range of a bound (>2% of draws listed): " +
                     ", ".join("%s@%s=%.3g (%.0f%%)" % (k, s, b, 100 * f)
                               for k, s, b, f in walls[:10]))
        else:
            d.append("no free parameter piles up against a prior bound")
        if len(many) >= 3:
            sev = "FAIL"
            d.append("%d parameters are simultaneously pinned to prior walls. "
                     "That is the exact signature of the selection/event "
                     "density inconsistency documented in "
                     "notes/2026-08-08-tabulated-selection-consistency.md, "
                     "where the numerator used the tabulated mass function and "
                     "the selection integral did not; the uncancelled, "
                     "one-sided interpolation error gave the sampler a "
                     "direction to climb and it drove every shape parameter to "
                     "whichever wall made the mass function sharpest.  The "
                     "run that produced it (runs/endO5_evo) had sigma, "
                     "dmbhmax, a, b, mpisndot and fpl all on bounds."
                     % len(many))
            d.append("DIAGNOSIS vs SPECULATION: the wall pattern is measured "
                     "here; the cause is not.  Confirm by checking whether the "
                     "run predates the tabulate_selection fix, and by the "
                     "identity test test_tabulated_selection_consistency in "
                     "scripts/test_fast_equivalence.py.  Other causes of the "
                     "same pattern are a genuinely mis-specified model and a "
                     "prior whose support excludes the truth.")
            rep.rec(1, "model: tabulate_selection consistency", "verify",
                    "%d parameters pinned to prior walls, the wall-mode "
                    "signature of the split-density bug" % len(many))
        elif pinned:
            sev = "WARN"
            d.append("Pinned (> 50% of draws on the bound): " + ", ".join(
                "%s@%s" % (k, s) for k, s, _, _ in pinned))
            d.append("A pinned parameter's median and interval are set by the "
                     "prior's edge, not by the data.")
        elif touching:
            sev = "NOTE"
        else:
            sev = "OK"
        rep.add("identifiability", sev, "prior boundaries", d,
                dict(walls=[dict(param=k, side=s, bound=b, frac=f)
                            for k, s, b, f in walls]))
        rep.metrics["walls"] = [dict(param=k, side=s, bound=b, frac=f)
                                for k, s, b, f in walls]
    else:
        rep.add("identifiability", "NOTE", "prior boundaries",
                "insufficient information: no prior file, so truncation bounds "
                "are unknown")

    # ---- degeneracies
    from scipy.stats import spearmanr
    C = np.corrcoef(X.T)
    S = np.atleast_2d(spearmanr(X).statistic)
    iu = np.triu_indices(len(free), 1)
    pairs = sorted(zip(np.abs(C[iu]), C[iu], S[iu],
                       [(free[i], free[j]) for i, j in zip(*iu)]),
                   key=lambda t: -t[0])
    max_abs_r = float(pairs[0][0]) if pairs else 0.0
    ev, evec = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    cond = float(ev[order][0] / max(ev[order][-1], 1e-12))
    corr_info.update(max_abs_r=max_abs_r, cond=cond)
    rep.metrics.update(max_abs_r=max_abs_r, corr_condition_number=cond)

    d = []
    strong = [p for p in pairs if p[0] > 0.7][:10]
    curved = []
    for a_, r, rho, nm in strong:
        tag = ""
        if abs(rho) - abs(r) > 0.05:
            tag = "  <- |rho| > |r|: CURVED (banana) degeneracy, not linear"
            curved.append(nm)
        d.append("  %-28s r = %+.3f   rho = %+.3f%s"
                 % ("%s - %s" % nm, r, rho, tag))
    if strong:
        d.insert(0, "strongest correlated pairs (Pearson r, Spearman rho):")
    else:
        d.append("no pair exceeds |r| = 0.7")
    d.append("correlation-matrix eigenvalues (of %d): %s"
             % (len(free), ", ".join("%.2f" % e for e in ev[order][:5])))
    for j in order[:3]:
        if ev[j] < 2.0:
            break
        comps = sorted(zip(free, evec[:, j]), key=lambda t: -abs(t[1]))
        comps = [c for c in comps if abs(c[1]) > 0.2][:7]
        d.append("  lambda = %.2f (%.0f%% of %d parameters' variance in one "
                 "direction): %s" % (ev[j], 100 * ev[j] / len(free), len(free),
                                     "  ".join("%s%+.2f" % c for c in comps)))
    d.append("condition number of the correlation matrix = %.0f.  A diagonal "
             "mass matrix must pay for this in trajectory length; a dense "
             "metric absorbs the linear part of it." % cond)
    if curved:
        d.append("A curved degeneracy is not fixed by dense_mass (a global "
                 "linear metric cannot straighten it); it needs "
                 "reparameterization.")
    if max_abs_r > 0.9:
        sev = "WARN"
    elif max_abs_r > 0.7:
        sev = "NOTE"
    else:
        sev = "OK"
    rep.add("identifiability", sev, "degeneracy directions", d,
            dict(max_abs_r=max_abs_r, condition_number=cond,
                 top_eigenvalues=[float(e) for e in ev[order][:5]],
                 curved_pairs=["%s-%s" % nm for nm in curved]))

    # ---- the specific low-mass-bump pathology
    section_bump(rep, post, free, prior, C, evec[:, order[0]], ev[order][0])

    return corr_info


def _bump_value(post, prior, name):
    """Posterior median if sampled, prior float if pinned, else None."""
    if name in post and np.ptp(np.asarray(post[name].values)) > 0:
        return float(np.median(np.asarray(post[name].values))), "posterior median"
    if prior is not None and name in prior and isinstance(prior[name], float):
        return float(prior[name]), "fixed by prior"
    if name in post:
        return float(np.asarray(post[name].values).ravel()[0]), "fixed"
    return None, None


def section_bump(rep, post, free, prior, C, top_vec, top_ev):
    from scipy.stats import norm
    msl, msl_src = _bump_value(post, prior, "msigma_low")
    mpl, mpl_src = _bump_value(post, prior, "mp_low")
    try:
        from utils import BUMP_MSIGMA_LOW_MAX as WIDTH_MAX
    except Exception:
        WIDTH_MAX = 2.5

    if msl is None or mpl is None:
        rep.add("identifiability", "NOTE", "low-mass bump vs CO-IMF slope `a`",
                "insufficient information: msigma_low / mp_low absent from the "
                "posterior, so the bump-width check cannot be made")
        return

    ratio = msl / mpl
    frac_6_20 = float(norm.cdf((20.0 - mpl) / msl) - norm.cdf((6.0 - mpl) / msl))
    # Fisher sigma(a) vs true width, log-interpolated from mass-model-audit.md.
    ws = np.array(sorted(BUMP_WIDTH_SIGMA_A))
    sg = np.array([BUMP_WIDTH_SIGMA_A[w] for w in ws])
    sigma_a_pred = float(np.exp(np.interp(np.clip(msl, ws[0], ws[-1]), ws, np.log(sg))))

    d = ["msigma_low = %.3f (%s), mp_low = %.3f (%s)" % (msl, msl_src, mpl, mpl_src),
         "msigma_low / mp_low = %.3f (limit %.2f); utils.BUMP_MSIGMA_LOW_MAX = "
         "%.1f Msun" % (ratio, BUMP_RATIO_MAX, WIDTH_MAX),
         "Gaussian mass inside 6-20 Msun = %.0f%% -- that window is the only "
         "place the CO-IMF index `a` has leverage (below mco_floor = 6 and "
         "above the 20 Msun break `a` is a pure amplitude, exactly absorbed by "
         "log_flow / log_fpl under the marginalized rate R)."
         % (100 * frac_6_20),
         "Fisher sigma(a) at this width, interpolated from mass-model-audit.md "
         "sec 6 (2.55 / 0.73 / 0.29 / 0.15 at width 4 / 3 / 2 / 1): ~%.2f"
         % sigma_a_pred]

    # correlation block
    present = [p for p in ("a", "log_flow", "msigma_low", "mp_low") if p in free]
    block_rs = []
    if "a" in free:
        ai = free.index("a")
        for p in present:
            if p == "a":
                continue
            block_rs.append((p, float(C[ai, free.index(p)])))
    if block_rs:
        d.append("`a` correlations within the bump block: " + ", ".join(
            "%s %+.3f" % (p, r) for p, r in block_rs))
    top_load = {n: v for n, v in zip(free, top_vec)}
    block_load = sum(abs(top_load.get(p, 0.0)) for p in BUMP_BLOCK & set(free))
    total_load = sum(abs(v) for v in top_vec)
    block_share = block_load / total_load if total_load else 0.0

    too_broad = (msl > WIDTH_MAX) or (ratio > BUMP_RATIO_MAX)
    block_active = bool(block_rs) and max(abs(r) for _, r in block_rs) > 0.6

    if too_broad and "a" in free:
        sev = "WARN"
        d.append("VERDICT: the low-mass Gaussian is too broad to be a bump.  "
                 "At this width it is a second continuum spanning the same "
                 "6-20 Msun window `a` controls, and the likelihood is flat in "
                 "`a` over roughly the lower half of its prior range.  DO NOT "
                 "report `a` as measured; quote it as prior-dominated or "
                 "profile it out (mass-model-audit.md sec 9.1).")
        if block_active:
            d.append("Confirmed in this chain: |r| up to %.2f between `a` and "
                     "the bump parameters, and the dominant correlation "
                     "direction (lambda = %.2f) puts %.0f%% of its loading on "
                     "the a - log_flow - msigma_low - mp_low block."
                     % (max(abs(r) for _, r in block_rs), top_ev,
                        100 * block_share))
        else:
            d.append("The a-bump correlation block is NOT visible in this "
                     "chain (max |r| = %.2f).  The width argument above is a "
                     "property of the model, not of the chain, so it stands; "
                     "but if the sampler section reports FAIL, these posterior "
                     "medians are themselves unreliable and the missing "
                     "correlations are most likely a mixing artifact rather "
                     "than evidence against the degeneracy."
                     % (max(abs(r) for _, r in block_rs) if block_rs else 0.0))
        rep.rec(2, "msigma_low prior (and true value in mock pop configs)",
                "cap at <= 2.5 Msun",
                "msigma_low = %.2f makes `a` unidentifiable "
                "(Fisher sigma(a) ~ %.2f)" % (msl, sigma_a_pred))
        rep.rec(3, "reparameterization (model)", "change",
                "sample the dN/dm slope over a fixed 15-20 Msun window and "
                "derive `a`, per mass-model-audit.md sec 9.2")
    elif too_broad:
        sev = "NOTE"
        d.append("The bump is broad, but `a` is not a free parameter in this "
                 "run, so nothing is lost here.")
    else:
        sev = "OK"
        d.append("The bump is narrow enough that `a` retains independent "
                 "leverage in 6-20 Msun.")
        if block_active:
            d.append("Note the a-bump correlations are still present (|r| up "
                     "to %.2f); they are just not fatal at this width."
                     % max(abs(r) for _, r in block_rs))
    rep.add("identifiability", sev, "low-mass bump vs CO-IMF slope `a`", d,
            dict(msigma_low=msl, mp_low=mpl, ratio=ratio,
                 bump_frac_6_20=frac_6_20, fisher_sigma_a=sigma_a_pred,
                 block_loading_share=block_share))
    rep.metrics.update(msigma_low=msl, mp_low=mpl, bump_ratio=ratio,
                       bump_frac_6_20=frac_6_20)


def section_truths(rep, post, free, truths, min_ess):
    if truths is None:
        rep.add("truths", "NOTE", "truth recovery",
                "insufficient information: no pop_config, so truths are unknown")
        return
    rows = []
    for k in free:
        if k not in truths:
            continue
        v = np.asarray(post[k].values).ravel()
        q = float(np.mean(v < truths[k]))
        rows.append((k, truths[k], float(np.median(v)), q))
    if not rows:
        rep.add("truths", "NOTE", "truth recovery",
                "the pop_config carries no truth for any free parameter")
        return
    in68 = sum(1 for _, _, _, q in rows if 0.16 <= q <= 0.84)
    in95 = sum(1 for _, _, _, q in rows if 0.025 <= q <= 0.975)
    misses = [r for r in rows if not (0.025 <= r[3] <= 0.975)]
    d = ["truth quantile within each marginal (0.5 = dead centre):"]
    for k, t, m, q in sorted(rows, key=lambda r: -abs(r[3] - 0.5)):
        flag = "  <- outside 95%" if not (0.025 <= q <= 0.975) else (
            "  <- outside 68%" if not (0.16 <= q <= 0.84) else "")
        d.append("  %-12s truth %10.4f   median %10.4f   q = %.3f%s"
                 % (k, t, m, q, flag))
    d.append("%d/%d truths inside 68%%, %d/%d inside 95%%"
             % (in68, len(rows), in95, len(rows)))

    unreliable = min_ess < 100
    if unreliable:
        sev = "NOTE"
        d.append("*** These quantiles are NOT interpretable at min bulk ESS "
                 "%.0f. ***  A marginal built from ~%.0f independent draws has "
                 "no resolved tails, so both 'recovered' and 'missed' verdicts "
                 "are chain artifacts.  Precedent: runs/endO5_fullcosmo_evo2 "
                 "at bulk ESS ~3 shows 17/18 truths inside 95%% while being "
                 "unusable, and runs/endO5_fullcosmo_evo3 at ESS 22 appears to "
                 "badly miss `a` for a reason that is a model property, not a "
                 "recovery failure.  Fix the sampler first, then read this "
                 "section."
                 % (min_ess, min_ess))
    elif misses:
        sev = "WARN"
        d.append("Truths outside 95%%: %s.  With ESS %.0f this is worth "
                 "investigating, but check the identifiability section first: "
                 "a prior-dominated or degenerate parameter can miss its truth "
                 "with a perfectly healthy sampler."
                 % (", ".join(m[0] for m in misses), min_ess))
    else:
        sev = "OK"
    rep.add("truths", sev, "truth recovery", d,
            dict(in68=in68, in95=in95, n=len(rows),
                 quantiles={k: q for k, _, _, q in rows},
                 ess_caveat=bool(unreliable)))


# ---------------------------------------------------------------- output

COLORS = {"OK": "\033[32m", "NOTE": "\033[36m", "WARN": "\033[33m", "FAIL": "\033[31m"}
RESET = "\033[0m"


def tag(sev, use_color):
    if use_color:
        return "%s[%-4s]%s" % (COLORS[sev], sev, RESET)
    return "[%-4s]" % sev


def print_report(rep, run_name, use_color):
    print("=" * 78)
    print("RUN DIAGNOSIS: %s" % run_name)
    print("=" * 78)
    titles = [("inventory", "0. RUN INVENTORY"),
              ("convergence", "1. SAMPLER CONVERGENCE"),
              ("monte-carlo", "2. MONTE-CARLO ADEQUACY"),
              ("geometry", "3. SAMPLER GEOMETRY"),
              ("identifiability", "4. MODEL CHOICE / IDENTIFIABILITY"),
              ("truths", "5. TRUTH RECOVERY")]
    present = {f.section for f in rep.findings}
    order = [s for s, _ in titles if s in present]
    order += [s for s in sorted(present) if s not in dict(titles)]
    titles = dict(titles)
    for sec in order:
        print("\n" + titles.get(sec, sec.upper()))
        print("-" * 78)
        for f in rep.findings:
            if f.section != sec:
                continue
            print("%s %s" % (tag(f.severity, use_color), f.title))
            for line in f.detail:
                for sub in str(line).split("\n"):
                    print("       " + sub)
    print("\n" + "=" * 78)
    print("RECOMMENDED ACTIONS (prioritized, deduplicated)")
    print("-" * 78)
    recs = rep.deduped_recs()
    if not recs:
        print("  none: no setting needs changing on the evidence in this .nc.")
    for i, r in enumerate(recs, 1):
        print("  %d. %-46s -> %s" % (i, r["key"], r["direction"]))
        print("     because: %s" % r["rationale"])
    worst = rep.worst()
    print("-" * 78)
    print("OVERALL: %s   (exit code %d; non-zero only on FAIL)"
          % (tag(worst, use_color), 1 if worst == "FAIL" else 0))
    print("=" * 78)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description="Diagnose a finished inference run from its .nc.")
    p.add_argument("--run", default=None, help="run directory name under --runs_dir")
    p.add_argument("--nc", default=None, help="explicit path to the .nc")
    p.add_argument("--prior", default=None, help="prior file (else inferred)")
    p.add_argument("--pop_config", default=None, help="truth file (else inferred)")
    p.add_argument("--runs_dir", default="../runs")
    p.add_argument("--priors_dir", default="/priors")
    p.add_argument("--pop_configs_dir", default="pop_configs")
    p.add_argument("--run_configs_dir", default="run_configs")
    p.add_argument("--json", action="store_true", help="emit findings as JSON")
    p.add_argument("--no_color", action="store_true")
    p.add_argument("--test-float32", action="store_true",
                   help="run the float32-vs-float64 potential check (needs "
                        "the run's PE and selection HDF5; compiles the model "
                        "twice and can take a few minutes)")
    p.add_argument("--float32-leg", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.float32_leg:
        return run_float32_leg(args.float32_leg)

    nc_path, run_dir, run_name = resolve_nc(args)

    ini_path = find_ini(run_dir, run_name, args.run_configs_dir)
    ini = read_ini(ini_path) if ini_path else {}

    prior_path = args.prior
    if prior_path is None and ini.get("prior"):
        cand = os.path.join(args.priors_dir, ini["prior"])
        if os.path.exists(cand):
            prior_path = cand
    if prior_path is not None and not os.path.exists(prior_path):
        prior_path = None

    pop_path = args.pop_config
    if pop_path is None and ini.get("pop_config_file"):
        for cand in (os.path.join(args.pop_configs_dir, ini["pop_config_file"]),
                     os.path.join(run_dir, ini["pop_config_file"])):
            if os.path.exists(cand):
                pop_path = cand
                break
    if pop_path is not None and not os.path.exists(pop_path):
        pop_path = None

    nobs = None
    if ini.get("evt_end") is not None:
        nobs = int(ini["evt_end"]) - int(ini.get("evt_start") or 0)

    idata = az.from_netcdf(nc_path)
    post = idata.posterior

    prior = None
    pinfo = None
    if prior_path:
        try:
            from utils import get_priors_from_file
            prior = get_priors_from_file(prior_path)
            pinfo = prior_summary(prior)
        except Exception as exc:
            print("WARNING: could not parse prior %s (%s); prior-based checks "
                  "will be skipped" % (prior_path, exc), file=sys.stderr)
            prior, pinfo = None, None

    truths = load_truths(pop_path) if pop_path else None

    rep = Report()
    section_inventory(rep, nc_path, idata, ini_path, ini, prior_path, pop_path, nobs)

    free = free_parameters(post, prior)
    rep.metrics["free_parameters"] = free
    rep.add("inventory", "OK", "free parameters",
            ["%d free (varying) parameters: %s" % (len(free), ", ".join(free)),
             "constant columns were removed by an exact range test "
             "(np.ptp == 0), not a std threshold" +
             ("" if prior else "; without a prior file the sampled/derived "
                               "split is taken from the site names")])
    if len(free) < 2:
        print("fewer than 2 free parameters; nothing to diagnose", file=sys.stderr)
        return 0

    summ, min_ess, max_rhat = section_convergence(rep, idata, free, ini)
    section_monte_carlo(rep, idata, nobs, ini)
    sel_file = resolve_sel_file(run_dir, ini)
    section_selection_tilt(rep, idata, free, truths, nobs, sel_file, ini)
    if args.test_float32:
        pe_file = resolve_pe_file(run_dir, ini)
        section_float32(rep, idata, truths, prior_path, pe_file, sel_file, ini, nobs)
    corr_info = section_identifiability(rep, idata, free, prior, pinfo)
    section_geometry(rep, idata, ini, min_ess, corr_info)
    section_truths(rep, post, free, truths, min_ess)

    if args.json:
        out = dict(run=run_name, nc=nc_path, config=ini_path, prior=prior_path,
                   pop_config=pop_path, nobs=nobs,
                   overall=rep.worst(),
                   findings=[f.as_dict() for f in rep.findings],
                   recommendations=rep.deduped_recs(),
                   metrics=rep.metrics)
        print(json.dumps(out, indent=2, default=float))
    else:
        print_report(rep, run_name,
                     use_color=(not args.no_color) and sys.stdout.isatty())

    return 1 if rep.worst() == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

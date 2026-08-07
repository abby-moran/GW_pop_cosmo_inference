"""
Is float32 safe for this model at production scale?

The equivalence suite compares float32 against float32 (fast vs slow) and scores
sub-components against float64 references, but its only full-potential checks run
at nobs=400 / nsamp=300 / nsel=40k.  Float32 accumulation error in the potential
grows with nobs and nsel, so those checks are ~20x more forgiving than a
production run.  This script closes that gap.

Three independent measurements, all at whatever scale you ask for:

  A. float32 vs float64, evaluated at *bit-identical* unconstrained points and
     on *bit-identical* (float32-rounded) input data, so the only difference is
     arithmetic precision.  Reports the potential error in nats -- the unit HMC's
     Metropolis step actually cares about -- decomposed into the 'loglike' and
     'selfactor' terms, plus per-component gradient errors.

  B. Reduction-order jitter, float32 only, no reference needed: permuting the
     event order and the selection-sample order leaves the potential
     mathematically invariant, so the spread over permutations *is* the float32
     noise floor as HMC sees it (a non-smooth jitter, unlike the smooth
     Monte-Carlo error of the estimator itself).

  C. Scaling with nobs.  Both of the above at several nobs, to check the
     predicted ~linear growth and extrapolate to future catalogue sizes.

Usage:
    uv run python test_float32_accuracy.py                    # production scale
    uv run python test_float32_accuracy.py --scaling           # + nobs scan
    uv run python test_float32_accuracy.py --nobs 2000 --nsamp 1000 --nsel 200000

The float64 leg re-executes this file in a subprocess with JAX_ENABLE_X64=1,
because x64 must be set before jax is imported.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.append("../src/")

import numpy as np

# Accept prob targeted by NUTS' dual averaging is 0.8, which corresponds to a
# typical per-trajectory |Delta H| of a few tenths of a nat.  Float32 jitter
# well below that is invisible to the sampler; jitter comparable to it degrades
# the acceptance rate and, in the limit, biases the stationary distribution.
HMC_ENERGY_SCALE = 0.2
# numpyro flags a divergence at delta_energy > max_delta_energy (default 1000).
DIVERGENCE_THRESHOLD = 1000.0

# Parameter points to probe.  A single point (the truth) is not a bound over the
# posterior: roundoff depends on how sharp the mass function is (a small `sigma`
# puts a narrow Gaussian into the tabulated mass grid) and on how many events sit
# near the edge of support (more events pushed to _LOG_ZERO_FLOOR).  These mirror
# the sweep in test_fast_equivalence.PARAM_POINTS, all inside the prior support.
STRESS_POINTS = {
    "truth": {},
    # sharpest PISN smoothing, low mass scale, high h.  sigma must be strictly
    # inside TruncatedNormal(0.1, 0.1, low=0.05): exactly 0.05 has log_prob
    # -inf, so initialize_model cannot find valid initial parameters.
    "sharp": dict(sigma=0.051, mpisn=22.0, dmbhmax=0.7, h=0.9, lam=0.5,
                  dkappa=1.2, zp=3.0),
    # broad / heavy: shifts many samples across the tail edge
    "broad": dict(sigma=0.35, mpisn=45.0, dmbhmax=6.0, c=5.0, a=2.0, b=3.0),
}


def _factor_values(im, model_args, model_kwargs, truth):
    """The individual log-factor contributions ('loglike', 'selfactor', guards),
    so an error in the total can be attributed to a term."""
    import jax
    import numpyro.handlers as handlers

    with handlers.seed(rng_seed=0), handlers.substitute(data=truth):
        tr = handlers.trace(im.pop_cosmo_model).get_trace(*model_args, **model_kwargs)
    out = {}
    for name, site in tr.items():
        fn = site.get("fn")
        if fn is not None and type(fn).__name__ == "Unit":
            out[name] = float(np.asarray(fn.log_factor))
        elif site["type"] == "deterministic" and np.asarray(site["value"]).size == 1:
            out[name] = float(np.asarray(site["value"]))
    return out


def _model_args(data, prior):
    return (
        data["m1s_det"], data["qs"], data["dls"], data["log_pdraw"],
        data["m1s_det_sel"], data["qs_sel"], data["dls_sel"], data["pdraw_sel"],
        data["Ndraw"], prior,
    )


def _build(im, data, prior, truth, use_low_bump, z0_override=None,
           extra_model_kwargs=None):
    """initialize_model at `truth` and return (potential_fn, z0, factors)."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import initialize_model
    from numpyro.infer import init_to_value

    model_args = _model_args(data, prior)
    model_kwargs = dict(use_low_bump=use_low_bump)
    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)
    truth_j = {k: jnp.asarray(v) for k, v in truth.items()}

    mi = initialize_model(
        jax.random.PRNGKey(0), im.pop_cosmo_model,
        model_args=model_args, model_kwargs=model_kwargs, dynamic_args=False,
        init_strategy=init_to_value(values=truth_j),
    )
    z0 = mi.param_info.z
    if z0_override is not None:
        # Evaluate at the *same* point as the other precision leg.  Without this
        # the two legs sit ~1e-7 apart in z, and with |dPE/dz| ~ 1e5 that alone
        # fabricates ~1e-2 nats of apparent difference.
        z0 = {k: jnp.asarray(np.asarray(z0_override[k], np.float64), dtype=z0[k].dtype)
              for k in z0}
    factors = _factor_values(im, model_args, model_kwargs, truth_j)
    return mi.potential_fn, z0, factors


def _round_to_f32(data):
    """Round every input array to float32 and store it back as float64.

    Both legs then see numerically identical inputs, so the comparison isolates
    arithmetic precision rather than input precision.
    """
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = np.asarray(np.asarray(v, np.float32), np.float64)
        else:
            out[k] = float(np.float32(v))
    return out


def run_leg(args):
    """One precision leg.  Writes potential, gradient and factor values to
    --out as JSON.  Also runs the permutation test when asked (float32 only)."""
    import jax
    import jax.numpy as jnp
    import importlib

    im = importlib.import_module(args.module)
    x64 = bool(jax.config.jax_enable_x64)
    print(f"  leg: x64={x64}  jax {jax.__version__} on {jax.devices()}", flush=True)

    from bench_model import make_synthetic_data, build_prior, TRUTH

    data = _round_to_f32(make_synthetic_data(args.nobs, args.nsamp, args.nsel))
    prior = build_prior(True, os.environ.get("SCRATCH_PRIOR", "/tmp/f32acc_prior.prior"))
    truth = {k: v for k, v in TRUTH.items()
             if k in prior and not isinstance(prior[k], float)}
    truth.update(STRESS_POINTS[args.point])

    z0_override = None
    if args.z0_in:
        with open(args.z0_in) as f:
            z0_override = json.load(f)

    # Recentering baselines.  Computed at the *unmodified* TRUTH point even
    # for stress points, so 'sharp'/'broad' probe an evaluation far from the
    # reference -- the situation a warmup trajectory is in.  The float64 leg
    # must load the float32 leg's baselines (like z0): legs with different
    # constants would differ by a real constant, not by roundoff.
    recenter_kwargs = None
    if args.recenter:
        if args.ref_in:
            with open(args.ref_in) as f:
                ref = json.load(f)
        else:
            ref_truth = {k: v for k, v in TRUTH.items()
                         if k in prior and not isinstance(prior[k], float)}
            ref = im.recentering_baselines(_model_args(data, prior), ref_truth,
                                           use_low_bump=not args.no_low_bump)
            ref = dict(loglike_ref=np.asarray(ref["loglike_ref"]).tolist(),
                       log_mu_sel_ref=ref["log_mu_sel_ref"],
                       offset=ref["offset"])
            if args.ref_out:
                with open(args.ref_out, "w") as f:
                    json.dump(ref, f)
        recenter_kwargs = dict(
            loglike_ref=np.asarray(ref["loglike_ref"], np.float64),
            log_mu_sel_ref=float(ref["log_mu_sel_ref"]),
        )
        print(f"  recentering on: dropped offset = {ref['offset']:.6e}", flush=True)

    pe_fn, z0, factors = _build(im, data, prior, truth, not args.no_low_bump,
                                z0_override=z0_override,
                                extra_model_kwargs=recenter_kwargs)
    vg = jax.jit(jax.value_and_grad(pe_fn))
    v, g = vg(z0)
    v = float(v)
    g = {k: float(x) for k, x in g.items()}

    result = dict(x64=x64, potential=v, grad=g, factors=factors,
                  z0={k: float(x) for k, x in z0.items()},
                  nobs=args.nobs, nsamp=args.nsamp, nsel=args.nsel,
                  recentered=bool(args.recenter))
    print(f"  potential = {v:.10e}", flush=True)

    # ---- B. reduction-order jitter (float32 leg only) --------------------
    if args.permutations and not x64:
        rng = np.random.default_rng(12345)
        vals = [v]
        for ip in range(args.permutations):
            pe = rng.permutation(args.nobs)
            ps = rng.permutation(args.nsel)
            pdata = dict(data)
            for k in ("m1s_det", "qs", "dls", "log_pdraw"):
                pdata[k] = data[k][pe]
            for k in ("m1s_det_sel", "qs_sel", "dls_sel", "pdraw_sel"):
                pdata[k] = data[k][ps]
            # Self-validation: a permutation probe that silently failed to
            # permute would report zero jitter regardless of the real noise,
            # which is exactly the false negative this test must not produce.
            for k in ("m1s_det", "m1s_det_sel"):
                assert not np.array_equal(pdata[k], data[k]), \
                    f"permutation {ip} did not change {k} -- jitter probe is a no-op"
                assert np.array_equal(np.sort(pdata[k], axis=None),
                                      np.sort(data[k], axis=None)), \
                    f"permutation {ip} changed the multiset of {k}"
            pkwargs = recenter_kwargs
            if recenter_kwargs is not None:
                # the per-event baseline must follow its events through the
                # permutation, or the potential really would change
                pkwargs = dict(recenter_kwargs,
                               loglike_ref=recenter_kwargs["loglike_ref"][pe])
            pe_fn_p, z0_p, _ = _build(im, pdata, prior, truth, not args.no_low_bump,
                                      z0_override=result["z0"],
                                      extra_model_kwargs=pkwargs)
            vp = float(jax.jit(jax.value_and_grad(pe_fn_p))(z0_p)[0])
            vals.append(vp)
            print(f"    permutation {ip}: {vp:.10e}  (delta {vp - v:+.3e})", flush=True)
        result["permutation_potentials"] = vals

    try:
        result["peak_gib"] = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30
    except Exception:
        pass

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    if args.z0_out:
        with open(args.z0_out, "w") as f:
            json.dump(result["z0"], f)
    return result


# --------------------------------------------------------------------------
# Driver: run the float32 leg in-process, the float64 leg in a subprocess.
# --------------------------------------------------------------------------
def compare(args, tag=""):
    scratch = args.scratch
    f32_out = os.path.join(scratch, f"f32{tag}.json")
    f64_out = os.path.join(scratch, f"f64{tag}.json")
    z0_file = os.path.join(scratch, f"z0{tag}.json")
    ref_file = os.path.join(scratch, f"ref{tag}.json")

    base = [sys.executable, os.path.abspath(__file__), "--leg",
            "--module", args.module, "--nobs", str(args.nobs),
            "--nsamp", str(args.nsamp), "--nsel", str(args.nsel),
            "--point", args.point]
    if args.no_low_bump:
        base.append("--no_low_bump")
    f32_extra, f64_extra = [], []
    if args.recenter:
        f32_extra = ["--recenter", "--ref_out", ref_file]
        f64_extra = ["--recenter", "--ref_in", ref_file]

    print(f"\n--- float32 leg (nobs={args.nobs}, nsamp={args.nsamp}, "
          f"nsel={args.nsel}{', recentered' if args.recenter else ''}) ---",
          flush=True)
    env = dict(os.environ, JAX_ENABLE_X64="0")
    subprocess.run(base + f32_extra + ["--out", f32_out, "--z0_out", z0_file,
                           "--permutations", str(args.permutations)],
                   check=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))

    print("\n--- float64 leg (reference, same point, same inputs) ---", flush=True)
    env = dict(os.environ, JAX_ENABLE_X64="1")
    subprocess.run(base + f64_extra + ["--out", f64_out, "--z0_in", z0_file,
                           "--permutations", "0"],
                   check=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))

    with open(f32_out) as f:
        r32 = json.load(f)
    with open(f64_out) as f:
        r64 = json.load(f)
    return r32, r64


def report(r32, r64):
    print("\n" + "=" * 74)
    print(f"RESULTS  nobs={r32['nobs']}  nsamp={r32['nsamp']}  nsel={r32['nsel']}")
    print("=" * 74)

    v32, v64 = r32["potential"], r64["potential"]
    dv = v32 - v64
    print(f"\nA. float32 vs float64 potential (same point, same inputs)")
    print(f"   float64 reference : {v64:+.10e}")
    print(f"   float32           : {v32:+.10e}")
    print(f"   absolute error    : {dv:+.4e} nats")
    print(f"   relative error    : {abs(dv)/max(abs(v64),1e-30):.3e}")
    print(f"   vs HMC energy scale (~{HMC_ENERGY_SCALE} nats): "
          f"{abs(dv)/HMC_ENERGY_SCALE:.4f}x")
    print(f"   vs divergence threshold ({DIVERGENCE_THRESHOLD:.0f} nats): "
          f"{abs(dv)/DIVERGENCE_THRESHOLD:.2e}x")

    print(f"\n   term decomposition (float32 - float64, nats):")
    keys = sorted(set(r32["factors"]) & set(r64["factors"]))
    for k in keys:
        a, b = r32["factors"][k], r64["factors"][k]
        d = a - b
        rel = abs(d) / max(abs(b), 1e-30)
        flag = "  <-- dominant" if abs(d) > 0.5 * abs(dv) and abs(dv) > 0 else ""
        print(f"     {k:22s} f64={b:+.8e}  d={d:+.3e}  rel={rel:.2e}{flag}")

    print(f"\n   gradient (relative error per component):")
    gk = sorted(set(r32["grad"]) & set(r64["grad"]))
    rels, coss_num, n32, n64 = [], 0.0, 0.0, 0.0
    for k in gk:
        a, b = r32["grad"][k], r64["grad"][k]
        rel = abs(a - b) / max(abs(b), 1e-30)
        rels.append(rel)
        coss_num += a * b
        n32 += a * a
        n64 += b * b
        print(f"     d/d{k:12s} f64={b:+.6e}  f32={a:+.6e}  rel={rel:.2e}")
    cos = coss_num / max(np.sqrt(n32 * n64), 1e-30)
    print(f"   worst component relative error : {max(rels):.3e}")
    print(f"   gradient direction cos(angle)  : {cos:.12f}  "
          f"(1 - cos = {1-cos:.3e})")
    print(f"   gradient norm ratio |f32|/|f64|: {np.sqrt(n32/max(n64,1e-30)):.10f}")

    jitter = None
    if "permutation_potentials" in r32:
        vals = np.array(r32["permutation_potentials"])
        jitter = float(vals.std(ddof=1))
        print(f"\nB. reduction-order jitter (float32, {len(vals)} orderings)")
        print(f"   spread (max-min)  : {vals.max()-vals.min():.4e} nats")
        print(f"   std               : {jitter:.4e} nats")
        print(f"   distinct values   : {len(np.unique(vals))} of {len(vals)}")
        if r32.get("recentered"):
            print("   (recentered: the centered sums are ~0, so no loglike-ulp "
                  "floor applies)")
        else:
            ulp = float(np.spacing(np.float32(abs(r32["factors"].get("loglike", vals[0])))))
            print(f"   1 ulp of the loglike sum: {ulp:.3e} nats "
                  f"(the floor this probe can resolve)")
        print(f"   vs HMC energy scale (~{HMC_ENERGY_SCALE} nats): "
              f"{jitter/HMC_ENERGY_SCALE:.4f}x")
        if len(np.unique(vals)) == 1:
            print("   NOTE: all orderings bit-identical -> jitter is below 1 ulp of "
                  "the dominant sum;\n         treat test A as the binding measurement.")
    if "peak_gib" in r64:
        print(f"\n   peak GPU memory: float32 {r32.get('peak_gib', float('nan')):.2f} GiB"
              f" | float64 {r64['peak_gib']:.2f} GiB")
    return dict(nobs=r32["nobs"], nsel=r32["nsel"], dv=dv, worst_grad_rel=max(rels),
                one_minus_cos=1 - cos, jitter=jitter)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leg", action="store_true", help="internal: run one precision leg")
    p.add_argument("--module", default="intensity_models_fast")
    p.add_argument("--nobs", type=int, default=9000)
    p.add_argument("--nsamp", type=int, default=4000)
    p.add_argument("--nsel", type=int, default=1_700_000)
    p.add_argument("--no_low_bump", action="store_true")
    p.add_argument("--permutations", type=int, default=4)
    p.add_argument("--point", default="truth", choices=sorted(STRESS_POINTS),
                   help="parameter point to evaluate at")
    p.add_argument("--stress", action="store_true",
                   help="repeat at every STRESS_POINTS entry (bound over the posterior, "
                        "not just the truth)")
    p.add_argument("--scaling", action="store_true",
                   help="also scan nobs to measure how the error grows")
    p.add_argument("--recenter", action="store_true",
                   help="subtract constant baselines (evaluated at TRUTH) inside "
                        "the model's sums; see notes/2026-08-07-float32-recentering.md")
    p.add_argument("--out", default="/tmp/f32acc.json")
    p.add_argument("--z0_in", default=None)
    p.add_argument("--z0_out", default=None)
    p.add_argument("--ref_in", default=None)
    p.add_argument("--ref_out", default=None)
    p.add_argument("--scratch", default="/tmp")
    args = p.parse_args()

    if args.leg:
        run_leg(args)
        return

    summaries = [report(*compare(args, tag=f"_{args.point}"))]

    if args.stress:
        print("\n\n" + "#" * 74)
        print("# D. other parameter points, same scale")
        print("#" * 74)
        stress = [(args.point, summaries[0])]
        for pt in sorted(STRESS_POINTS):
            if pt == args.point:
                continue
            sub = argparse.Namespace(**vars(args))
            sub.point = pt
            sub.scaling = sub.stress = False
            print(f"\n>>> parameter point: {pt}  {STRESS_POINTS[pt]}", flush=True)
            stress.append((pt, report(*compare(sub, tag=f"_{pt}"))))
        print("\n  point      |dPE| nats   jitter nats   worst grad rel   1-cos")
        for name, s in stress:
            j = "n/a" if s["jitter"] is None else f"{s['jitter']:.3e}"
            print(f"  {name:9s}  {abs(s['dv']):.4e}   {j:>11s}   "
                  f"{s['worst_grad_rel']:.3e}        {s['one_minus_cos']:.2e}")
        worst = max(abs(s["dv"]) for _, s in stress)
        print(f"\n  worst potential error over {len(stress)} points: {worst:.4e} nats "
              f"({worst/HMC_ENERGY_SCALE:.3f}x the HMC energy scale)")

    if args.scaling:
        print("\n\n" + "#" * 74)
        print("# C. scaling with catalogue size")
        print("#" * 74)
        for nobs in (1000, 3000):
            sub = argparse.Namespace(**vars(args))
            sub.nobs = nobs
            sub.nsel = max(50_000, int(args.nsel * nobs / args.nobs))
            summaries.append(report(*compare(sub, tag=f"_n{nobs}")))
        summaries.sort(key=lambda s: s["nobs"])
        print("\n  nobs      nsel     |dPE| nats   jitter nats   worst grad rel")
        for s in summaries:
            j = "n/a" if s["jitter"] is None else f"{s['jitter']:.3e}"
            print(f"  {s['nobs']:6d}  {s['nsel']:9d}   {abs(s['dv']):.4e}   "
                  f"{j:>11s}   {s['worst_grad_rel']:.3e}")
        big = summaries[-1]
        if big["jitter"]:
            for target in (100_000, 1_000_000):
                print(f"    extrapolated jitter at nobs={target:,}: "
                      f"~{big['jitter']*target/big['nobs']:.2e} nats "
                      f"({big['jitter']*target/big['nobs']/HMC_ENERGY_SCALE:.3f}"
                      f"x the HMC energy scale)")

    print()


if __name__ == "__main__":
    main()

"""
Final check: time a single fully-jitted forward pass and gradient of the
REAL population-model log-likelihood + selection-function computation,
using your ACTUAL PE samples and injection data (not synthetic noise).

This is the number that actually predicts per-leapfrog-step wall time in
your real MCMC run. Multiply the 'grad' number by up to 127 (your
max_tree_depth=7 -> 2^7-1 steps) to compare against observed iteration time.
"""
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import sys
sys.path.append('../src/')
import intensity_models
from intensity_models import FlatwCDMCosmology

N_REPEATS = 10


def time_fwd_and_grad(name, scalar_fn, args, n_repeats=N_REPEATS):
    scalar_fn = jax.jit(scalar_fn)
    grad_fn = jax.jit(jax.grad(scalar_fn))

    # warmup / compile (excluded from timing)
    out = scalar_fn(*args)
    jax.block_until_ready(out)
    g = grad_fn(*args)
    jax.block_until_ready(g)

    fwd_times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = scalar_fn(*args)
        jax.block_until_ready(out)
        fwd_times.append(time.perf_counter() - t0)

    grad_times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        g = grad_fn(*args)
        jax.block_until_ready(g)
        grad_times.append(time.perf_counter() - t0)

    fwd_med = np.median(fwd_times) * 1000
    grad_med = np.median(grad_times) * 1000
    print(f"{name:60s} fwd: {fwd_med:8.2f} ms   grad: {grad_med:8.2f} ms")
    return fwd_med, grad_med


def load_true_vals(filename):
    tv = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                tv[key.strip()] = float(val.strip())

    # transformed params expected by model
    tv['dkappa'] = tv['kappa'] - tv['lam']
    tv['dmbhmax'] = tv['mbhmax'] - tv['mpisn']
    tv['log_flow'] = float(np.log(tv['flow']))
    tv['log_fpl'] = float(np.log(tv['fpl']))

    del tv['kappa']
    del tv['fpl']
    del tv['mbhmax']

    return {k: jnp.array(v) for k, v in tv.items()}


def main():
    # ---- load REAL data (adjust paths/slice as needed) ----
    pe_file = "../runs/mock_TC5_noevo_jul3/PE_noevo.h5"
    sel_file = "../runs/mock_TC5_noevo_jul3/sel_noevo.h5"

    pe_samples_mock = pd.read_hdf(pe_file, key='samples').iloc[6000:7000]
    m1s = np.asarray(pe_samples_mock['m1'].to_list())
    qs = np.asarray(pe_samples_mock['q'].to_list())
    dls = np.asarray(pe_samples_mock['dl'].to_list())
    pdraws = np.asarray(pe_samples_mock['pdraw'].to_list())
    pdraws = jnp.nan_to_num(pdraws, neginf=-1e30, posinf=1e30)
    print("PE array shapes (nevents, nsamples):", m1s.shape, qs.shape, dls.shape, pdraws.shape)

    sel_samples = pd.read_hdf(sel_file, key='true_parameters')
    ndraw = float(sel_samples['ndraw'].iloc[0])
    print("nsel (surviving injections):", len(sel_samples), " Ndraw (attempted):", ndraw)

    truth_params = load_true_vals('../scripts/pop_configs/mock_GWTC5_noevo.txt')
    extra_params = intensity_models.get_deterministic_parameters(truth_params)
    truth_params.update(extra_params)

    m1s_det = jnp.asarray(m1s)
    qs = jnp.asarray(qs)
    dls = jnp.asarray(dls)
    log_pdraw = jnp.asarray(pdraws)

    m1s_det_sel = jnp.asarray(sel_samples['m1d'].to_list())
    qs_sel = jnp.asarray(sel_samples['q'].to_list())
    dls_sel = jnp.asarray(sel_samples['dl'].to_list())
    pdraw_sel = jnp.asarray(sel_samples['pdraw_sel'].to_list())

    nobs, nsamp = m1s_det.shape
    print(f"\nnobs={nobs}, nsamp={nsamp}, nobs*nsamp={nobs*nsamp}, nsel={m1s_det_sel.shape[0]}\n")

    p = truth_params  # shorthand

    # ---- the real full gradient step, using REAL data throughout ----
    @jax.jit
    def full_gradient_step_real(mpisn):
        cosmo = FlatwCDMCosmology(p['h'], p['Om'], p['w'], zmax=p['zmax'])
        ld = intensity_models.LogDNDMDQDV(
            a=p['a'], b=p['b'], c=p['c'], mpisn=mpisn, mpisndot=p['mpisndot'],
            mbhmax=p['mbhmax'], sigma=p['sigma'], fpl=p['fpl'], beta=p['beta'],
            lam=p['lam'], kappa=p['kappa'], zp=p['zp'],
            mp_low=p['mp_low'], msigma_low=p['msigma_low'], flow=p['flow'],
            mbh_min=p['mbh_min'], delta_m=p['delta_m'],
            zmax=p['zmax'],
        )

        zs = cosmo.z_of_dL(dls)
        m1s_src = m1s_det / (1 + zs)
        log_wts = (ld(m1s_src, qs, zs) - log_pdraw - 2 * jnp.log1p(zs)
                   - jnp.log(cosmo.ddL_dz(zs)) + jnp.log(cosmo.dVCdz(zs)))
        # safe logsumexp guard for any fully -inf rows (shouldn't happen with
        # real data per our earlier check, but cheap insurance)
        log_wts_safe = jnp.where(jnp.isneginf(log_wts), -1e10, log_wts)
        log_like = jnp.sum(jax.scipy.special.logsumexp(log_wts_safe, axis=1) - jnp.log(nsamp))

        zs_sel = cosmo.z_of_dL(dls_sel)
        m1s_sel_src = m1s_det_sel / (1 + zs_sel)
        log_sel_wts = (ld(m1s_sel_src, qs_sel, zs_sel) - jnp.log(pdraw_sel)
                       - 2 * jnp.log1p(zs_sel) + jnp.log(cosmo.dVCdz(zs_sel)) - jnp.log(cosmo.ddL_dz(zs_sel)))
        log_sel_wts_safe = jnp.where(jnp.isneginf(log_sel_wts), -1e10, log_sel_wts)
        log_mu = jax.scipy.special.logsumexp(log_sel_wts_safe) - jnp.log(ndraw)

        return log_like - nobs * log_mu

    fwd_ms, grad_ms = time_fwd_and_grad(
        f"REAL full gradient step (nobs*nsamp={nobs*nsamp}, nsel={m1s_det_sel.shape[0]})",
        full_gradient_step_real, (p['mpisn'],)
    )

    print(f"\nfwd/grad ratio: {grad_ms/fwd_ms:.1f}x  (2-5x is normal; much higher suggests a NaN/masking issue)")
    print(f"Estimated per-iteration wall time at max_tree_depth=7 (up to 127 steps): "
          f"{grad_ms * 127 / 1000:.1f} s")
    print(f"Compare to your observed ~18-23 s/iteration.")


if __name__ == "__main__":
    main()
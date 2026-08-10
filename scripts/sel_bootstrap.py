"""Bootstrap the MC error of Delta log_mu_sel between sigma=0.0539 (truth)
and sigma=0.09 for the width15 and val2 selection half-sets, mimicking the
model's selection-weight computation exactly."""
import sys
sys.path.append('../src/')
import numpy as np
import pandas as pd
import jax.numpy as jnp
import intensity_models_fast as im


def pop_params_from_config(path):
    tv = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                k, v = line.split("=", 1)
                tv[k.strip()] = float(v.strip())
    return tv


def sel_log_wts(sel, tv, sigma):
    sample = dict(
        a=tv['a'], b=tv['b'], c=tv['c'], mpisn=tv['mpisn'],
        mpisndot=tv['mpisndot'], mbhmax=tv['mbhmax'], sigma=sigma,
        fpl=tv['fpl'], beta=tv['beta'], lam=tv['lam'], kappa=tv['kappa'],
        zp=tv['zp'], zmax=tv['zmax'], mbh_min=tv['mbh_min'],
        delta_m=tv['delta_m'], mp_low=tv['mp_low'],
        msigma_low=tv['msigma_low'], flow=tv['flow'],
    )
    cosmo = im.FlatwCDMCosmology(tv['h'], tv['Om'], tv['w'], zmax=tv['zmax'])
    log_dN = im.build_population_model(sample, use_low_bump=True, n_z=30,
                                       smooth_tail_edge=True)
    m1d = jnp.asarray(sel['m1d'].values)
    q = jnp.asarray(sel['q'].values)
    dl = jnp.asarray(sel['dl'].values)
    log_dl = jnp.log(dl)
    log1p_z, J = cosmo.z_and_log_jacobian(log_dl)
    opz = jnp.exp(log1p_z)
    z = opz - 1.0
    m1 = m1d / opz
    lw = (log_dN.call_from_logs(m1, jnp.log(m1), jnp.log(q), z, log1p_z)
          - jnp.log(jnp.asarray(sel['pdraw_sel'].values)) + J)
    return np.asarray(lw, dtype=np.float64)


def analyze(name, sel_file, pop_file, nobs, nboot=400, seed=7):
    sel_all = pd.read_hdf(sel_file, key='true_parameters')
    nhalf = int(np.round(len(sel_all) / 2))
    sel = sel_all.iloc[:nhalf]
    tv = pop_params_from_config(pop_file)

    lw1 = sel_log_wts(sel, tv, 0.0539)
    lw2 = sel_log_wts(sel, tv, 0.09)
    m = max(lw1.max(), lw2.max())
    w1 = np.exp(lw1 - m)
    w2 = np.exp(lw2 - m)
    n = len(w1)
    slope = np.log(w2.sum()) - np.log(w1.sum())
    print(f"{name}: nsel_half={n}  Delta log_mu (0.0539 -> 0.09) = {slope:.6f}")

    rng = np.random.default_rng(seed)
    slopes = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        s1 = w1[idx].sum(); s2 = w2[idx].sum()
        slopes[i] = np.log(s2) - np.log(s1)
    print(f"  bootstrap slope sd = {slopes.std():.6f}  -> nobs*sd = {nobs*slopes.std():.2f} nats")

    # slope excluding the first 10000 rows (the analyzed events live there)
    if n > 20000:
        w1x, w2x = w1[10000:], w2[10000:]
        sx = np.log(w2x.sum()) - np.log(w1x.sum())
        print(f"  slope excluding first 10000 rows = {sx:.6f} "
              f"(shift {sx-slope:+.6f}, nobs*shift = {nobs*(sx-slope):+.2f} nats)")
        w1o, w2o = w1[:10000], w2[:10000]
        so = np.log(w2o.sum()) - np.log(w1o.sum())
        print(f"  slope of first 10000 rows alone  = {so:.6f}")
    return slope, slopes.std()


if __name__ == "__main__":
    s_w15, sd_w15 = analyze(
        'width15', '../runs/endO5_fullcosmo_evo7/sel_width15.h5',
        'pop_configs/mock_O5_width15.txt', nobs=8000)
    s_v2, sd_v2 = analyze(
        'val2', '/mnt/home/misi/src/GW_pop_cosmo_inference/runs/endO5_val2/sel_noevo.h5',
        'pop_configs/mock_O5_noevo.txt', nobs=9000)
    print(f"\nslope difference (width15 - val2) = {s_w15 - s_v2:.6f}")
    print(f"combined bootstrap sd = {np.hypot(sd_w15, sd_v2):.6f}")

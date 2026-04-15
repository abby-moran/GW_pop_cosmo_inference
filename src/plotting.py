import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
import intensity_models
import pandas as pd
import seaborn as sns

def load_posterior_samples(nc_file, n_draws=None, seed=0):
    fit = az.from_netcdf(nc_file)
    posterior = fit.posterior.stack(sample=("chain", "draw"))

    params = {}
    lengths = []

    for k, v in posterior.data_vars.items():
        arr = v.values
        if arr.ndim == 1:
            params[k] = arr
            lengths.append(len(arr))

    N = min(lengths)
    params = {k: v[:N] for k, v in params.items()}

    if n_draws is not None and n_draws < N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_draws, replace=False)
        params = {k: v[idx] for k, v in params.items()}

    return params

def compute_ppd_grids(params, m1_grid, q_grid, z_grid):
    N = len(next(iter(params.values())))

    p_m1 = np.zeros((N, len(m1_grid)))
    p_q  = np.zeros((N, len(q_grid)))
    p_z  = np.zeros((N, len(z_grid)))

    for i in range(N):
        #for each draw of the posterior, calcualte the probabilities of q, m, z at each point on the 
        # grid, slicing across reference values of the other parameters
        sample = {k: v[i] for k, v in params.items()}

        sample["mbhmax"] = sample["mpisn"] + sample["dmbhmax"]
        sample["fpl"] = np.exp(sample["log_fpl"])
        sample["kappa"] = sample["lam"] + sample["dkappa"]

        cosmo = intensity_models.FlatwCDMCosmology(
            sample["h"], sample["Om"], sample["w"], zmax=sample["zmax"])
        log_dN = intensity_models.build_population_model(sample)

        # m1
        log_vals = log_dN(m1_grid, 0.5, 0.2)
        p = np.exp(log_vals - np.max(log_vals))
        p_m1[i] = p / np.trapezoid(p, m1_grid)

        # q
        log_vals = log_dN(20.0, q_grid, 0.2)
        p = np.exp(log_vals - np.max(log_vals))
        p_q[i] = p / np.trapezoid(p, q_grid)

        # z
        log_vals = (log_dN(20.0, 0.5, z_grid)+ np.log(cosmo.dVCdz(z_grid))
            - np.log(cosmo.ddL_dz(z_grid)) - 2*np.log1p(z_grid))
        p = np.exp(log_vals - np.max(log_vals))
        p_z[i] = p / np.trapezoid(p, z_grid)

    return p_m1, p_q, p_z

def compute_or_load_ppd(cache_file, compute_fn):
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["p_m1"], data["p_q"], data["p_z"]

    p_m1, p_q, p_z = compute_fn()
    np.savez(cache_file, p_m1=p_m1, p_q=p_q, p_z=p_z)
    return p_m1, p_q, p_z   

def downsample_per_event(arr, n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for row in arr:
        if len(row) > n:
            idx = rng.choice(len(row), size=n, replace=False)
            out.append(row[idx])
        else:
            out.append(row)
    return np.concatenate(out)

def detected_ppd_from_existing(params, m1s_det_sel, qs_sel, dls_sel, pdraw_sel):
    cosmo_ref = intensity_models.FlatwCDMCosmology(0.7, 0.3, -1.0, zmax=5.0)
    N = len(next(iter(params.values())))

    zs_sel = cosmo_ref.z_of_dL(dls_sel)
    m1s_src = m1s_det_sel / (1 + zs_sel)

    weights = []

    for i in range(N):
        sample = {k: v[i] for k, v in params.items()}

        sample["mbhmax"] = sample["mpisn"] + sample["dmbhmax"]
        sample["fpl"] = np.exp(sample["log_fpl"])
        sample["kappa"] = sample["lam"] + sample["dkappa"]

        cosmo = intensity_models.FlatwCDMCosmology(
            sample["h"], sample["Om"], sample["w"], zmax=sample["zmax"])

        log_dN = intensity_models.build_population_model(sample)
        # weight by the selection integral for each sample
        log_w = (log_dN(m1s_src, qs_sel, zs_sel) - np.log(pdraw_sel)- 2*np.log1p(zs_sel)
            + np.log(cosmo.dVCdz(zs_sel)) - np.log(cosmo.ddL_dz(zs_sel)))

        w = np.exp(log_w - np.max(log_w))
        w /= np.sum(w)

        weights.append(w)

    return np.array(weights)
def plot_band_with_data(ax, grid, arr, data, label=None):
    lo, med, hi = np.percentile(arr, [5, 50, 95], axis=0)

    ax.fill_between(grid, lo, hi, alpha=0.3)
    ax.plot(grid, med, label=label)

    # histogram of observed data
    ax.hist(data, bins=50, density=True, histtype='step', linewidth=1.5, label='data')

def make_ppd_plot(nc_file, pe_samples, m1s_det_sel=None, qs_sel=None, dls_sel=None, detected_ppd=None,
                  cache_file="ppd_cache.npz", ndraws=300):

    params = load_posterior_samples(nc_file, n_draws=ndraws)

    m1_grid = np.linspace(5, 200, 250)
    q_grid  = np.linspace(0.2, 1.0, 250)
    z_grid  = np.linspace(0.01, 2.0, 250)

    def compute():
        return compute_ppd_grids(params, m1_grid, q_grid, z_grid)

    p_m1, p_q, p_z = compute_or_load_ppd(cache_file, compute)

    #PE samples
    m1s, qs, dls = pe_samples

    m1_obs = downsample_per_event(m1s).flatten()
    q_obs  = downsample_per_event(qs).flatten()

    cosmo_ref = intensity_models.FlatwCDMCosmology(0.7, 0.3, -1.0, zmax=5.0)
    z_obs = cosmo_ref.z_of_dL(dls.flatten())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].set_yscale('log')
    axes[0].set_xlabel('m1')
    axes[0].set_ylabel('p')
    axes[1].set_xlabel('q')
    axes[2].set_xlabel('z')

    plot_band_with_data(axes[0], m1_grid, p_m1, m1_obs, label="intrinsic")
    plot_band_with_data(axes[1], q_grid, p_q, q_obs, label="intrinsic")
    plot_band_with_data(axes[2], z_grid, p_z,z_obs,label="intrinsic")

    # PPD after selection effects ('detected')
    if detected_ppd is not None:

        W = detected_ppd  # (N_post, N_events)

        def weighted_hist(ax, grid, data):
            curves = []
            for w in W:
                w = w / np.sum(w)
                h, _ = np.histogram(data, bins=grid, weights=w, density=True)
                curves.append(h)
            curves = np.array(curves)

            med = np.median(curves, axis=0)
            ax.plot(grid[:-1], med, "k--", label="detected PPD")

        weighted_hist(axes[0], m1_grid, m1s_det_sel)
        weighted_hist(axes[1], q_grid,  qs_sel)
        weighted_hist(axes[2], z_grid,  cosmo_ref.z_of_dL(dls_sel))

    for ax in axes:
        ax.legend()
    axes[0].set_xlim(5, 220)
    axes[1].set_xlim(.15, 1)
    axes[2].set_xlim(.0001, 2)

    plt.tight_layout()
    plt.show()

def load_true_vals(filename):
    tv = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # skip blanks/comments
                key, val = line.split("=", 1)
                tv[key.strip()] = float(val.strip())

    tv['dkappa'] = tv['kappa'] - tv['lam']
    tv['log_fpl'] = np.log(tv['fpl'])
    tv['dmbhmax'] = tv['mbhmax'] - tv['mpisn']

    return tv

def plot_corner(idatas, true_vals, axes_labels={'mpisn': r'$m_\mathrm{PISN}$', 'mbhmax': r'$m_\mathrm{BH,max}$', 'beta': 'beta','b': 'b'}, run_names=None):
    """
    idatas is an array of length n dataframes
    """
    sns.set_palette('colorblind')
    n=len(idatas)
    dfs_named=dict()
    for i in range(n):
        dfs_named['df'+str(i)]=pd.DataFrame({axes_labels[k]: idatas[i].posterior[k].values.flatten() for k in axes_labels.keys()})
        if run_names is not None:
            dfs_named['df'+str(i)]['run']=run_names[i]

    plot_vars = list(axes_labels.values())  # the actual parameters being plotted 

    df_all=pd.concat(dfs_named, ignore_index=True)
    if run_names is not None:
        pg = sns.PairGrid(df_all, diag_sharey=False, hue='run')
    else:
        pg = sns.PairGrid(df_all, diag_sharey=False)
    pg.map_diag(sns.kdeplot, common_norm=False, levels=[0.1, 0.5, 0.9])
    if len(plot_vars) > 1:
        pg.map_lower(sns.kdeplot, common_norm=False, levels=[0.1, 0.5, 0.9])
    #pg.map_lower(sns.kdeplot, common_norm=False)
    if len(axes_labels) == 1 and run_names is not None:
        var = list(axes_labels.values())[0]
        sns.kdeplot(data=df_all, x=var, hue="run", common_norm=False, levels=[0.1, 0.5, 0.9])
        if true_vals is not None:
            plt.axvline(true_vals[list(axes_labels.keys())[0]], color="red", ls="--")
        plt.tight_layout()
        return
    
    for i, j in zip(*np.triu_indices_from(pg.axes, 1)):
        pg.axes[i, j].set_visible(False)
    if true_vals is not None: #common_norm=Flase?
        for i, xvar in enumerate(plot_vars):
            for j, yvar in enumerate(plot_vars):
                ax = pg.axes[i, j]
                if ax is None:
                    continue
                if i == j:
                    val = true_vals[list(axes_labels.keys())[i]]
                    ax.axvline(val, color='red', linestyle='--', lw=1.2)
                elif i > j:
                    xval = true_vals[list(axes_labels.keys())[j]]
                    yval = true_vals[list(axes_labels.keys())[i]]
                    ax.axvline(xval, color='red', linestyle='--', lw=1.0)
                    ax.axhline(yval, color='red', linestyle='--', lw=1.0)
    pg.add_legend(loc='upper center', fontsize=24)
    
    for i, row_axes in enumerate(pg.axes):
        for j, ax in enumerate(row_axes):
            if ax is not None:
                ax.xaxis.label.set_size(20)  # x-axis label size
                ax.yaxis.label.set_size(20) 
    plt.tight_layout()

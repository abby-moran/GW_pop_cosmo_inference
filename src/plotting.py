import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
# The fast model is the one run_inf.py samples with; it is also the one whose
# build_population_model takes the static `pairing` flag threaded below.
import intensity_models_fast as intensity_models
import pandas as pd
import seaborn as sns
import jax
import jax.numpy as jnp

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

def canonical_params(params):
    """Fill the physical keys build_population_model reads (mbhmax, kappa,
    fpl, flow, Om) from whatever coordinates a saved chain carries -- the same
    derivations as intensity_models_fast.get_deterministic_parameters, applied
    to flat posterior arrays.  Keys already present are left alone."""
    p = dict(params)
    if "mpisn" not in p and "mpisn_ref" in p:
        zp = p.get("zpivot", 0.75)
        p["mpisn"] = p["mpisn_ref"] - p.get("mpisndot", 0.0) * zp / (1.0 + zp)
    if "mbhmax" not in p and "mpisn" in p and "dmbhmax" in p:
        p["mbhmax"] = p["mpisn"] + p["dmbhmax"]
    if "kappa" not in p and "lam" in p and "dkappa" in p:
        p["kappa"] = p["lam"] + p["dkappa"]
    if "fpl" not in p:
        # Tail height: always the join-height ratio r anchored at zref, under
        # whichever name the prior sampled it.
        if "r" in p:
            p["fpl"] = p["r"]
        elif "log_r" in p:
            p["fpl"] = np.exp(p["log_r"])
        elif "log_fpl" in p:
            p["fpl"] = np.exp(p["log_fpl"])
    if "flow" not in p:
        if "log_flow" in p:
            p["flow"] = np.exp(p["log_flow"])
        elif "log_fpeak" in p and "msigma_low" in p:
            p["flow"] = np.exp(p["log_fpeak"]) * p["msigma_low"]
    if "Om" not in p and "Omh2" in p and "h" in p:
        p["Om"] = p["Omh2"] / p["h"] ** 2
    return p


def pairing_from_nc(nc_file):
    """Pairing convention a saved run used, for threading into the draw
    functions below so they rebuild the model the chain was sampled under."""
    from utils import pairing_from_attrs
    return pairing_from_attrs(az.from_netcdf(nc_file).posterior.attrs)


def rate_m1_for_one_draw(sample, m1_grid, q_grid, z0, pairing="q"):

    model = intensity_models.build_population_model(sample, pairing=pairing)

    m1_grid = jnp.asarray(m1_grid)
    q_grid = jnp.asarray(q_grid)

    logvals = jax.vmap(
        lambda m1: jax.vmap(lambda q: model(m1, q, z0))(q_grid)
    )(m1_grid)

    vals = jnp.exp(logvals)

    # integrate over q → result shape (m1,)
    return jnp.trapezoid(vals, q_grid, axis=1)
def _eval_m1_q(model, m1_grid, q_grid, z0):
    return jax.vmap(
        lambda m1: jax.vmap(lambda q: model(m1, q, z0))(q_grid)
    )(m1_grid)

def rate_z_for_one_draw(sample, m1_grid, q_grid, z_grid, pairing="q"):
    """
    Compute dR/dz by marginalizing over m1 and q.
    """

    model = intensity_models.build_population_model(sample, pairing=pairing)

    m1_grid = jnp.asarray(m1_grid)
    q_grid = jnp.asarray(q_grid)
    z_grid = jnp.asarray(z_grid)

    def eval_qz(q, z):
        return model(m1_grid, q, z)

    logvals = jax.vmap(jax.vmap(eval_qz, in_axes=(None, 0)), in_axes=(0, None))(q_grid, z_grid)

    vals = jnp.exp(logvals)

    # integrate m1, q, leave z
    int_m1 = jnp.trapezoid(vals, m1_grid, axis=2)
    dR_dz = jnp.trapezoid(int_m1, q_grid, axis=0)

    return dR_dz


def dR_dm1_at_z_for_one_draw(sample, m1_grid, q_grid, z0, pairing="q"):
    """
    Differential rate dR/dm1 at fixed z=z0, marginalizing over q only.
    """
    model = intensity_models.build_population_model(sample, pairing=pairing)

    m1_grid = jnp.asarray(m1_grid)
    q_grid = jnp.asarray(q_grid)
    z0 = jnp.asarray(z0)

    # Broadcast evaluation on the (nm1, nq) plane rather than nested scalar
    # vmaps: the fast model's __call__ derives its internal grid shapes from
    # the argument shapes and does not vmap over scalar m1 / q.
    logvals = model(m1_grid[:, None], q_grid[None, :], z0)   # (nm1, nq)
    return jnp.trapezoid(jnp.exp(logvals), q_grid, axis=1)


def pm1_at_z_for_one_draw(sample, m1_grid, q_grid, z0, pairing="q"):
    """
    Normalized p(m1 | z=z0), proportional to dR/dm1 at fixed z0.
    """
    dR_dm1 = dR_dm1_at_z_for_one_draw(sample, m1_grid, q_grid, z0, pairing=pairing)
    return dR_dm1 / jnp.trapezoid(dR_dm1, m1_grid)

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

def pq_for_one_draw(sample, m1_grid, q_grid, z_grid=None, pairing="q"):
    """
    Compute p(q) for one hyperposterior draw by marginalizing over m1
    and optionally over z.
    """

    model = intensity_models.build_population_model(sample, pairing=pairing)

    m1_grid = jnp.asarray(m1_grid)
    q_grid  = jnp.asarray(q_grid)
    z_grid  = jnp.asarray(z_grid)

    # evaluate log density on full grid:
    # shape = (nq, nz, nm1)

    def eval_qz(q, z):
        return model(m1_grid, q, z)

    eval_z = jax.vmap(eval_qz, in_axes=(None, 0))      # over z
    eval_q = jax.vmap(eval_z, in_axes=(0, None))       # over q

    logvals = eval_q(q_grid, z_grid)

    # finite mask
    finite = jnp.isfinite(logvals)

    # replace -inf with large negative
    safe = jnp.where(finite, logvals, -1e30)

    # stabilize exponentials along m1 axis
    maxv = jnp.max(safe, axis=2, keepdims=True)

    vals = jnp.exp(safe - maxv)
    vals = jnp.where(finite, vals, 0.0)

    # integrate over m1, z
    int_m1 = jnp.trapezoid(vals, m1_grid, axis=2)
    int_z = jnp.trapezoid(int_m1, z_grid, axis=1)

    # normalize pq by q integral 
    pq = int_z / jnp.trapezoid(int_z, q_grid)

    return pq

def compute_pq_draws(params, m1_grid, q_grid, z_grid=None, n_draws=100, rng=None, pairing="q"):
    rng = np.random.default_rng(rng)
    N = len(next(iter(params.values())))
    draw_idx = rng.choice(N, size=min(n_draws, N), replace=False)

    pq_draws = []

    for i in draw_idx:
        sample = {k: v[i] for k, v in params.items()}
        pq = pq_for_one_draw(sample, m1_grid, q_grid, z_grid=z_grid, pairing=pairing)
        pq_draws.append(pq)

    return np.array(pq_draws), draw_idx

def plot_band_with_data(ax, grid, arr, data=None, label=None):
    lo, med, hi = np.percentile(arr, [5, 50, 95], axis=0)

    ax.fill_between(grid, lo, hi, alpha=0.3)
    ax.plot(grid, med, label=label)

    # histogram of observed data
    if data is not None:
        ax.hist(data, bins=50, density=True, histtype='step', linewidth=1.5, label='data')

def make_ppd_plot(nc_file, pe_samples=None, m1s_det_sel=None, qs_sel=None, dls_sel=None, detected_ppd=None,
                  cache_file="ppd_cache.npz", ndraws=300):

    params = canonical_params(load_posterior_samples(nc_file, n_draws=ndraws))
    pairing = pairing_from_nc(nc_file)

    m1_grid = np.linspace(5, 200, 200)
    q_grid  = np.linspace(0.2, 1.0, 200)
    z_grid  = np.linspace(0.01, 2.0, 200)

    def compute():
        # Marginal (not fixed-slice) unit-area densities per posterior draw,
        # through the per-draw helpers above, rebuilt with the pairing the
        # chain was sampled under:
        #   p_m1 = p(m1 | z=0.2)   (integrated over q)
        #   p_q  = p(q)            (integrated over m1 and z)
        #   p_z  = R(z)/int R(z)dz (dN/dVc dt, integrated over m1 and q)
        n = len(next(iter(params.values())))
        p_m1 = np.zeros((n, len(m1_grid)))
        p_z = np.zeros((n, len(z_grid)))
        for i in range(n):
            sample = {k: v[i] for k, v in params.items()}
            p_m1[i] = np.asarray(pm1_at_z_for_one_draw(
                sample, m1_grid, q_grid, 0.2, pairing=pairing))
            rz = np.asarray(rate_z_for_one_draw(
                sample, m1_grid, q_grid, z_grid, pairing=pairing))
            p_z[i] = rz / np.trapezoid(rz, z_grid)
        p_q, idx = compute_pq_draws(params, m1_grid, q_grid, z_grid=z_grid,
                                    n_draws=n, rng=0, pairing=pairing)
        p_q = p_q[np.argsort(idx)]      # same draw order as p_m1 / p_z
        return p_m1, p_q, p_z

    p_m1, p_q, p_z = compute_or_load_ppd(cache_file, compute)
    cosmo_ref = intensity_models.FlatwCDMCosmology(0.7, 0.3, -1.0, zmax=5.0)

    #PE samples
    if pe_samples is not None:
        m1s, qs, dls = pe_samples

        m1_obs = downsample_per_event(m1s).flatten()
        q_obs  = downsample_per_event(qs).flatten()

        z_obs = cosmo_ref.z_of_dL(dls.flatten())
    else:
        m1_obs = None
        q_obs = None
        z_obs = None    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].set_yscale('log')
    axes[0].set_xlabel('m1')
    axes[0].set_ylabel('p')
    axes[1].set_xlabel('q')
    axes[2].set_xlabel('z')

    plot_band_with_data(axes[0], m1_grid, p_m1,data= m1_obs, label="intrinsic")
    plot_band_with_data(axes[1], q_grid, p_q, data=q_obs, label="intrinsic")
    plot_band_with_data(axes[2], z_grid, p_z,data=z_obs,label="intrinsic")

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
    tv['dmbhmax'] = tv['mbhmax'] - tv['mpisn']
    if 'flow' in tv:
        tv['log_flow'] = float(np.log(tv['flow']))
        if 'msigma_low' in tv:
            tv['log_fpeak'] = float(tv['log_flow'] - np.log(tv['msigma_low']))
    tv['log_fpl'] = float(np.log(tv['fpl']))
    if 'Omh2' not in tv and 'Om' in tv and 'h' in tv:
        tv['Omh2'] = tv['Om'] * tv['h'] ** 2

    return tv

def plot_corner(idatas, true_vals, axes_labels={'mpisn': r'$m_\mathrm{PISN}$', 'mbhmax': r'$m_\mathrm{BH,max}$', 'beta': 'beta', 'b': 'b'}, run_names=None, save_fig=None, figsize=(19,19)):
    """
    idatas is an array of length n dataframes
    """
    sns.set_palette('colorblind')
    n=len(idatas)
    dfs_named=dict()

    for i in range(n):
        data={}
        for k in axes_labels.keys():
            if k in idatas[i].posterior:
                data[axes_labels[k]]=idatas[i].posterior[k].values.flatten()
            else:
                print(f"Warning: '{k}' not found in dataset {i}; skipping this variable for that dataset.")
                data[axes_labels[k]]=np.full(len(idatas[i].posterior[list(idatas[i].posterior.data_vars)[0]].values.flatten()), np.nan)
        dfs_named['df'+str(i)]=pd.DataFrame(data)
        if run_names is not None:
            dfs_named['df'+str(i)]['Run']=run_names[i]

    plot_vars=list(axes_labels.values())
    df_all=pd.concat(dfs_named, ignore_index=True)

    if run_names is not None:
        pg=sns.PairGrid(df_all, diag_sharey=False, hue='Run')
    else:
        pg=sns.PairGrid(df_all, diag_sharey=False)

    pg.figure.set_size_inches(figsize[0], figsize[1])

    pg.map_diag(sns.kdeplot, lw=2, common_norm=False, levels=[0.1, 0.5, 0.9])
    if len(plot_vars)>1:
        pg.map_lower(sns.kdeplot, common_norm=False, linewidths=2, levels=[0.1, 0.5, 0.9])

    if len(axes_labels)==1 and run_names is not None:
        var=list(axes_labels.values())[0]
        sns.kdeplot(data=df_all, x=var, hue="Run", common_norm=False, lw=2, levels=[0.1, 0.5, 0.9])
        if true_vals is not None:
            plt.axvline(true_vals[list(axes_labels.keys())[0]], color="red", ls="--")
        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(save_fig, bbox_inches='tight')
        return

    for i,j in zip(*np.triu_indices_from(pg.axes,1)):
        pg.axes[i,j].set_visible(False)

    if true_vals is not None:
        for i,xvar in enumerate(plot_vars):
            for j,yvar in enumerate(plot_vars):
                ax=pg.axes[i,j]
                if ax is None:
                    continue
                if i==j:
                    val=true_vals[list(axes_labels.keys())[i]]
                    ax.axvline(val, color='red', linestyle='--', lw=1.8, alpha=0.7)
                elif i>j:
                    xval=true_vals[list(axes_labels.keys())[j]]
                    yval=true_vals[list(axes_labels.keys())[i]]
                    ax.axvline(xval, color='red', linestyle='--', lw=1.8, alpha=0.7)
                    ax.axhline(yval, color='red', linestyle='--', lw=1.8, alpha=0.7)

    for i,row_axes in enumerate(pg.axes):
        for j,ax in enumerate(row_axes):
            if ax is not None:
                xvar=plot_vars[j]
                yvar=plot_vars[i]
                x=df_all[xvar].dropna()
                xmin,xmax=x.min(),x.max()
                xpad=0.05*(xmax-xmin)
                ax.set_xlim(xmin-xpad,xmax+xpad)

                if i!=j:
                    y=df_all[yvar].dropna()
                    ymin,ymax=y.min(),y.max()
                    ypad=0.05*(ymax-ymin)
                    ax.set_ylim(ymin-ypad,ymax+ypad)

                ax.xaxis.label.set_size(28)
                ax.yaxis.label.set_size(28)
                ax.tick_params(axis='both', which='major', labelsize=25)

    if run_names is not None:
        pg.add_legend(loc='upper center')
        pg._legend.get_title().set_fontsize(32)
        for text in pg._legend.get_texts():
            text.set_fontsize(32)

    plt.tight_layout()

    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches='tight')
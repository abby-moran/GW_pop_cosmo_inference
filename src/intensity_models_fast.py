"""
Replacement for intensity_models.py with these changes:

1. No ``jnp.interp`` / ``searchsorted`` - grids are uniform so we can use the grid indicies
2. One fused cosmology lookup instead of three. 
3. The PISN mco integral is done as a max-subtracted trapezoid in linear space (one ``exp`` over the big grid)
4. When ``mpisndot`` is a fixed 0 the PISN grid is z-independent, so we collapse 2D => 1D interpolation
5. The per-event ``logsumexp`` and the ``neff`` diagnostic share one max-subtracted pass
6. ``get_deterministic_parameters`` is no longer wrapped in ``jax.jit``. Lets deterministic params reach output
7. The mass function is tabulated once per likelihood call, just do table lookups per sample 
"""
from astropy.cosmology import Planck18
import astropy.units as u
import dataclasses
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import jax.scipy.stats as jsst
from jax import lax
import numpy as np
import numpyro
import numpyro.distributions as dist
from utils import jnp_cumtrapz, sample_parameters_from_dict, log_expit
from jax.scipy.ndimage import map_coordinates
from functools import partial

_LOG_2PI = float(np.log(2 * np.pi))

# Static-value helpers: if the prior is just a number (fixed), return as a float
def _static_value(x):
    """Return x as a float if it is a concrete scalar, else None."""
    if isinstance(x, jax.core.Tracer):
        return None
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.shape == () and np.issubdtype(arr.dtype, np.number):
        return float(arr)
    return None

def _is_static_zero(x):
    v = _static_value(x)
    return v is not None and v == 0.0

# Log-uniform / log1p-uniform axes with closed-form index lookup
class _LogAxis:
    """A grid uniform in log(x). ``frac_index`` maps log(x) to a fractional grid index by arithmetic (no binary search)."""
    def __init__(self, lo, hi, n):
        self.n = int(n)
        self.log_lo = float(np.log(lo))
        self.log_hi = float(np.log(hi))
        # Build the grid the same way linspace+exp does so that grid[-1] is `hi'
        self.log_grid = np.linspace(self.log_lo, self.log_hi, self.n)
        self.grid = jnp.asarray(np.exp(self.log_grid))
        self.inv_dlog = (self.n - 1) / (self.log_hi - self.log_lo)

    def frac_index(self, log_x):
        return jnp.clip((log_x - self.log_lo) * self.inv_dlog, 0.0, self.n - 1.0)

    def cell_and_frac(self, log_x):
        """(cell index, within-cell weight), interpolating linearly in log(x)."""
        t = self.frac_index(log_x)
        i0f = jnp.floor(t)
        return i0f.astype(jnp.int32), t - i0f


class _Log1pAxis:
    """A grid uniform in log1p(z) from 0 to zmax (same nodes as the original's ``expm1(linspace(log 1, log(1+zmax), n))``)."""

    def __init__(self, zmax, n):
        self.n = int(n)
        self.log1p_hi = jnp.log1p(zmax)
        self.log1p_grid = jnp.linspace(0.0, self.log1p_hi, self.n)
        self.grid = jnp.expm1(self.log1p_grid)
        # zmax may be sampled, so keep this as a traced scalar.
        self.inv_dlog = (self.n - 1) / self.log1p_hi

    def cell_and_frac(self, z, log1p_z):
        """(cell index, within-cell weight) reproducing ``jnp.interp(z, z_array, arange(n))`` exactly.

        The cell index still comes from log1p(z) in closed form, but the weight is computed linearly in z.  
        This axis has only ~30 nodes spanning z in [0, zmax], so linear-in-z and linear-in-log1p(z) differ by up to ~1e-2 in
        log-density, well above float32 noise.
        """
        t = jnp.clip(log1p_z * self.inv_dlog, 0.0, self.n - 1.0)
        i0 = jnp.floor(t).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, self.n - 1)
        z0 = self.grid[i0]
        z1 = self.grid[i1]
        dz = z1 - z0
        safe_dz = jnp.where(dz > 0, dz, 1.0)
        frac = jnp.where(dz > 0, (z - z0) / safe_dz, 0.0)
        return i0, jnp.clip(frac, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Scatter-contention mitigation: for Reverse-mode AD, keep R identical copies of the table so we can
# send neighboring points to different copies
# ---------------------------------------------------------------------------
SCATTER_REPLICAS = 32

# Below this many points the scatter is not contended enough to be worth the extra index arithmetic.
_REPLICATE_MIN_SIZE = 1 << 16


def _replicas_for(shape, R=None):
    R = SCATTER_REPLICAS if R is None else R
    if R <= 1 or len(shape) == 0:
        return 1
    if int(np.prod(shape)) < _REPLICATE_MIN_SIZE:
        return 1
    return int(R)

def _replica_offset(shape, R, n_table):
    """Per-point offset into the replicated table.  Varying along the last axis means adjacent lanes in a 
    warp hit different replicas, which is where the conflicts would otherwise be."""
    iota = lax.broadcasted_iota(jnp.int32, shape, len(shape) - 1)
    # np.int32 rather than Python int so this also works under jax_enable_x64
    return lax.rem(iota, np.int32(R)) * np.int32(n_table)


def _lerp1d(table, t, n, R=None):
    """Linear interpolation into ``table`` (1-D, length n) at fractional indices ``t``.  
    t must be clipped to [0, n-1], to reproduce jnp.interp's map_coordinates(mode='nearest') edge clamping."""
    i0f = jnp.floor(t)
    return _gather_lerp1d(table, i0f.astype(jnp.int32), t - i0f, n, R=R)


def _gather_lerp1d(table, i0, frac, n, R=None):
    i1 = jnp.minimum(i0 + 1, n - 1)
    flat = table
    R = _replicas_for(jnp.shape(i0), R)
    if R > 1:
        flat = jnp.broadcast_to(table, (R, n)).reshape(-1)
        off = _replica_offset(jnp.shape(i0), R, n)
        i0 = i0 + off
        i1 = i1 + off
    a = flat[i0]
    return a + frac * (flat[i1] - a)

def _gather_lerp2d(table, im0, fm, iz0, fz, nm, nz, R=None):
    """Bilinear interpolation into ``table`` of shape (nz, nm).  ``m`` is the
    fast axis so the four gathers hit adjacent memory."""
    im1 = jnp.minimum(im0 + 1, nm - 1)
    iz1 = jnp.minimum(iz0 + 1, nz - 1)
    ntot = nm * nz
    flat = table.reshape(-1)
    off = 0
    R = _replicas_for(jnp.shape(im0), R)
    if R > 1:
        flat = jnp.broadcast_to(flat, (R, ntot)).reshape(-1)
        off = _replica_offset(jnp.shape(im0), R, ntot)
    b0 = iz0 * nm + off
    b1 = iz1 * nm + off
    g00 = flat[b0 + im0]
    g10 = flat[b0 + im1]
    g01 = flat[b1 + im0]
    g11 = flat[b1 + im1]
    lo = g00 + fm * (g10 - g00)
    hi = g01 + fm * (g11 - g01)
    return lo + fz * (hi - lo)


# ---------------------------------------------------------------------------
# Scatter-free VJPs for parameter-dependent table lookups.
# Replica trick still is slow on d(potential)/d(table) => with tangent tables U_k = dT/dtheta_k since we constructed the table
# from theta_k. Compute one per likleihood call, and its just k extra table builds of ~250k entries each,
# Gradient is mathemtically identical to ordinary reverse mode
# ---------------------------------------------------------------------------

def _linearize_table(build, params):
    """(T, U, theta) for ``T = build(*params)``.

    ``params`` may mix traced scalars and compile-time constants.  ``theta``
    is the stacked vector of the traced ones and ``U = dT/dtheta`` has shape
    ``T.shape + (k,)``.  With no traced parameters, returns (T, None, None)
    (the table is a constant and plain lookups have no backward scatter).
    """
    traced_idx = [i for i, p in enumerate(params) if isinstance(p, jax.core.Tracer)]
    if not traced_idx:
        return build(*params), None, None
    dtype = jnp.result_type(*([params[i] for i in traced_idx] + [jnp.float32]))
    theta = jnp.stack([jnp.asarray(params[i], dtype) for i in traced_idx])

    def f(th):
        full = list(params)
        for j, i in enumerate(traced_idx):
            full[i] = th[j]
        return build(*full)

    T, jvp = jax.linearize(f, theta)
    # Channel-first layout (k, *T.shape): each tangent channel is then a
    # contiguous table, which is what the Pallas backward kernels gather from
    # (the 2-D mass tangents need no transpose at all).
    U = jax.vmap(jvp)(jnp.eye(len(traced_idx), dtype=dtype))
    return T, U, theta


@jax.custom_vjp
def _sf_lookup1d(theta, T, U, t):
    """Multi-channel linear interpolation ``T[t]`` with the table-value
    gradient routed through ``theta``.

    T: (n, C); U: (k, n, C); t: fractional index, any shape, already clipped
    to [0, n-1].  Returns (t.shape..., C).
    """
    del theta, U
    n = T.shape[0]
    i0 = jnp.floor(t).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, n - 1)
    a = T[i0]
    return a + (t - i0)[..., None] * (T[i1] - a)


def _sf_lookup1d_fwd(theta, T, U, t):
    return _sf_lookup1d(theta, T, U, t), (theta, T, U, t)

# The theta_bar accumulation is the performance-critical piece of the
# scatter-free backward. 
# Small Pallas kernel: each program loads a block of points once, keeps the k partial sums in registers, 
# and gathers tangent-table rows that stay hot in L2  (the tables are ~1 MB per channel).
_SF_BLOCK = 1024
_SF_CHUNK = 1 << 21

def _use_pallas():
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False

def _sf_theta_kernel_2d(tm_ref, tz_ref, g_ref, U_ref, o_ref, *, npts, K, NZ, NM):
    """theta_bar for one bilinear lookup: each program streams a block of
    points once, holds the K partial sums in registers, and gathers  tangent-table rows that stay hot in L2."""
    from jax.experimental import pallas as pl
    NT = NZ * NM
    pid = pl.program_id(0)
    offs = pid * _SF_BLOCK + jnp.arange(_SF_BLOCK, dtype=jnp.int32)
    mask = offs < npts
    tm_v = pl.load(tm_ref, (offs,), mask=mask, other=0.0)
    tz_v = pl.load(tz_ref, (offs,), mask=mask, other=0.0)
    g_v = pl.load(g_ref, (offs,), mask=mask, other=0.0)
    im0 = jnp.floor(tm_v).astype(jnp.int32)
    im1 = jnp.minimum(im0 + 1, NM - 1)
    iz0 = jnp.floor(tz_v).astype(jnp.int32)
    iz1 = jnp.minimum(iz0 + 1, NZ - 1)
    fm = tm_v - im0
    fz = tz_v - iz0
    b00 = iz0 * NM + im0
    b10 = iz0 * NM + im1
    b01 = iz1 * NM + im0
    b11 = iz1 * NM + im1
    w00 = (1 - fm) * (1 - fz) * g_v
    w10 = fm * (1 - fz) * g_v
    w01 = (1 - fm) * fz * g_v
    w11 = fm * fz * g_v
    for k in range(K):
        base = k * NT
        s = (jnp.sum(w00 * pl.load(U_ref, (base + b00,)))
             + jnp.sum(w10 * pl.load(U_ref, (base + b10,))) + jnp.sum(w01 * pl.load(U_ref, (base + b01,)))
             + jnp.sum(w11 * pl.load(U_ref, (base + b11,))))
        pl.store(o_ref, (pid, k), s)

def _sf_theta_kernel_1d(t_ref, g_ref, U_ref, o_ref, *, npts, K, C, N):
    # g_ref holds the cotangent in its natural (npts, C) row-major layout;
    # element (p, c) sits at p*C + c, so no transpose copy is needed.
    from jax.experimental import pallas as pl
    pid = pl.program_id(0)
    offs = pid * _SF_BLOCK + jnp.arange(_SF_BLOCK, dtype=jnp.int32)
    mask = offs < npts
    t_v = pl.load(t_ref, (offs,), mask=mask, other=0.0)
    i0 = jnp.floor(t_v).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, N - 1)
    frac = t_v - i0
    gs = [pl.load(g_ref, (offs * C + c,), mask=mask, other=0.0) for c in range(C)]
    for k in range(K):
        s = jnp.zeros((), dtype=gs[0].dtype)
        for c in range(C):
            base = (c * K + k) * N
            u0 = pl.load(U_ref, (base + i0,))
            u1 = pl.load(U_ref, (base + i1,))
            s += jnp.sum(gs[c] * (u0 + frac * (u1 - u0)))
        pl.store(o_ref, (pid, k), s)


def _pallas_theta_bar_2d(tm, tz, g, U):
    """theta_bar for a 2-D lookup: U (k, nz, nm), point arrays any shape."""
    from jax.experimental import pallas as pl
    k, nz, nm = U.shape
    tm_f, tz_f, g_f = tm.reshape(-1), tz.reshape(-1), g.reshape(-1)
    npts = tz_f.shape[0]
    nprog = (npts + _SF_BLOCK - 1) // _SF_BLOCK
    out = pl.pallas_call(
        partial(_sf_theta_kernel_2d, npts=npts, K=k, NZ=nz, NM=nm),
        grid=(nprog,),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 4,
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((nprog, k), g.dtype),
    )(tm_f, tz_f, g_f, U.reshape(-1))
    return jnp.sum(out, axis=0)


def _pallas_theta_bar_1d(t, g, U):
    """theta_bar for the multi-channel 1-D lookup: U (k, n, C), g (..., C)."""
    from jax.experimental import pallas as pl
    k, n, C = U.shape
    Uf = jnp.transpose(U, (2, 0, 1)).reshape(-1)        # (C, k, n) flat; tiny
    t_f = t.reshape(-1)
    npts = t_f.shape[0]
    g_f = g.reshape(-1)                                 # (npts, C) row-major flat
    nprog = (npts + _SF_BLOCK - 1) // _SF_BLOCK
    out = pl.pallas_call(
        partial(_sf_theta_kernel_1d, npts=npts, K=k, C=C, N=n),
        grid=(nprog,),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 3,
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((nprog, k), g.dtype),
    )(t_f, g_f, Uf)
    return jnp.sum(out, axis=0)


def _chunked_sum(arrays, k, partial_fn):
    """CPU fallback: probably not necessary here
    The arrays are zero-padded; ``partial_fn`` must map zero inputs to zero contributions"""
    flat = [a.reshape((-1,) + a.shape[a.ndim - extra:]) if extra else a.reshape(-1)
            for a, extra in arrays]
    npts = flat[0].shape[0]
    if npts <= _SF_CHUNK:
        return partial_fn(*flat)
    npad = (-npts) % _SF_CHUNK
    nchunk = (npts + npad) // _SF_CHUNK
    xs = []
    for a in flat:
        pad = [(0, npad)] + [(0, 0)] * (a.ndim - 1)
        xs.append(jnp.pad(a, pad).reshape((nchunk, _SF_CHUNK) + a.shape[1:]))

    def body(carry, chunk):
        return carry + partial_fn(*chunk), None

    total, _ = lax.scan(body, jnp.zeros((k,), flat[0].dtype), tuple(xs))
    return total

def _sf_lookup1d_bwd(res, g):
    theta, T, U, t = res
    n = T.shape[0]
    i0 = jnp.floor(t).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, n - 1)
    t_bar = jnp.sum(g * (T[i1] - T[i0]), axis=-1)
    k = U.shape[0]
    if _use_pallas():
        theta_bar = _pallas_theta_bar_1d(t, g, U)
    else:
        def partial_fn(g_c, t_c):
            j0 = jnp.floor(t_c).astype(jnp.int32)
            j1 = jnp.minimum(j0 + 1, n - 1)
            v = U[:, j0] + (t_c - j0)[None, :, None] * (U[:, j1] - U[:, j0])
            return jnp.einsum('pc,kpc->k', g_c, v)

        theta_bar = _chunked_sum([(g, 1), (t, 0)], k, partial_fn)
    return (theta_bar.astype(theta.dtype), jnp.zeros_like(T), jnp.zeros_like(U), t_bar)


_sf_lookup1d.defvjp(_sf_lookup1d_fwd, _sf_lookup1d_bwd)

def _bilinear2d(T, tm, tz):
    nz, nm = T.shape
    im0 = jnp.floor(tm).astype(jnp.int32)
    im1 = jnp.minimum(im0 + 1, nm - 1)
    iz0 = jnp.floor(tz).astype(jnp.int32)
    iz1 = jnp.minimum(iz0 + 1, nz - 1)
    fm = tm - im0
    fz = tz - iz0
    flat = T.reshape(-1)
    b0 = iz0 * nm
    b1 = iz1 * nm
    g00 = flat[b0 + im0]
    g10 = flat[b0 + im1]
    g01 = flat[b1 + im0]
    g11 = flat[b1 + im1]
    lo = g00 + fm * (g10 - g00)
    hi = g01 + fm * (g11 - g01)
    val = lo + fz * (hi - lo)
    dm = (1.0 - fz) * (g10 - g00) + fz * (g11 - g01)
    dz = hi - lo
    return val, dm, dz

@jax.custom_vjp
def _sf_lookup2d(theta, T, U, tm, tz):
    """Bilinear interpolation ``T[tz, tm]`` (single channel) with the table-value gradient routed through ``theta``.
    T: (nz, nm); U: (k, nz, nm); tm, tz: fractional indices, clipped. """
    del theta, U
    val, _, _ = _bilinear2d(T, tm, tz)
    return val

def _sf_lookup2d_fwd(theta, T, U, tm, tz):
    return _sf_lookup2d(theta, T, U, tm, tz), (theta, T, U, tm, tz)

def _sf_lookup2d_bwd(res, g):
    theta, T, U, tm, tz = res
    nz, nm = T.shape
    _, dm, dz = _bilinear2d(T, tm, tz)
    tm_bar = g * dm
    tz_bar = g * dz

    k = U.shape[0]

    if _use_pallas():
        theta_bar = _pallas_theta_bar_2d(tm, tz, g, U)
    else:
        Uf = U.reshape(k, -1)

        def partial_fn(g_c, tm_c, tz_c):
            jz0 = jnp.floor(tz_c).astype(jnp.int32)
            jz1 = jnp.minimum(jz0 + 1, nz - 1)
            fz = (tz_c - jz0)[None, :]
            c0 = jz0 * nm
            c1 = jz1 * nm
            jm0 = jnp.floor(tm_c).astype(jnp.int32)
            jm1 = jnp.minimum(jm0 + 1, nm - 1)
            fm = (tm_c - jm0)[None, :]
            vlo = Uf[:, c0 + jm0] + fm * (Uf[:, c0 + jm1] - Uf[:, c0 + jm0])
            vhi = Uf[:, c1 + jm0] + fm * (Uf[:, c1 + jm1] - Uf[:, c1 + jm0])
            return (vlo + fz * (vhi - vlo)) @ g_c            # (k,)

        theta_bar = _chunked_sum([(g, 0), (tm, 0), (tz, 0)], k, partial_fn)
    return (theta_bar.astype(theta.dtype), jnp.zeros_like(T), jnp.zeros_like(U), tm_bar, tz_bar)

_sf_lookup2d.defvjp(_sf_lookup2d_fwd, _sf_lookup2d_bwd)

# ---------------------------------------------------------------------------
# Shape functions. NOT @jax.jit-decorated: those decorated become pjit call boundries, can block fusion with
# surrounding elementwise ops.
# ---------------------------------------------------------------------------
def mean_mbh_from_mco(mco, mpisn, mbhmax):
    a = 1 / (4 * (mpisn - mbhmax))
    mcomax = 2 * mbhmax - mpisn
    return jnp.where(mco < mpisn, mco, mbhmax + a * jnp.square(mco - mcomax))


def largest_mco(mpisn, mbhmax):
    mcomax = 2 * mbhmax - mpisn
    return mcomax + jnp.sqrt(4 * mbhmax * (mbhmax - mpisn))


def log_dNdmCO(mco, a, b, mco_floor=6.0):
    """Log of the CO-core IMF: a broken power law with indices -a (below the
    fixed break at 20 Msun) and -b (above), cf. Eq. 2 of Golomb, Isi & Farr (2024), arXiv:2312.03973 -- 
    except that the power law is flattened below mco_floor (the density is constant on [mco_min, mco_floor]).

    The floor is deliberate, the remnant map is the identity below mpisn, CO cores of 4-6 Msun feed BH masses of ~4-6 Msun
    directly, and an un-floored power law would diverge towards arb integration cutoff at mco_min,

    Flattening below mco_floor keeps support down to mco_min while leaving the power law trusted only where the data
    constrain it (above ~6 Msun).
    """
    mtr = 20.0
    mco_eff = jnp.maximum(mco, mco_floor)
    x = mco_eff / mtr
    return jnp.where(mco_eff < mtr, -a * jnp.log(x), -b * jnp.log(x))


def log_dNdmCO_from_log(log_mco, a, b, mco_floor=6.0):
    """log_dNdmCO with log(mco) supplied, avoiding a redundant log.
    See log_dNdmCO for the role of mco_floor."""
    log_mtr = float(np.log(20.0))
    # np.log for a plain number; jnp.log otherwise,
    log_floor = (float(np.log(mco_floor)) if isinstance(mco_floor, (int, float, np.floating))
                 else jnp.log(mco_floor))
    lx = jnp.maximum(log_mco, log_floor) - log_mtr
    return jnp.where(lx < 0.0, -a * lx, -b * lx)


def smooth_log_dNdmCO(xx, a, b):
    xtr = 20
    delta = 0.05
    return -a * jnp.log(xx / xtr) + delta * (a - b) * jnp.log(0.5 * (1 + (xx / xtr) ** (1 / delta)))


def log_smooth_turnon(m, mmin, width=0.05):
    # -log1p(exp(-u)) == -softplus(-u), but softplus never forms exp(large),
    # so it cannot produce an inf that turns into a NaN gradient.
    dm = mmin * width
    return -jax.nn.softplus(-(m - mmin) / dm)


def mmin_log_smooth_turnon(m, delta_m, mmin):
    shifted_mass = jnp.nan_to_num((m - mmin) / delta_m, nan=0)
    shifted_mass = jnp.clip(shifted_mass, 1e-6, 1 - 1e-6)
    exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
    exponent = jnp.where(exponent > 87.0, 87.0, exponent)
    # log(logistic(-e)) == -softplus(e): one transcendental instead of two, exact (rather than log(exp()))
    return jnp.where(m < mmin, -jnp.inf, -jax.nn.softplus(exponent))


def log_gaussian_bump(m, mu, sigma):
    return -0.5 * jnp.square((m - mu) / sigma)

def log_trapz_grid(log_f, x):
    log_dx = jnp.log(jnp.diff(x))
    return jss.logsumexp(jnp.log(0.5) + jnp.logaddexp(log_f[..., :-1], log_f[..., 1:]) + log_dx, axis=-1,)

def log_normalized_gaussian(m, mu, sigma):
    return log_gaussian_bump(m, mu, sigma) - 0.5 * _LOG_2PI - jnp.log(sigma)

def log_normalized_power_law_tail(m, mbhmax, c):
    return jnp.log(c - 1) - jnp.log(mbhmax) - c * jnp.log(m / mbhmax)

def log_normalized_power_law_tail_from_log(log_m, log_mbhmax, c):
    return jnp.log(c - 1) - log_mbhmax - c * (log_m - log_mbhmax)

def safe_log(x, eps=None):
    # Clamp at the smallest normal value of the actual dtype instead.
    if eps is None:
        eps = float(np.finfo(np.float32).tiny)
    return jnp.log(jnp.clip(x, eps, None))


# ---------------------------------------------------------------------------
# PISN mass function grid
# ---------------------------------------------------------------------------
@dataclass
class LogDNDMPISN(object):
    a: object
    b: object
    mpisn: object
    mbhmax: object
    sigma: object
    mco_min: object = 4.0
    mco_floor: object = 6.0
    n_m: object = 512
    mbh_axis: object = dataclasses.field(init=False)
    mbh_grid: object = dataclasses.field(init=False)
    log_dN_grid: object = dataclasses.field(init=False)
    log_Z_grid: object = dataclasses.field(init=False)

    def __post_init__(self):
        min_bh_mass = 1.5
        min_co_mass = 1.0
        max_bh_mass = 100.0
        max_co_mass = 100.0

        n_m = int(self.n_m)
        self.mbh_axis = _LogAxis(min_bh_mass, max_bh_mass, n_m + 2)
        mco_axis = _LogAxis(min_co_mass, max_co_mass, n_m)

        log_mbh = jnp.asarray(self.mbh_axis.log_grid)          # (n_mbh,)
        log_mco = jnp.asarray(mco_axis.log_grid)               # (n_mco,)
        mco = mco_axis.grid

        sigma = self.sigma
        mpisn = jnp.atleast_1d(self.mpisn)[:, None]            # (nz, 1)
        mbhmax = jnp.atleast_1d(self.mbhmax)[:, None]

        mu = mean_mbh_from_mco(mco[None, :], mpisn, mbhmax)    # (nz, n_mco)
        mu_min = 0.1
        mu = jnp.where(mu > 0, mu, mu_min)
        log_mu = jnp.log(mu)

        # Terms that do not depend on mco are pulled out of the integral.
        log_wco = log_dNdmCO_from_log(log_mco, self.a, self.b, mco_floor=self.mco_floor
        ) + log_smooth_turnon(mco, self.mco_min, width=0.05)    # (n_mco,)

        # Integrand in (z, mbh, mco) layout so the mco reduction is over thecontiguous trailing axis.
        lw = log_wco[None, None, :] - 0.5 * jnp.square((
            log_mbh[None, :, None] - log_mu[:, None, :]) / sigma) # (nz, n_mbh, n_mco)

        # Max-subtracted trapezoid in linear space: one exp over the big grid,
        # versus logaddexp+logsumexp which needs three transcendental passes.
        M = jnp.max(lw, axis=-1, keepdims=True)
        M = jnp.where(jnp.isfinite(M), M, 0.0)
        p = jnp.exp(lw - M)
        dmco = jnp.diff(mco)                                    # (n_mco-1,)
        integral = 0.5 * jnp.sum((p[..., :-1] + p[..., 1:]) * dmco, axis=-1)
        log_int = jnp.log(integral) + M[..., 0]                 # (nz, n_mbh)

        # Constants pulled out above, restored here.
        self.log_dN_grid = log_int - 0.5 * _LOG_2PI - jnp.log(sigma) - log_mbh[None, :]
        self.mbh_grid = self.mbh_axis.grid
        self.log_Z_grid = log_trapz_grid(self.log_dN_grid, self.mbh_grid)

@dataclass
class LogDNDM(object):
    a: object
    b: object
    c: object
    mpisn: object
    mpisndot: object
    mbhmax: object
    sigma: object
    fpl: object
    mp_low: object
    msigma_low: object
    flow: object
    mco_min: object = 4.0
    mco_floor: object = 6.0
    mbh_min: object = 3.0
    delta_m: object = 2.5
    zmax: object = 20
    mref: object = 30.0
    zref: object = 0.001
    n_z: object = 30
    use_low_bump: bool = True
    smooth_tail_edge: bool = False
    log_dndm_pisn: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.dmbhmax = self.mbhmax - self.mpisn
        self.setup_interp()

    def setup_interp(self):
        self._z_dependent = not _is_static_zero(self.mpisndot)
        n_z = int(self.n_z) if self._z_dependent else 1

        self.z_axis = _Log1pAxis(self.zmax, n_z)
        self.z_array = self.z_axis.grid
        if self._z_dependent:
            mpisns = self.mpisn + self.mpisndot * (1 - 1 / (1 + self.z_array))
        else:
            mpisns = jnp.atleast_1d(self.mpisn)
        mbhmaxs = mpisns + self.dmbhmax

        self.log_dndm_pisn = LogDNDMPISN(
            self.a, self.b, mpisns, mbhmaxs, self.sigma, 
            mco_min=self.mco_min, mco_floor=self.mco_floor
        )
        self.mbh_axis = self.log_dndm_pisn.mbh_axis
        self.mbh_grid = self.log_dndm_pisn.mbh_grid
        self.log_dndm_pisn_grid = self.log_dndm_pisn.log_dN_grid
        self.log_Z_pisn_grid = self.log_dndm_pisn.log_Z_grid
        self.mbhmaxs = jnp.asarray(mbhmaxs)
        self._n_mbh = self.mbh_axis.n
        self._n_z = n_z

    def __call__(self, m, z):
        m, z = jnp.broadcast_arrays(jnp.asarray(m), jnp.asarray(z))
        log_m = jnp.log(m)
        log1p_z = jnp.log1p(z)
        mbhmax_at_samples = self.mbhmax_at_z(z)
        return self.call_from_logs(m, log_m, z, log1p_z, mbhmax_at_samples)

    def interp_2d_dndmpisn(self, m, z):
        m, z = jnp.broadcast_arrays(jnp.asarray(m), jnp.asarray(z))
        return self._interp_from_log(jnp.log(m), z, jnp.log1p(z))

    def _z_cell(self, z, log1p_z):
        return self.z_axis.cell_and_frac(z, log1p_z)

    def _interp_from_log(self, log_m, z, log1p_z):
        im0, fm = self.mbh_axis.cell_and_frac(log_m)
        if not self._z_dependent:
            return _gather_lerp1d(self.log_dndm_pisn_grid[0], im0, fm, self._n_mbh)
        iz0, fz = self._z_cell(z, log1p_z)
        return _gather_lerp2d(self.log_dndm_pisn_grid, im0, fm, iz0, fz, self._n_mbh, self._n_z)

    def log_Z_pisn_at_z(self, z):
        z = jnp.asarray(z)
        return self._log_Z_from_z(z, jnp.log1p(z))

    def _log_Z_from_z(self, z, log1p_z):
        if not self._z_dependent:
            return self.log_Z_pisn_grid[0]
        iz0, fz = self._z_cell(z, log1p_z)
        return _gather_lerp1d(self.log_Z_pisn_grid, iz0, fz, self._n_z)

    def mbhmax_at_z(self, z):
        if not self._z_dependent:
            return self.mpisn + self.dmbhmax
        return self.mpisn + self.mpisndot * (1 - 1 / (1 + z)) + self.dmbhmax

    def call_from_logs(self, m, log_m, z, log1p_z, mbhmax_at_samples):
        # 1. Base PISN log-density
        log_p_pisn_raw = self._interp_from_log(log_m, z, log1p_z)
        log_p_pisn_raw = jnp.where(m >= self.mbh_grid[-1], -jnp.inf, log_p_pisn_raw)
        log_p_pisn = log_p_pisn_raw - self._log_Z_from_z(z, log1p_z)

        log_mbhmax = jnp.log(mbhmax_at_samples)

        # 2. Boundary density evaluated at m = mbhmax(z)
        log_p_pisn_at_mbhmax = (
            self._interp_from_log(log_mbhmax, z, log1p_z) 
            - self._log_Z_from_z(z, log1p_z)
        )

        # 3. Anchored tail: log p_PL(m) = log p_PISN(mbhmax) - c * (log m - log mbhmax)
        log_p_pl = log_p_pisn_at_mbhmax - self.c * (log_m - log_mbhmax)

        # 4. Piecewise exact switch (C0 continuous)
        log_p_main_unnorm = jnp.where(m <= mbhmax_at_samples, log_p_pisn, log_p_pl)

        # 5. Normalization adjustment for the added tail area above mbhmax
        log_A_tail = log_p_pisn_at_mbhmax + log_mbhmax - jnp.log(jnp.maximum(self.c - 1.0, 1e-4))
        log_Z_total = jnp.logaddexp(0.0, log_A_tail)
        
        log_p_main = log_p_main_unnorm - log_Z_total

        # 6. Mixture combination
        if self.use_low_bump:
            log_p_low = log_normalized_gaussian(m, self.mp_low, self.msigma_low)
            log_denom = jnp.log1p(self.flow)
            log_w_main = -log_denom
            log_w_low = safe_log(self.flow) - log_denom

            log_dNdm = jnp.logaddexp(log_w_main + log_p_main, log_w_low + log_p_low)
        else:
            log_dNdm = log_p_main

        logwindow = mmin_log_smooth_turnon(m, delta_m=self.delta_m, mmin=self.mbh_min)
        return log_dNdm + logwindow

@dataclass
class LogDNDV(object):
    r"""
    Madau-Dickinson-like merger rate density over cosmic time:
     \frac{\mathrm{d} N}{\mathrm{d} V \mathrm{d} t} \propto \frac{\left( 1 + z \right)^\lambda}{1 + \left( \frac{1 + z}{1 + z_p}
     \right)^\kappa} """
    lam: object
    kappa: object
    zp: object
    zref: object = 0.001
    zmax: object = 20
    log_norm: object = 0.0

    def __post_init__(self):
        self.log_norm = -self(self.zref)

    def __call__(self, z):
        z = jnp.asarray(z)
        return self.from_log1p(jnp.log1p(z))

    def from_log1p(self, log1p_z):
        log1p_zmax = jnp.log1p(self.zmax)
        return jnp.where(log1p_z < log1p_zmax,
            self.lam * log1p_z - jnp.log1p(jnp.exp(self.kappa * (log1p_z - jnp.log1p(self.zp))))
            + self.log_norm, -jnp.inf,)

@dataclass
class LogDNDMDQDV(object):
    a: object
    b: object
    c: object
    mpisn: object
    mpisndot: object
    mbhmax: object
    sigma: object
    fpl: object
    beta: object
    lam: object
    kappa: object
    zp: object
    mp_low: object
    msigma_low: object
    flow: object
    mref: object = 30.0
    qref: object = 1.0
    zref: object = 0.001
    zmax: object = 20
    mbh_min: object = 3.0
    delta_m: object = 2.5
    mco_min: object = 4.0
    mco_floor: object = 6.0
    n_m: object = 512
    n_z: object = 30
    log_dndm: object = dataclasses.field(init=False)
    log_dndv: object = dataclasses.field(init=False)
    use_low_bump: object = True
    smooth_tail_edge: bool = False

    def __post_init__(self):
        self.log_dndm = LogDNDM(
            self.a, self.b, self.c, self.mpisn, self.mpisndot, self.mbhmax, self.sigma,
            self.fpl, mp_low=self.mp_low, msigma_low=self.msigma_low, flow=self.flow,
            mref=self.mref, zmax=self.zmax, zref=self.zref, mbh_min=self.mbh_min,
            delta_m=self.delta_m, mco_min=self.mco_min, mco_floor=self.mco_floor,
            n_z=self.n_z, use_low_bump=self.use_low_bump, smooth_tail_edge=self.smooth_tail_edge,)
        self.log_dndv = LogDNDV(self.lam, self.kappa, self.zp, self.zref, zmax=self.zmax)
        self._normalize()

    def _normalize(self):
        self.log_norm = 0
        log_dN_ref = self(self.mref, self.qref, self.zref)
        self.log_norm = jnp.log(self.mref) + log_dN_ref

    def __call__(self, m1, q, z):
        # atleast_1d matches the original, (scalars were given shape (1,), log_norm needs that shape
        m1, q, z = jnp.broadcast_arrays(jnp.atleast_1d(jnp.asarray(m1)), jnp.atleast_1d(jnp.asarray(q)),
            jnp.atleast_1d(jnp.asarray(z)),)
        return self.call_from_logs(m1, jnp.log(m1), jnp.log(q), z, jnp.log1p(z))

    def call_from_logs(self, m1, log_m1, log_q, z, log1p_z):
        """Same value as ``__call__`` but takes the logs the caller already has, eliminate redundancies."""
        m2 = m1 * jnp.exp(log_q)
        log_m2 = log_m1 + log_q
        mt = m1 + m2

        ld = self.log_dndm
        # Computed once and shared between the m1 and m2 evaluations, depends only on z
        mbhmax_at_samples = jnp.broadcast_to( jnp.asarray(ld.mbhmax_at_z(z)), jnp.shape(m1))

        return ( ld.call_from_logs(m1, log_m1, z, log1p_z, mbhmax_at_samples)
            + ld.call_from_logs(m2, log_m2, z, log1p_z, mbhmax_at_samples)
            + self.beta * jnp.log(mt / (self.mref * (1 + self.qref)))
            + log_m1 + self.log_dndv.from_log1p(log1p_z)- self.log_norm)

@dataclass
class FlatwCDMCosmology(object):
    """
    Function-like object representing a flat w-CDM cosmology.
    """
    h: object
    Om: object
    w: object
    zmax: object = 20.0
    ninterp: object = 1024
    # Size of auxiliary table used by z_and_log_jacobian, indexed by log(dL/dH) (not z).
    ndl: object = 2048
    zmin_table: object = 1e-5
    zinterp: object = dataclasses.field(init=False)
    dcinterp: object = dataclasses.field(init=False)
    dlinterp: object = dataclasses.field(init=False)
    ddlinterp: object = dataclasses.field(init=False)
    vcinterp: object = dataclasses.field(init=False)
    dvcinterp: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.zinterp = jnp.expm1(jnp.linspace(np.log(1), jnp.log(1 + self.zmax), self.ninterp))
        self.Einterp = self.E(self.zinterp)
        self.dcinterp = self.dH * jnp_cumtrapz(1 / self.Einterp, self.zinterp)
        self.dlinterp = self.dcinterp * (1 + self.zinterp)
        self.ddlinterp = self.dcinterp + self.dH * (1 + self.zinterp) / self.Einterp
        self.vcinterp = 4 / 3 * np.pi * self.dcinterp * self.dcinterp * self.dcinterp
        self.dvcinterp = 4 * np.pi * jnp.square(self.dcinterp) * self.dH / self.Einterp
        self.dlinterp_dimless = self.dlinterp / self.dH
        self.dcinterp_dimless = self.dcinterp / self.dH
        self._setup_dl_table()

    def _setup_dl_table(self):
        r"""Tabulate, against a grid uniform in u = \log(d_L/d_H)

          * ``log1p(z)``  -- every downstream use of z wants log1p(z) or 1+z
          * ``J(u) = log(dVC/dz) - log(ddL/dz) - 2 log1p(z) - 2 log(dH)``

        Dimensionless distances depend only on (Om, w), => when those are fixed, whole table is a compile-time constant.
        """
        n = int(self.ndl)
        x = self.dlinterp_dimless

        # Lower edge: a z far below any real event, so clamping there is inconsequential. Upper edge: top of z table.
        z_lo = float(self.zmin_table)
        x_lo = jnp.interp(z_lo, self.zinterp, x)
        x_hi = x[-1]

        self._u_lo = jnp.log(x_lo)
        u_hi = jnp.log(x_hi)
        self._inv_du = (n - 1) / (u_hi - self._u_lo)
        self._n_dl = n

        u_grid = jnp.linspace(self._u_lo, u_hi, n)
        # One 1024-point searchsorted at setup; negligible next to the (nobs, nsamp) arrays it saves searching.
        z_grid = jnp.interp(jnp.exp(u_grid), x, self.zinterp)
        self._log1p_z_table = jnp.log1p(z_grid)

        E_g = self.E(z_grid)
        dc_dimless = jnp.interp(z_grid, self.zinterp, self.dcinterp_dimless)
        ddl_dimless = dc_dimless + (1 + z_grid) / E_g
        dvc_dimless = 4 * np.pi * jnp.square(dc_dimless) / E_g
        self._J_table = ( jnp.log(dvc_dimless) - jnp.log(ddl_dimless) - 2 * self._log1p_z_table)

    def z_and_log_jacobian(self, log_dl):
        """Given ``log(d_L)``, return ``(log1p(z), J)`` where
            J == log(dVC/dz) - log(ddL/dz) - 2*log1p(z)
        """
        t = jnp.clip((log_dl - jnp.log(self.dH) - self._u_lo) * self._inv_du, 0.0, self._n_dl - 1.0)
        log1p_z = _lerp1d(self._log1p_z_table, t, self._n_dl)
        J = _lerp1d(self._J_table, t, self._n_dl) + 2 * jnp.log(self.dH)
        return log1p_z, J

    @property
    def dH(self):
        return 2.99792 / self.h

    @property
    def Ol(self):
        return 1 - self.Om

    @property
    def om(self):
        return self.Om * jnp.square(self.h)

    @property
    def ol(self):
        return self.Ol * jnp.square(self.h)

    def E(self, z):
        opz = 1 + z
        opz3 = opz * opz * opz
        return jnp.sqrt(self.Om * opz3 + (1 - self.Om) * opz ** (3 * (1 + self.w)))

    def dC(self, z):
        return jnp.interp(z, self.zinterp, self.dcinterp)

    def dL(self, z):
        return jnp.interp(z, self.zinterp, self.dlinterp)

    def VC(self, z):
        return jnp.interp(z, self.zinterp, self.vcinterp)

    def dVCdz(self, z):
        return jnp.interp(z, self.zinterp, self.dvcinterp)

    def ddL_dz(self, z):
        return jnp.interp(z, self.zinterp, self.ddlinterp)

    def z_of_dC(self, dC):
        return jnp.interp(dC / self.dH, self.dcinterp_dimless, self.zinterp)

    def z_of_dL(self, dL):
        return jnp.interp(dL / self.dH, self.dlinterp_dimless, self.zinterp)


coords = {
    'm_grid': np.exp(np.linspace(np.log(1), np.log(450), 128)),
    'q_grid': np.linspace(0, 1, 129)[1:],
    'z_grid': np.expm1(np.linspace(np.log1p(0), np.log1p(20), 128)),
}


def get_deterministic_parameters(sample, use_low_bump=True):
    out = {}

    if 'log_h' in sample and 'h' not in sample:
        out['h'] = numpyro.deterministic('h', jnp.exp(sample['log_h']))
    if 'log_sigma' in sample and 'sigma' not in sample:
        out['sigma'] = numpyro.deterministic('sigma', jnp.exp(sample['log_sigma']))
    if 'log_mp_low' in sample and 'mp_low' not in sample:
        out['mp_low'] = numpyro.deterministic('mp_low', jnp.exp(sample['log_mp_low']))
    h = out.get('h', sample.get('h'))

    # --- mpisn / mbhmax: exactly one of these three parameterizations ------
    #   (1) mbhmax + dmbhmax sampled directly  [NEW -- the well-constrained
    #       edge location is primary, mpisn is derived]
    #   (2) mpisn_ref (+ zpivot, mpisndot) pivoted     [existing]
    #   (3) direct mpisn                                [existing]
    if 'mpisn' not in sample and 'mbhmax' in sample and 'dmbhmax' in sample:
        out['mpisn'] = numpyro.deterministic('mpisn', sample['mbhmax'] - sample['dmbhmax'])
    elif 'mpisn' not in sample and ('mpisn_ref' in sample or 'log_mpisn_ref' in sample):
        if 'zpivot' not in sample:
            raise KeyError("Sampling mpisn_ref/log_mpisn_ref requires a fixed "
                           "zpivot in the prior file")
        if 'log_mpisn_ref' in sample:
            mpisn_ref = numpyro.deterministic(
                'mpisn_ref', jnp.exp(sample['log_mpisn_ref']))
        else:
            mpisn_ref = sample['mpisn_ref']
        xpivot = sample['zpivot'] / (1.0 + sample['zpivot'])
        out['mpisn'] = numpyro.deterministic(
            'mpisn', mpisn_ref - sample['mpisndot'] * xpivot)
    elif 'log_mpisn' in sample and 'mpisn' not in sample:
        out['mpisn'] = numpyro.deterministic('mpisn', jnp.exp(sample['log_mpisn']))
    mpisn = out.get('mpisn', sample.get('mpisn'))
    # -----------------------------------------------------------------------

    out['kappa'] = numpyro.deterministic('kappa', sample['lam'] + sample['dkappa'])

    # mbhmax: only derive it if it wasn't sampled directly (case (1) above).
    if 'mbhmax' not in sample:
        out['mbhmax'] = numpyro.deterministic('mbhmax', mpisn + sample['dmbhmax'])
    else:
        out['mbhmax'] = sample['mbhmax']
    # -----------------------------------------------------------------------

    # Default cosmology parameterization: sample Omh2 = Om*h^2 (less
    # degenerate with h than Om) and derive Om.  A prior that still samples Om directly is unchanged
    if 'Omh2' in sample and 'Om' not in sample:
        out['Om'] = numpyro.deterministic(
            'Om', sample['Omh2'] / jnp.square(h)
        )

    if use_low_bump:
        if 'logit_flow' in sample:
            out['flow'] = numpyro.deterministic('flow', jax.nn.sigmoid(sample['logit_flow']))
        elif 'flow' in sample:
            out['flow'] = sample['flow']
        elif 'log_flow' in sample:
            # Prefer log_fpeak; print once (utils holds the flag).
            from utils import warn_log_flow_deprecated
            warn_log_flow_deprecated()
            out['flow'] = numpyro.deterministic('flow', jnp.exp(sample['log_flow']))
        elif 'log_fpeak' in sample:
            # Peak-height parametrization: log_fpeak = log_flow - log(msigma_low); sampling log_fpeak removes the built-in
            # amplitude-width correlation
            log_flow = numpyro.deterministic('log_flow', sample['log_fpeak'] + jnp.log(sample['msigma_low']))
            out['flow'] = numpyro.deterministic('flow', jnp.exp(log_flow))
        else:
            raise KeyError("Need one of logit_flow, flow, log_flow, or log_fpeak")

    if 'logit_fpl' in sample:
        out['fpl'] = numpyro.deterministic('fpl', jax.nn.sigmoid(sample['logit_fpl']))
    elif 'fpl' in sample:
        out['fpl'] = sample['fpl']
    elif 'log_fpl' in sample:
        out['fpl'] = numpyro.deterministic('fpl', jnp.exp(sample['log_fpl']))
    else:
        raise KeyError("Need one of logit_fpl, fpl, or log_fpl")
    return out


def map_truths_to_prior_coords(truths, prior):
    """Map canonical truth values into whatever coordinates the prior file actually samples, so init_to_value and
    recentering_baselines can start at the truth under a reparameterized  prior."""
    tv = dict(truths)
    if ('mpisn_ref' in prior or 'log_mpisn_ref' in prior) and 'mpisn' in tv:
        zpivot = prior['zpivot']  # fixed float in the prior file
        xpivot = zpivot / (1.0 + zpivot)
        ref = tv['mpisn'] + tv.get('mpisndot', 0.0) * xpivot
        if 'mpisn_ref' in prior:
            tv['mpisn_ref'] = ref
        else:
            tv['log_mpisn_ref'] = jnp.log(ref)
    for lin, log in (('h', 'log_h'), ('mpisn', 'log_mpisn'),
                     ('sigma', 'log_sigma'), ('mp_low', 'log_mp_low')):
        if log in prior and lin in tv:
            tv[log] = jnp.log(tv[lin])
    return tv


def log_smooth_neff_boundary(values, criteria):
    scaled_x = (values - criteria) / (0.05 * criteria)
    return jnp.minimum(0.0, scaled_x)

def build_population_model(sample, use_low_bump=True, n_z=30, smooth_tail_edge=False):
    return LogDNDMDQDV(a=sample['a'], b=sample['b'], c=sample['c'], mpisn=sample['mpisn'],
        mpisndot=sample['mpisndot'], mbhmax=sample['mbhmax'], sigma=sample['sigma'],
        fpl=sample['fpl'], beta=sample['beta'], lam=sample['lam'], kappa=sample['kappa'],
        zp=sample['zp'], zmax=sample['zmax'], mbh_min=sample['mbh_min'],
        delta_m=sample['delta_m'], mp_low=sample.get('mp_low', 1.0), msigma_low=sample.get('msigma_low', 1.0),
        flow=sample.get('flow', 0.0), use_low_bump=use_low_bump, n_z=n_z,
        smooth_tail_edge=smooth_tail_edge, mco_min=sample.get('mco_min', 4.0),
        mco_floor=sample.get('mco_floor', 6.0),)


# Floor used when a whole reduction underflows to zero weight.  
_LOG_ZERO_FLOOR = -1e6


def _logsumexp_and_neff(log_wts, axis):
    """One max-subtracted pass giving logsumexp(w), logsumexp(2w) and the
    importance-sampling n_eff = (sum w)^2 / sum w^2 - saves an exponentation
    Gradient safe - no nan gradients (which NUTS sees as a divergence). Cutoff with a jnp.where whos grad=0"""
    # The max subtraction cancels analytically, so its gradient is spurious.
    M = jnp.max(log_wts, axis=axis, keepdims=True)
    M = lax.stop_gradient(jnp.where(jnp.isfinite(M), M, 0.0))
    e = jnp.exp(log_wts - M)
    s1 = jnp.sum(e, axis=axis)
    s2 = jnp.sum(jnp.square(e), axis=axis)
    Ms = jnp.squeeze(M, axis=axis)

    alive = s1 > 0
    s1_safe = jnp.where(alive, s1, 1.0)
    s2_safe = jnp.where(alive, s2, 1.0)

    lse1 = jnp.where(alive, jnp.log(s1_safe) + Ms, _LOG_ZERO_FLOOR)
    lse2 = jnp.where(alive, jnp.log(s2_safe) + 2 * Ms, 2 * _LOG_ZERO_FLOOR)
    neff = jnp.where(alive, jnp.square(s1_safe) / s2_safe, 0.0)
    return lse1, lse2, neff


def pop_cosmo_model(m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel,
                    Ndraw, priors=None, use_low_bump=True, n_z=30, store_per_event=False, neff_criterion=None,
                    neff_penalty="mc_variance", mc_variance_budget=5.0, tabulate_mass_function=None, n_mass_table=8192,
                    tabulate_selection=None, scatter_free_tables=None, smooth_tail_edge=True,
                    loglike_ref=None, log_mu_sel_ref=None, log_pdraw_sel_scale=0.0):
    """
    - Ndraw is # of events in the injection samples used to estimate the selection function
    - store_per_event: record the per-event log-likelihood and n_eff arrays as deterministi
    - neff_penalty: what gaurd?
        1. "mc_variance" (default): penalize when the total MC variance of the LL sum_i 1/n_eff_i > `mc_variance_budget`
            (Talbot & Golomb 2023, arXiv:2304.06138), cannot be dominated by nsamp 
        2. "min_neff": penalize when the min-over-events n_eff drops below `neff_criterion`. Old version used this
        3. "none": no per-event factor in the potential; still records diagnostics b
        For "mc_variance" and "none," uses a smooth -softplus boundary instead of the original min(0, .) kink
    - neff_criterion: target for min-over-events of the per-event n_eff, when neff_penalty="min_neff".  Defaults to `nobs`
    - mc_variance_budget: threshold for sum_i 1/n_eff_i when neff_penalty="mc_variance"
    - tabulate_mass_function: evaluate the single-mass function log_dndm(m, z) once per call on log-m grid
        when mpisndot==0, just do one slice
    - tabulate_selection: whether the selection samples use the same table as the event samples. 
        Should be True for accuracy, setting to False with tabulation on emits RuntimeWArning
    - scatter_free_tables: route the gradient of the tabulated lookups through per-parameter tangent tables 
    - smooth_tail_edge: drop the hard zero of the power-law tail below m = mbhmax 
        Makes the population density continuous at the edge
        Default True (set False for old behavior)
    - loglike_ref, log_mu_sel_ref: float32 recentering baselines
        Default None: no recentering, bit-identical to the previous behaviour.
    - log_pdraw_sel_scale: constant added to ``log(pdraw_sel)`` before the selection weights.  
        Used to park the float32 ``log_mu_sel`` scalar near zero; Default 0: no scaling.
    """
    # Static bounds for the tabulated mass axis, taken from the data *before* touched by jnp (don't want them as tracers)
    try:
        _m1_np = np.asarray(m1s_det)
        _m1sel_np = np.asarray(m1s_det_sel)
        _mass_table_hi = 1.001 * max(float(_m1_np.max()), float(_m1sel_np.max()))
        _mass_table_lo = min(1.0, float((_m1_np * np.asarray(qs)).min()),
            float((_m1sel_np * np.asarray(qs_sel)).min()),) / 21.0
    except Exception:  # tracer inputs: fall back to a generous fixed range
        _mass_table_lo, _mass_table_hi = 0.05, 1000.0

    (m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel) = map(jnp.asarray,
        (m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel),)

    # Numerical scale only: does not mutate the caller's pdraw_sel array.
    log_pdraw_sel = jnp.log(pdraw_sel) + log_pdraw_sel_scale
    nobs = m1s_det.shape[0]
    nsamp = m1s_det.shape[1]
    nsel = m1s_det_sel.shape[0]
    if neff_criterion is None:
        neff_criterion = nobs

    # Constant-in-the-sampler quantities, depend only on the data, removes three logs per step from the (nobs, nsamp) arrays
    log_m1s_det = jnp.log(m1s_det)
    log_qs = jnp.log(qs)
    log_dls = jnp.log(dls)
    log_m1s_det_sel = jnp.log(m1s_det_sel)
    log_qs_sel = jnp.log(qs_sel)
    log_dls_sel = jnp.log(dls_sel)

    sample = sample_parameters_from_dict(priors)
    deterministic_parameters = get_deterministic_parameters(sample, use_low_bump=use_low_bump)
    sample.update(deterministic_parameters)

    cosmo = FlatwCDMCosmology(sample['h'], sample['Om'], sample['w'], zmax=sample['zmax'])
    log_dN = build_population_model(sample, use_low_bump=use_low_bump, n_z=n_z, smooth_tail_edge=smooth_tail_edge)
    ld = log_dN.log_dndm

    if tabulate_mass_function is None:
        tabulate_mass_function = True
    if tabulate_selection is None:
        tabulate_selection = tabulate_mass_function
    if tabulate_mass_function and not tabulate_selection:
        # Trace-time warning (fires once per compile, not per step).
        import warnings
        warnings.warn(
            "tabulate_selection=False with tabulate_mass_function on evaluates "
            "the selection integral with a DIFFERENT density than the event samples."  
            "This is not a valid hierarchical likelihood and is exploitable by the sampler (see "
            "notes/2026-08-08-tabulated-selection-consistency.md). Diagnostics/benchmarking only -- never use for inference.",
            stacklevel=2,)

    if tabulate_mass_function:
        m_axis = _LogAxis(_mass_table_lo, _mass_table_hi, int(n_mass_table))
        n_tab = m_axis.n
        if scatter_free_tables is None:
            scatter_free_tables = True

        # -inf table entries (below mbh_min, above zmax) are floored
        def _build_mass_table(a_, b_, c_, mpisn_, mpisndot_, mbhmax_, sigma_,
                              fpl_, mp_low_, msigma_low_, flow_, mbh_min_,
                              delta_m_, zmax_, mco_min_, mco_floor_):
            ld_ = LogDNDM(a_, b_, c_, mpisn_, mpisndot_, mbhmax_, sigma_, fpl_,
                          mp_low=mp_low_, msigma_low=msigma_low_, flow=flow_,
                          mco_min=mco_min_, mco_floor=mco_floor_,
                          mbh_min=mbh_min_, delta_m=delta_m_, zmax=zmax_,
                          mref=ld.mref, zref=ld.zref, n_z=ld.n_z, use_low_bump=ld.use_low_bump,
                          smooth_tail_edge=ld.smooth_tail_edge)
            if ld_._z_dependent:
                out = ld_(m_axis.grid[None, :], ld_.z_array[:, None])
            else:
                out = ld_(m_axis.grid, 0.0)
            return jnp.maximum(out, _LOG_ZERO_FLOOR)

        _mass_params = (sample['a'], sample['b'], sample['c'], sample['mpisn'], sample['mpisndot'], 
                        sample['mbhmax'], sample['sigma'], sample['fpl'], sample.get('mp_low', 1.0), 
                        sample.get('msigma_low', 1.0), sample.get('flow', 0.0), sample['mbh_min'], sample['delta_m'], 
                        sample['zmax'], sample.get('mco_min', 4.0), sample.get('mco_floor', 6.0),)

        # Fused dL table
        def _build_dl_table(Om_, w_, lam_, kappa_, zp_, zmax_):
            cos_ = FlatwCDMCosmology(1.0, Om_, w_, zmax=zmax_, ninterp=cosmo.ninterp, ndl=cosmo.ndl,
                                     zmin_table=cosmo.zmin_table)
            dndv_ = LogDNDV(lam_, kappa_, zp_, zref=log_dN.log_dndv.zref,zmax=zmax_)
            log1p_t = cos_._log1p_z_table
            Jg_t = jnp.maximum(cos_._J_table + dndv_.from_log1p(log1p_t), _LOG_ZERO_FLOOR)
            return jnp.stack([log1p_t, Jg_t], axis=-1)

        _dl_params = (sample['Om'], sample['w'], sample['lam'],
                      sample['kappa'], sample['zp'], sample['zmax'])

        if scatter_free_tables:
            f_tab, f_U, f_theta = _linearize_table(_build_mass_table, _mass_params)
            dl_tab, dl_U, dl_theta = _linearize_table(_build_dl_table, _dl_params)
        else:
            f_tab = _build_mass_table(*_mass_params)
            dl_tab = _build_dl_table(*_dl_params)
            f_U = f_theta = dl_U = dl_theta = None

        _two_log_dH = 2 * jnp.log(cosmo.dH)
        log_pair_ref = jnp.log(log_dN.mref * (1 + log_dN.qref))

        def _log_weights(log_m1s_det_, log_qs_, log1p_qs_, log_dls_, log_pdraw_):
            t = jnp.clip((log_dls_ - jnp.log(cosmo.dH) - cosmo._u_lo) * cosmo._inv_du, 0.0, cosmo._n_dl - 1.0,)
            if dl_theta is not None:
                both = _sf_lookup1d(dl_theta, dl_tab, dl_U, t)
                log1p_zs_ = both[..., 0]
                Jg = both[..., 1] + _two_log_dH
            else:
                log1p_zs_ = _lerp1d(dl_tab[:, 0], t, cosmo._n_dl)
                Jg = _lerp1d(dl_tab[:, 1], t, cosmo._n_dl) + _two_log_dH
            log_m1s_ = log_m1s_det_ - log1p_zs_
            if ld._z_dependent:
                # One z cell shared by the m1 and m2 lookups, same linear-in-z weights the direct path's PISN interp
                zs_ = jnp.expm1(log1p_zs_)
                iz0, fz = ld.z_axis.cell_and_frac(zs_, log1p_zs_)
                tm1 = m_axis.frac_index(log_m1s_)
                tm2 = m_axis.frac_index(log_m1s_ + log_qs_)
                if f_theta is not None:
                    tz = iz0.astype(tm1.dtype) + fz
                    fsum = (_sf_lookup2d(f_theta, f_tab, f_U, tm1, tz) + _sf_lookup2d(f_theta, f_tab, f_U, tm2, tz))
                else:
                    im1 = jnp.floor(tm1).astype(jnp.int32)
                    im2 = jnp.floor(tm2).astype(jnp.int32)
                    fsum = (_gather_lerp2d(f_tab, im1, tm1 - im1, iz0, fz, n_tab, ld._n_z)
                            + _gather_lerp2d(f_tab, im2, tm2 - im2, iz0, fz,n_tab, ld._n_z))
            else:
                tm1 = m_axis.frac_index(log_m1s_)
                tm2 = m_axis.frac_index(log_m1s_ + log_qs_)
                if f_theta is not None:
                    fsum = (_sf_lookup1d(f_theta, f_tab[:, None], f_U[:, :, None], tm1)[..., 0]
                            + _sf_lookup1d(f_theta, f_tab[:, None], f_U[:, :, None], tm2)[..., 0])
                else:
                    fsum = (_lerp1d(f_tab, tm1, n_tab) + _lerp1d(f_tab, tm2, n_tab))
            return (fsum + log_dN.beta * (log_m1s_ + log1p_qs_ - log_pair_ref)
                    + log_m1s_ + Jg - log_dN.log_norm - log_pdraw_)

        log_wts = _log_weights(log_m1s_det, log_qs, jnp.log1p(qs), log_dls, log_pdraw)
        if not tabulate_selection:
            # Diagnostic / benchmarking only.
            log1p_zs_sel, J_sel = cosmo.z_and_log_jacobian(log_dls_sel)
            opz_sel = jnp.exp(log1p_zs_sel)
            log_sel_wts = (log_dN.call_from_logs(m1s_det_sel / opz_sel, log_m1s_det_sel - log1p_zs_sel,
                                      log_qs_sel, opz_sel - 1.0, log1p_zs_sel) - log_pdraw_sel + J_sel)
        else:
            log_sel_wts = _log_weights(log_m1s_det_sel, log_qs_sel, jnp.log1p(qs_sel),log_dls_sel, log_pdraw_sel)
    else:
        # detected events 
        log1p_zs, J = cosmo.z_and_log_jacobian(log_dls)
        opz = jnp.exp(log1p_zs)
        zs = opz - 1.0
        m1s = m1s_det / opz
        log_m1s = log_m1s_det - log1p_zs

        log_wts = log_dN.call_from_logs(m1s, log_m1s, log_qs, zs, log1p_zs) - log_pdraw + J

        #  selection samples 
        log1p_zs_sel, J_sel = cosmo.z_and_log_jacobian(log_dls_sel)
        opz_sel = jnp.exp(log1p_zs_sel)
        zs_sel = opz_sel - 1.0
        m1s_sel = m1s_det_sel / opz_sel
        log_m1s_sel = log_m1s_det_sel - log1p_zs_sel

        log_sel_wts = (log_dN.call_from_logs(m1s_sel, log_m1s_sel, log_qs_sel, zs_sel, log1p_zs_sel)- log_pdraw_sel + J_sel)

    lse1, lse2, neff = _logsumexp_and_neff(log_wts, axis=1)
    # lse1 is already floored and gradient-safe, so no nan_to_num, 
    # change: don't map NaN to 0, gives LL of 1
    log_like_per_event = lse1 - jnp.log(nsamp)
    if store_per_event:
        _ = numpyro.deterministic("loglik_array_dim", log_like_per_event)

    # Recentering
    if loglike_ref is not None:
        log_like = jnp.sum(log_like_per_event - jnp.asarray(loglike_ref))
    else:
        log_like = jnp.sum(log_like_per_event)
    _ = numpyro.factor('loglike', log_like)

    # selection function
    lse_sel, lse2_sel, _ = _logsumexp_and_neff(log_sel_wts[None, :], axis=1)
    # Keep scaled value for the float32-sensitive arithmetic below; report the physical value as the deterministic
    log_mu_sel_scaled = jnp.squeeze(lse_sel) - jnp.log(Ndraw)
    log_mu_sel = log_mu_sel_scaled + log_pdraw_sel_scale
    numpyro.deterministic('log_mu_sel', log_mu_sel)
    # Peanlize selection integral underflowing to 0 
    sel_dead = jnp.squeeze(lse_sel) <= _LOG_ZERO_FLOOR
    # Recentering selection factor
    if log_mu_sel_ref is not None:
        sel_log_factor = -nobs * (log_mu_sel_scaled - log_mu_sel_ref)
    else:
        sel_log_factor = -nobs * log_mu_sel_scaled
    _ = numpyro.factor('selfactor', jnp.where(sel_dead, _LOG_ZERO_FLOOR, sel_log_factor))

    log_mu2 = jnp.squeeze(lse2_sel) - 2 * jnp.log(Ndraw)
    x = 2 * log_mu_sel_scaled - jnp.log(Ndraw) - log_mu2
    log_s2 = log_mu2 + jnp.log(-jnp.expm1(jnp.minimum(x, -1e-7)))

    # n_eff guards 
    min_neff = jnp.min(neff)
    mc_var = jnp.sum(1.0 / jnp.clip(neff, 1.0, None))
    if store_per_event:
        numpyro.deterministic("neff", neff)
    numpyro.deterministic("min_neff", min_neff)
    numpyro.deterministic("mc_var_loglike", mc_var)
    if neff_penalty == "min_neff":
        numpyro.factor("neff_criteria", log_smooth_neff_boundary(min_neff, neff_criterion))
    elif neff_penalty == "mc_variance":
        # One-sided penalty, active when mc_var rises *above* the budget.
        # -softplus(x) is the smooth (C-infinity) counterpart of the kinked
        numpyro.factor("neff_criteria",  -jax.nn.softplus((mc_var - mc_variance_budget) / (0.05 * mc_variance_budget)),)
    elif neff_penalty not in (None, "none"):
        raise ValueError(f"unknown neff_penalty: {neff_penalty!r}")

    neff_sel = jnp.exp(2 * log_mu_sel_scaled - log_s2)
    numpyro.deterministic("neff_sel", neff_sel)
    if neff_penalty == "min_neff":
        # legacy-exact mode: keep the original kinked selection guard too
        numpyro.factor("neff_sel_criteria", log_smooth_neff_boundary(neff_sel, 4 * nobs))
    else:
        # smooth counterpart, same rationale as the mc_variance factor above
        # (active when neff_sel drops *below* 4*nobs)
        numpyro.factor(
            "neff_sel_criteria",
            -jax.nn.softplus((4 * nobs - neff_sel) / (0.05 * 4 * nobs)),
        )
    # Physical mu_sel / R: undo log_pdraw_sel_scale so the rate posterior
    # matches the true-drawing-density convention of pdraw_sel.
    mu_sel = jnp.exp(log_mu_sel)

    R_unit = numpyro.sample('R_unit', dist.Normal(0, 1))
    R = numpyro.deterministic('R', nobs / mu_sel + jnp.sqrt(nobs) / mu_sel * R_unit)

    _ = numpyro.deterministic('mdNdmdVdt_fixed_qz',
        coords['m_grid'] * R * jnp.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)),)
    _ = numpyro.deterministic('dNdqdVdt_fixed_mz',
        log_dN.mref * R * jnp.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)),)
    _ = numpyro.deterministic('dNdVdt_fixed_mq',
        log_dN.mref * R * jnp.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])),)
    _ = numpyro.deterministic('hz', cosmo.h * cosmo.E(coords['z_grid']))


def recentering_baselines(model_args, ref_params, rng_seed=0, **model_kwargs):
    """Evaluate per-event log likelihoods and log_mu_sel once at a fixed ref point
    Baselines need to be near typical posterior values - so init point works

    - model_args: the positional arguments of pop_cosmo_model (data + prior).
    - ref_params: dict of parameter values to condition on (e.g. the init/truth point). 
        Parameters not in the dict are drawn from the prior with `rng_seed`
    - model_kwargs: forwarded to pop_cosmo_model, this always evaluates the physical (unscaled) model.

    Returns a dict meant to be splatted into pop_cosmo_model:

      * ``loglike_ref`` (np.float64 (nobs,)): per-event baseline
      * ``log_pdraw_sel_scale`` (float): set to the physical ``log_mu_sel`` at ref
      * ``log_mu_sel_ref`` (float): 0.0 -- scaled selection recentering baseline after applying ``log_pdraw_sel_scale``
      * ``offset`` (float): ``sum(loglike_ref) - nobs*log_mu_sel_phys``; add to centered potential to recover absolute LL
      * ``log_mu_sel_phys_ref`` (float): unscaled ``log_mu_sel`` at ref (same as ``log_pdraw_sel_scale``)
    """
    import numpyro.handlers as handlers

    model_kwargs = dict(model_kwargs)
    model_kwargs["store_per_event"] = True
    model_kwargs.pop("loglike_ref", None)
    model_kwargs.pop("log_mu_sel_ref", None)
    model_kwargs.pop("log_pdraw_sel_scale", None)
    ref_params = {k: jnp.asarray(v) for k, v in ref_params.items()}
    with handlers.seed(rng_seed=rng_seed), handlers.substitute(data=ref_params):
        tr = handlers.trace(pop_cosmo_model).get_trace(*model_args, **model_kwargs)

    loglike_ref = np.asarray(tr["loglik_array_dim"]["value"], dtype=np.float64)
    log_mu_sel_phys = float(np.asarray(tr["log_mu_sel"]["value"]))
    n_dead = int(np.sum(loglike_ref <= 0.5 * _LOG_ZERO_FLOOR))
    if n_dead:
        # A dead ref event carries a baseline of ~_LOG_ZERO_FLOOR, residual is 1e6
        # => huge magnitude right back into the sum and defeats the recentering
        import warnings
        warnings.warn(
            f"recentering_baselines: {n_dead} event(s) have (near-)zero "
            f"likelihood at the reference point; recentering will not help "
            f"until the reference point is moved inside the support.")
    nobs = loglike_ref.shape[0]
    return dict(loglike_ref=loglike_ref, log_pdraw_sel_scale=log_mu_sel_phys, log_mu_sel_ref=0.0,
        log_mu_sel_phys_ref=log_mu_sel_phys, offset=float(loglike_ref.sum() - nobs * log_mu_sel_phys),)
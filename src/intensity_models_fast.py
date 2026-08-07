"""
Drop-in replacement for ``intensity_models`` with the same public API and the
same math, restructured for GPU throughput.

What changed, and why (see bench_model.py for the measurements):

1. No ``jnp.interp`` / ``searchsorted`` in the hot path.  Every interpolation
   table in this model lives on a grid that is uniform in ``log(x)`` or
   ``log1p(z)``, so the fractional grid index is available in closed form.
   ``jnp.interp`` instead runs a ~10-iteration binary search, i.e. ten extra
   gather passes over the (nobs, nsamp) array per lookup.  There are five such
   lookups per likelihood evaluation.

2. One fused cosmology lookup instead of three.  The whole Jacobian block
       -2*log1p(z) - log(ddL/dz) + log(dVC/dz)
   equals ``J(u) + 2*log(dH)`` with ``u = log(dL) - log(dH)``, where J depends
   only on (Om, w).  So we tabulate ``log1p(z)`` and ``J`` against a grid
   uniform in u and get both with one index computation.  When Om and w are
   fixed (the current production setup) those tables are compile-time
   constants and XLA folds them away entirely.

3. The PISN mco integral is done as a max-subtracted trapezoid in linear
   space (one ``exp`` over the big grid) rather than
   ``logsumexp(logaddexp(...))`` (three transcendental passes), with the mco
   axis moved last so the reduction is contiguous.

4. When ``mpisndot`` is a fixed 0 the PISN grid is z-independent, so it is
   built with a single z slice and the 2-D interpolation collapses to 1-D.
   This is detected statically from the prior, so nothing changes for runs
   that do sample mpisndot.

5. The per-event ``logsumexp`` and the ``neff`` diagnostic share one
   max-subtracted pass instead of exponentiating the (nobs, nsamp) array
   twice.

6. ``get_deterministic_parameters`` is no longer wrapped in ``jax.jit``.  That
   wrapper made ``numpyro.deterministic`` sites vanish on a jit cache hit, so
   ``kappa``, ``mbhmax``, ``fpl`` and ``flow`` never reached the output.
   The default cosmology prior samples ``Omh2 = Om*h^2``; this helper then
   records ``Om = Omh2/h^2`` as a deterministic.

Behaviour-preserving throughout except where noted with a "# CHANGED:" comment.
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


# ---------------------------------------------------------------------------
# Static-value helpers
#
# Parameters that a prior file pins to a number arrive here as plain Python
# floats (sample_parameters_from_dict wraps them in numpyro.deterministic,
# which returns the value unchanged), while sampled ones arrive as tracers.
# That lets us specialise the graph on fixed parameters at trace time.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Log-uniform / log1p-uniform axes with closed-form index lookup
# ---------------------------------------------------------------------------
class _LogAxis:
    """A grid uniform in ``log(x)``.  ``frac_index`` maps ``log(x)`` to a
    fractional grid index by arithmetic instead of a binary search."""

    def __init__(self, lo, hi, n):
        self.n = int(n)
        self.log_lo = float(np.log(lo))
        self.log_hi = float(np.log(hi))
        # Build the grid the same way linspace+exp does so that grid[-1] is
        # exactly `hi` (the model compares m against mbh_grid[-1]).
        self.log_grid = np.linspace(self.log_lo, self.log_hi, self.n)
        self.grid = jnp.asarray(np.exp(self.log_grid))
        self.inv_dlog = (self.n - 1) / (self.log_hi - self.log_lo)

    def frac_index(self, log_x):
        return jnp.clip((log_x - self.log_lo) * self.inv_dlog, 0.0, self.n - 1.0)

    def cell_and_frac(self, log_x):
        """(cell index, within-cell weight), interpolating linearly in log(x).

        The original used ``jnp.interp(x, grid, arange(n))``, i.e. linear in x.
        On this grid the cells are 0.8% wide in mass, so the two agree to
        ~5e-5 in log-density -- and linear-in-log-x is the more accurate of the
        two for the power-law-like functions tabulated here.
        """
        t = self.frac_index(log_x)
        i0f = jnp.floor(t)
        return i0f.astype(jnp.int32), t - i0f


class _Log1pAxis:
    """A grid uniform in ``log1p(z)`` from 0 to zmax (same nodes as the
    original's ``expm1(linspace(log 1, log(1+zmax), n))``)."""

    def __init__(self, zmax, n):
        self.n = int(n)
        self.log1p_hi = jnp.log1p(zmax)
        self.log1p_grid = jnp.linspace(0.0, self.log1p_hi, self.n)
        self.grid = jnp.expm1(self.log1p_grid)
        # zmax may be sampled, so keep this as a traced scalar.
        self.inv_dlog = (self.n - 1) / self.log1p_hi

    def cell_and_frac(self, z, log1p_z):
        """(cell index, within-cell weight) reproducing
        ``jnp.interp(z, z_array, arange(n))`` exactly.

        The cell index still comes from log1p(z) in closed form (the nodes are
        log1p-uniform, so ``floor(log1p(z)/dlog1p)`` picks the same cell that a
        binary search would), but the weight is computed linearly in z.  That
        matters here: this axis has only ~30 nodes spanning z in [0, zmax], so
        linear-in-z and linear-in-log1p(z) differ by up to ~1e-2 in
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
# Scatter-contention mitigation.
#
# Reverse-mode AD turns a gather from a parameter-dependent table into a
# scatter-add of one value per data point into the table's slots.  With 36e6
# points and a 514-entry table that is 70000 atomic adds per slot, and it
# dominates the gradient: measured 14.6 ms per gather-VJP, versus 1.0 ms for
# the whole forward pass.  It gets *worse* for smaller tables.
#
# Fix: keep R identical copies of the table and send neighbouring points to
# different copies, so the atomics spread over R*n slots instead of n.  The
# copies are summed by the VJP of the broadcast, which costs nothing.  R=32
# takes that 14.6 ms to 2.9 ms; beyond ~64 the returns flatten.
# ---------------------------------------------------------------------------
SCATTER_REPLICAS = 32

# Below this many points the scatter is not contended enough to be worth the
# extra index arithmetic.
_REPLICATE_MIN_SIZE = 1 << 16


def _replicas_for(shape, R=None):
    R = SCATTER_REPLICAS if R is None else R
    if R <= 1 or len(shape) == 0:
        return 1
    if int(np.prod(shape)) < _REPLICATE_MIN_SIZE:
        return 1
    return int(R)


def _replica_offset(shape, R, n_table):
    """Per-point offset into the replicated table.  Varying along the last axis
    means adjacent lanes in a warp hit different replicas, which is where the
    conflicts would otherwise be."""
    iota = lax.broadcasted_iota(jnp.int32, shape, len(shape) - 1)
    # np.int32 rather than Python int so this also works under jax_enable_x64
    # (a Python int would promote to int64 and lax.rem requires equal dtypes).
    return lax.rem(iota, np.int32(R)) * np.int32(n_table)


def _lerp1d(table, t, n, R=None):
    """Linear interpolation into ``table`` (1-D, length n) at fractional
    indices ``t``.  ``t`` must already be clipped to [0, n-1], which reproduces
    jnp.interp's / map_coordinates(mode='nearest')'s edge clamping."""
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
# Shape functions.  These are deliberately NOT @jax.jit-decorated: under an
# outer jit each decorator becomes a pjit call boundary that XLA has to inline
# and which can block fusion with the surrounding elementwise ops.
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
    fixed break at 20 Msun) and -b (above), cf. Eq. 2 of Golomb, Isi & Farr
    (2024), arXiv:2312.03973 -- except that the power law is flattened below
    mco_floor (the density is constant on [mco_min, mco_floor]).

    The floor is deliberate, not a bug, and matters because this model extends
    the paper's with explicit low-mass features (the Gaussian bump at mp_low
    and the smooth low edge at mbh_min).  Since the remnant map is the
    identity below mpisn, CO cores of 4-6 Msun feed BH masses of ~4-6 Msun
    directly, and an un-floored power law (a can reach ~6 under the prior)
    would diverge toward the arbitrary integration cutoff at mco_min,

      * piling density at the cutoff and making the total rate / selection
        normalization depend on the model's least observable corner (for
        a > 1 the power law is non-integrable toward zero mass),
      * forcing `a` to fit both the 6-20 Msun slope and the 3-6 Msun region,
        degenerate with the bump parameters (flow, mp_low, msigma_low), and
      * concentrating the mco quadrature's integrand in its first few cells.

    Flattening below mco_floor keeps support down to mco_min (so the lower
    edge of the mass function is set by the mbh_min turn-on window, not the
    CO cutoff) while leaving the power law trusted only where the data
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
    # np.log for a plain number (a compile-time constant); jnp.log otherwise,
    # since a prior file's fixed value arrives as a jnp scalar via
    # numpyro.deterministic and float() would raise on a tracer.
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
    # log(logistic(-e)) == -softplus(e): one transcendental instead of two,
    # and exact rather than log(exp(...)).
    return jnp.where(m < mmin, -jnp.inf, -jax.nn.softplus(exponent))


def log_gaussian_bump(m, mu, sigma):
    return -0.5 * jnp.square((m - mu) / sigma)


def log_trapz_grid(log_f, x):
    log_dx = jnp.log(jnp.diff(x))
    return jss.logsumexp(
        jnp.log(0.5) + jnp.logaddexp(log_f[..., :-1], log_f[..., 1:]) + log_dx,
        axis=-1,
    )


def log_normalized_gaussian(m, mu, sigma):
    return log_gaussian_bump(m, mu, sigma) - 0.5 * _LOG_2PI - jnp.log(sigma)


def log_normalized_power_law_tail(m, mbhmax, c):
    return jnp.log(c - 1) - jnp.log(mbhmax) - c * jnp.log(m / mbhmax)


def log_normalized_power_law_tail_from_log(log_m, log_mbhmax, c):
    return jnp.log(c - 1) - log_mbhmax - c * (log_m - log_mbhmax)


def safe_log(x, eps=None):
    # CHANGED: the original default eps=1e-300 underflows to exactly 0 in
    # float32, so safe_log was a no-op there.  Clamp at the smallest normal
    # value of the actual dtype instead.
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
        log_wco = log_dNdmCO_from_log(
            log_mco, self.a, self.b, mco_floor=self.mco_floor
        ) + log_smooth_turnon(mco, self.mco_min, width=0.05)    # (n_mco,)

        # Integrand in (z, mbh, mco) layout so the mco reduction is over the
        # contiguous trailing axis.
        lw = log_wco[None, None, :] - 0.5 * jnp.square(
            (log_mbh[None, :, None] - log_mu[:, None, :]) / sigma
        )                                                       # (nz, n_mbh, n_mco)

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


# ---------------------------------------------------------------------------
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
    # The original model hard-zeroes the power-law tail below m = mbhmax, but
    # the smooth turn-on it multiplies by is only 1/2 there, so the density
    # (and hence the potential) has a step discontinuity at m = mbhmax.  AD
    # cannot see the contribution of samples crossing that edge, so d/dh,
    # d/dmpisn and d/ddmbhmax disagree with finite differences of the potential
    # by 10-30% at typical parameter points.  With smooth_tail_edge=True the
    # hard cut is dropped: the turn-on already suppresses the tail
    # exponentially below the edge (scale 0.05*mbhmax), the density becomes
    # continuous, and AD agrees with finite differences.  The class-level
    # default False keeps the original model exactly; pop_cosmo_model defaults
    # it to True (recommended for sampling -- set it to False there to
    # reproduce the old behaviour exactly).
    smooth_tail_edge: bool = False
    log_dndm_pisn: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.dmbhmax = self.mbhmax - self.mpisn
        self.setup_interp()

    def setup_interp(self):
        # If mpisndot is pinned to zero the PISN grid has no z dependence, so
        # build a single slice and skip the z interpolation entirely.
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
            self.a, self.b, mpisns, mbhmaxs, self.sigma, mco_min=self.mco_min,
            mco_floor=self.mco_floor,
        )
        self.mbh_axis = self.log_dndm_pisn.mbh_axis
        self.mbh_grid = self.log_dndm_pisn.mbh_grid
        # (n_z, n_mbh): mbh is the fast axis, matching _lerp2d's expectation.
        self.log_dndm_pisn_grid = self.log_dndm_pisn.log_dN_grid
        self.log_Z_pisn_grid = self.log_dndm_pisn.log_Z_grid
        self.mbhmaxs = jnp.asarray(mbhmaxs)
        self._n_mbh = self.mbh_axis.n
        self._n_z = n_z

    # -- interpolation -----------------------------------------------------
    def interp_2d_dndmpisn(self, m, z):
        """Public API kept for external callers; prefer the *_from_log form."""
        m, z = jnp.broadcast_arrays(jnp.asarray(m), jnp.asarray(z))
        return self._interp_from_log(jnp.log(m), z, jnp.log1p(z))

    def _z_cell(self, z, log1p_z):
        return self.z_axis.cell_and_frac(z, log1p_z)

    def _interp_from_log(self, log_m, z, log1p_z):
        im0, fm = self.mbh_axis.cell_and_frac(log_m)
        if not self._z_dependent:
            return _gather_lerp1d(self.log_dndm_pisn_grid[0], im0, fm, self._n_mbh)
        iz0, fz = self._z_cell(z, log1p_z)
        return _gather_lerp2d(self.log_dndm_pisn_grid, im0, fm, iz0, fz,
                              self._n_mbh, self._n_z)

    def log_Z_pisn_at_z(self, z):
        z = jnp.asarray(z)
        return self._log_Z_from_z(z, jnp.log1p(z))

    def _log_Z_from_z(self, z, log1p_z):
        if not self._z_dependent:
            return self.log_Z_pisn_grid[0]
        iz0, fz = self._z_cell(z, log1p_z)
        return _gather_lerp1d(self.log_Z_pisn_grid, iz0, fz, self._n_z)

    # -- join point --------------------------------------------------------
    def mbhmax_at_z(self, z):
        if not self._z_dependent:
            return self.mpisn + self.dmbhmax
        return self.mpisn + self.mpisndot * (1 - 1 / (1 + z)) + self.dmbhmax

    # -- evaluation --------------------------------------------------------
    def __call__(self, m, z):
        m = jnp.atleast_1d(jnp.asarray(m))
        z = jnp.atleast_1d(jnp.asarray(z))
        m, z = jnp.broadcast_arrays(m, z)
        mbhmax_at_samples = jnp.asarray(self.mbhmax_at_z(z))
        return self.call_from_logs(
            m, jnp.log(m), z, jnp.log1p(z), jnp.broadcast_to(mbhmax_at_samples, m.shape)
        )

    def call_from_logs(self, m, log_m, z, log1p_z, mbhmax_at_samples):
        log_p_pisn_raw = self._interp_from_log(log_m, z, log1p_z)
        log_p_pisn_raw = jnp.where(m >= self.mbh_grid[-1], -jnp.inf, log_p_pisn_raw)
        log_p_pisn = log_p_pisn_raw - self._log_Z_from_z(z, log1p_z)

        log_mbhmax = jnp.log(mbhmax_at_samples)
        log_p_pl_raw = log_normalized_power_law_tail_from_log(log_m, log_mbhmax, self.c)
        if not self.smooth_tail_edge:
            log_p_pl_raw = jnp.where(log_m < log_mbhmax, -jnp.inf, log_p_pl_raw)
        log_p_pl = log_p_pl_raw + log_smooth_turnon(m, mbhmax_at_samples)

        if self.use_low_bump:
            log_p_low = log_normalized_gaussian(m, self.mp_low, self.msigma_low)
            log_denom = jnp.log1p(self.flow + self.fpl)
            log_w_pisn = -log_denom
            log_w_low = safe_log(self.flow) - log_denom
            log_w_pl = safe_log(self.fpl) - log_denom

            log_dNdm = jnp.logaddexp(log_w_pisn + log_p_pisn, log_w_low + log_p_low)
            log_dNdm = jnp.logaddexp(log_dNdm, log_w_pl + log_p_pl)
        else:
            log_denom = jnp.log1p(self.fpl)
            log_w_pisn = -log_denom
            log_w_pl = safe_log(self.fpl) - log_denom
            log_dNdm = jnp.logaddexp(log_w_pisn + log_p_pisn, log_w_pl + log_p_pl)

        # The original also applied `where(m < mbh_min, -inf, ...)` here; the
        # window below is already -inf for m < mbh_min, so it was redundant.
        logwindow = mmin_log_smooth_turnon(m, delta_m=self.delta_m, mmin=self.mbh_min)
        return log_dNdm + logwindow


# ---------------------------------------------------------------------------
@dataclass
class LogDNDV(object):
    r"""
    Madau-Dickinson-like merger rate density over cosmic time:

    .. math::
        \frac{\mathrm{d} N}{\mathrm{d} V \mathrm{d} t} \propto \frac{\left( 1 + z \right)^\lambda}{1 + \left( \frac{1 + z}{1 + z_p} \right)^\kappa}
    """
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
        return jnp.where(
            log1p_z < log1p_zmax,
            self.lam * log1p_z
            - jnp.log1p(jnp.exp(self.kappa * (log1p_z - jnp.log1p(self.zp))))
            + self.log_norm,
            -jnp.inf,
        )


# ---------------------------------------------------------------------------
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
            n_z=self.n_z,
            use_low_bump=self.use_low_bump, smooth_tail_edge=self.smooth_tail_edge,
        )
        self.log_dndv = LogDNDV(self.lam, self.kappa, self.zp, self.zref, zmax=self.zmax)
        self._normalize()

    def _normalize(self):
        self.log_norm = 0
        log_dN_ref = self(self.mref, self.qref, self.zref)
        self.log_norm = jnp.log(self.mref) + log_dN_ref

    def __call__(self, m1, q, z):
        # atleast_1d matches the original, whose LogDNDM.__call__ promoted
        # scalars to shape (1,) -- log_norm inherits that shape.
        m1, q, z = jnp.broadcast_arrays(
            jnp.atleast_1d(jnp.asarray(m1)), jnp.atleast_1d(jnp.asarray(q)),
            jnp.atleast_1d(jnp.asarray(z)),
        )
        return self.call_from_logs(m1, jnp.log(m1), jnp.log(q), z, jnp.log1p(z))

    def call_from_logs(self, m1, log_m1, log_q, z, log1p_z):
        """Same value as ``__call__`` but takes the logs the caller already has,
        which is where most of the redundant transcendentals were."""
        m2 = m1 * jnp.exp(log_q)
        log_m2 = log_m1 + log_q
        mt = m1 + m2

        ld = self.log_dndm
        # Computed once and shared between the m1 and m2 evaluations, since it
        # depends only on z.
        mbhmax_at_samples = jnp.broadcast_to(
            jnp.asarray(ld.mbhmax_at_z(z)), jnp.shape(m1)
        )

        return (
            ld.call_from_logs(m1, log_m1, z, log1p_z, mbhmax_at_samples)
            + ld.call_from_logs(m2, log_m2, z, log1p_z, mbhmax_at_samples)
            + self.beta * jnp.log(mt / (self.mref * (1 + self.qref)))
            + log_m1
            + self.log_dndv.from_log1p(log1p_z)
            - self.log_norm
        )


# ---------------------------------------------------------------------------
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
    # Size of the auxiliary table used by z_and_log_jacobian, which is indexed
    # by log(dL/dH) rather than by z.
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
        r"""Tabulate, against a grid uniform in :math:`u = \log(d_L/d_H)`:

          * ``log1p(z)``  -- every downstream use of z wants log1p(z) or 1+z
          * ``J(u) = log(dVC/dz) - log(ddL/dz) - 2 log1p(z) - 2 log(dH)``

        so that the entire per-sample cosmology block is two gathers plus a
        scalar ``2 log(dH)``.  The dimensionless distances depend only on
        (Om, w), so when those are fixed this whole table is a compile-time
        constant.
        """
        n = int(self.ndl)
        x = self.dlinterp_dimless

        # Lower edge: a z far below any real event, so clamping there is
        # inconsequential.  Upper edge: the top of the z table.
        z_lo = float(self.zmin_table)
        x_lo = jnp.interp(z_lo, self.zinterp, x)
        x_hi = x[-1]

        self._u_lo = jnp.log(x_lo)
        u_hi = jnp.log(x_hi)
        self._inv_du = (n - 1) / (u_hi - self._u_lo)
        self._n_dl = n

        u_grid = jnp.linspace(self._u_lo, u_hi, n)
        # One 1024-point searchsorted at setup; negligible next to the
        # (nobs, nsamp) arrays it saves searching.
        z_grid = jnp.interp(jnp.exp(u_grid), x, self.zinterp)
        self._log1p_z_table = jnp.log1p(z_grid)

        E_g = self.E(z_grid)
        dc_dimless = jnp.interp(z_grid, self.zinterp, self.dcinterp_dimless)
        ddl_dimless = dc_dimless + (1 + z_grid) / E_g
        dvc_dimless = 4 * np.pi * jnp.square(dc_dimless) / E_g
        self._J_table = (
            jnp.log(dvc_dimless) - jnp.log(ddl_dimless) - 2 * self._log1p_z_table
        )

    def z_and_log_jacobian(self, log_dl):
        """Given ``log(d_L)``, return ``(log1p(z), J)`` where

            J == log(dVC/dz) - log(ddL/dz) - 2*log1p(z)

        i.e. exactly the cosmology block of the population weights.
        """
        t = jnp.clip((log_dl - jnp.log(self.dH) - self._u_lo) * self._inv_du,
                     0.0, self._n_dl - 1.0)
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


# CHANGED: no @jax.jit here.  numpyro.deterministic inside a jit records its
# site during the *inner* trace, so on a jit cache hit the sites disappear and
# kappa / mbhmax / fpl / flow never appear in the MCMC output at all.
def get_deterministic_parameters(sample, use_low_bump=True):
    kappa = numpyro.deterministic('kappa', sample['lam'] + sample['dkappa'])
    mbhmax = numpyro.deterministic('mbhmax', sample['mpisn'] + sample['dmbhmax'])

    out = dict(kappa=kappa, mbhmax=mbhmax)

    # Default cosmology parameterization: sample Omh2 = Om*h^2 (less
    # degenerate with h than Om) and derive Om.  A prior that still samples
    # Om directly is unchanged -- the deterministic is only installed when Om
    # is absent.
    if 'Omh2' in sample and 'Om' not in sample:
        out['Om'] = numpyro.deterministic(
            'Om', sample['Omh2'] / jnp.square(sample['h'])
        )

    if use_low_bump:
        if 'logit_flow' in sample:
            out['flow'] = numpyro.deterministic('flow', jax.nn.sigmoid(sample['logit_flow']))
        elif 'flow' in sample:
            out['flow'] = sample['flow']
        elif 'log_flow' in sample:
            out['flow'] = numpyro.deterministic('flow', jnp.exp(sample['log_flow']))
        else:
            raise KeyError("Need one of logit_flow, flow, or log_flow")

    if 'logit_fpl' in sample:
        out['fpl'] = numpyro.deterministic('fpl', jax.nn.sigmoid(sample['logit_fpl']))
    elif 'fpl' in sample:
        out['fpl'] = sample['fpl']
    elif 'log_fpl' in sample:
        out['fpl'] = numpyro.deterministic('fpl', jnp.exp(sample['log_fpl']))
    else:
        raise KeyError("Need one of logit_fpl, fpl, or log_fpl")

    return out


def log_smooth_neff_boundary(values, criteria):
    scaled_x = (values - criteria) / (0.05 * criteria)
    return jnp.minimum(0.0, scaled_x)


def build_population_model(sample, use_low_bump=True, n_z=30, smooth_tail_edge=False):
    return LogDNDMDQDV(
        a=sample['a'], b=sample['b'], c=sample['c'], mpisn=sample['mpisn'],
        mpisndot=sample['mpisndot'], mbhmax=sample['mbhmax'], sigma=sample['sigma'],
        fpl=sample['fpl'], beta=sample['beta'], lam=sample['lam'], kappa=sample['kappa'],
        zp=sample['zp'], zmax=sample['zmax'], mbh_min=sample['mbh_min'],
        delta_m=sample['delta_m'],
        mp_low=sample.get('mp_low', 1.0), msigma_low=sample.get('msigma_low', 1.0),
        flow=sample.get('flow', 0.0), use_low_bump=use_low_bump, n_z=n_z,
        smooth_tail_edge=smooth_tail_edge,
        # CHANGED: the original silently ignored mco_min from the prior file,
        # and hardcoded the CO-IMF flattening scale (see log_dNdmCO).  Both are
        # now settable from the prior file; the defaults reproduce the original.
        mco_min=sample.get('mco_min', 4.0),
        mco_floor=sample.get('mco_floor', 6.0),
    )


# Floor used when a whole reduction underflows to zero weight.  Large enough to
# be decisively rejected, small enough not to swamp float32 arithmetic the way
# the original 1e30 did.
_LOG_ZERO_FLOOR = -1e6


def _logsumexp_and_neff(log_wts, axis):
    """One max-subtracted pass giving logsumexp(w), logsumexp(2w) and the
    importance-sampling n_eff = (sum w)^2 / sum w^2.

    The original exponentiated the (nobs, nsamp) array twice, once for
    logsumexp(log_wts) and once for logsumexp(2*log_wts).

    It is also gradient-safe.  If every weight in a reduction underflows to
    zero -- which happens whenever a proposed parameter point puts all of an
    event's posterior samples outside the model's support -- the original
    produced a NaN *gradient* (jnp.nan_to_num fixes the value but not the
    derivative), and NUTS reads a NaN gradient as a divergence.  Here the
    dead reductions are cut off by a jnp.where whose gradient is exactly zero.
    """
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
                    Ndraw, priors=None, use_low_bump=True, n_z=30,
                    store_per_event=False, neff_criterion=None,
                    neff_penalty="mc_variance", mc_variance_budget=5.0,
                    tabulate_mass_function=None, n_mass_table=8192,
                    smooth_tail_edge=True,
                    loglike_ref=None, log_mu_sel_ref=None,
                    log_pdraw_sel_scale=0.0):
    """
    Ndraw is # of events in the injection samples used to estimate the selection function

    store_per_event: record the per-event log-likelihood and n_eff arrays as
        deterministic sites.  The original always did; at nobs=9000 that is two
        (9000,) arrays per posterior sample, i.e. hundreds of MB of output for
        a production run.

    neff_penalty: which Monte-Carlo-accuracy guard to include in the potential.
        "mc_variance" (default): penalize when the total MC variance of the
            log likelihood, sum_i 1/n_eff_i, exceeds `mc_variance_budget`.
            This is the quantity that actually controls the MC error of the
            total log-likelihood (Talbot & Golomb 2023, arXiv:2304.06138), so
            it is the more principled guard; it also cannot be dominated by
            nsamp the way a min-n_eff = nobs target can.
        "min_neff": penalize when the min-over-events n_eff drops below
            `neff_criterion`.  This is the legacy-exact mode: it also keeps
            the original kinked (piecewise-linear) form of the selection
            guard `neff_sel_criteria`, so neff_penalty="min_neff" (with the
            default neff_criterion) reproduces the original model's guards
            exactly.
        "none": no per-event factor in the potential; the diagnostics below
            are still recorded so the accuracy can be checked a posteriori.
        In the "mc_variance" and "none" modes the selection guard uses a
        smooth -softplus boundary instead of the original min(0, .) kink:
        same asymptotic slope, ~0 well inside the criterion, -log(2) at it.
        A gradient jump at the boundary causes HMC energy errors
        (divergences) exactly when trajectories probe the guard, so the
        smooth form is preferred unless exact reproduction is needed.
        In every mode `min_neff` and `mc_var_loglike` (= sum_i 1/n_eff_i) are
        recorded as deterministic sites.

    neff_criterion: target for min-over-events of the per-event n_eff, used
        when neff_penalty="min_neff".  Defaults to `nobs`, matching the
        original -- but note that a per-event n_eff can never exceed nsamp, so
        with nsamp < nobs that penalty is unsatisfiable and therefore always
        active with a nonzero gradient.  A per-event target (e.g. 10-100) is
        what this guard is normally meant to enforce.

    mc_variance_budget: threshold for sum_i 1/n_eff_i when
        neff_penalty="mc_variance".  The MC standard deviation of the total
        log likelihood is sqrt(sum 1/n_eff), so a budget of 5 keeps it below
        ~2.2 nats; a budget of 1 keeps it below 1 nat.

    tabulate_mass_function: when mpisndot is pinned to 0 the single-mass
        function log_dndm(m, z) has no z dependence, so it can be evaluated
        once per likelihood call on a fine log-m grid (n_mass_table nodes) and
        every per-sample mass evaluation becomes a single 1-D lerp; the rate
        density log_dndv is likewise folded into the dL lookup table.  This is
        ~2x faster and, as a side effect, smears the model's step discontinuity
        at m = mbhmax (see LogDNDM.call_from_logs) over one table cell, which
        makes the AD gradient of the potential agree with finite differences
        of the potential -- the direct evaluation's d/dh, d/dmpisn and
        d/ddmbhmax miss the contribution of samples crossing that edge and are
        off by 10-20% at typical parameter points.  Default (None): enabled
        exactly when mpisndot is statically 0.  Forced True is ignored when
        mpisndot is sampled (the table would need a z axis).

    smooth_tail_edge: drop the hard zero of the power-law tail below
        m = mbhmax (see LogDNDM.smooth_tail_edge).  This makes the population
        density continuous at the edge, which is what makes d/dmpisn and
        d/ddmbhmax agree with finite differences of the potential; the
        tabulated evaluation alone fixes only d/dh.  Default True (recommended:
        with the hard edge, the NUTS gradients for h, mpisn and dmbhmax are
        wrong by 10-30% at typical parameter points).  NOTE: this is a (small)
        change to the population model itself -- set smooth_tail_edge=False to
        reproduce the original model exactly.

    loglike_ref, log_mu_sel_ref: float32 recentering baselines (see
        notes/2026-08-07-float32-recentering.md and `recentering_baselines`).
        `loglike_ref` is a constant (nobs,) array subtracted from the
        per-event log likelihoods *inside* the sum, and `log_mu_sel_ref` a
        constant scalar subtracted from log_mu_sel inside the selection
        factor.  Both shift the potential by a constant, so the posterior and
        all gradients are unchanged -- but the summed 'loglike' factor then
        scales with the *variation* of the log likelihood over the posterior
        (O(10) nats, independent of nobs) instead of its magnitude
        (O(16*nobs) nats), which removes the dominant float32 roundoff term
        identified in notes/2026-08-07-float32-accuracy-audit.md.  The
        recorded 'loglike' factor and the potential energy ('lp' in arviz)
        are shifted by `-(sum(loglike_ref) - nobs*log_mu_sel_ref)`; use the
        'offset' entry returned by `recentering_baselines` to recover
        absolute values in post-processing.  Default None: no recentering,
        bit-identical to the previous behaviour.

    log_pdraw_sel_scale: constant added to ``log(pdraw_sel)`` before the
        selection weights (equivalent to multiplying every ``pdraw_sel`` by
        ``exp(scale)``).  Used to park the float32 ``log_mu_sel`` scalar near
        0 instead of ~14, shrinking its ulp ~8x so the residual
        ``nobs * ulp(log_mu_sel)`` floor after recentering drops by the same
        factor (see notes/2026-08-07-float32-recentering.md).  The on-disk
        ``pdraw_sel`` is left alone -- this is a numerical knob inside the
        model only.  ``R`` and the recorded ``log_mu_sel`` deterministic are
        corrected back to the physical (unscaled) convention, so rate
        posteriors and diagnostics stay comparable to unscaled runs.
        Default 0: no scaling.
    """
    # Static bounds for the tabulated mass axis, taken from the data *before*
    # it is touched by jnp (inside numpyro's jit the arrays become tracers and
    # can no longer be inspected).  Source-frame masses satisfy
    # m2 <= m1 <= m1_det and m >= m1_det*q/(1+z) with z <= 20, so
    # [min(m1_det*q)/21, max(m1_det)] covers every query.
    try:
        _m1_np = np.asarray(m1s_det)
        _m1sel_np = np.asarray(m1s_det_sel)
        _mass_table_hi = 1.001 * max(float(_m1_np.max()), float(_m1sel_np.max()))
        _mass_table_lo = min(
            1.0,
            float((_m1_np * np.asarray(qs)).min()),
            float((_m1sel_np * np.asarray(qs_sel)).min()),
        ) / 21.0
    except Exception:  # tracer inputs: fall back to a generous fixed range
        _mass_table_lo, _mass_table_hi = 0.05, 1000.0

    (m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel) = map(
        jnp.asarray,
        (m1s_det, qs, dls, log_pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel),
    )

    # Numerical scale only: does not mutate the caller's pdraw_sel array.
    log_pdraw_sel = jnp.log(pdraw_sel) + log_pdraw_sel_scale
    nobs = m1s_det.shape[0]
    nsamp = m1s_det.shape[1]
    nsel = m1s_det_sel.shape[0]
    if neff_criterion is None:
        neff_criterion = nobs

    # Constant-in-the-sampler quantities.  These depend only on the data, so
    # they are the same at every leapfrog step; computing them in log space
    # here removes three logs per step from the (nobs, nsamp) arrays.
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
    log_dN = build_population_model(sample, use_low_bump=use_low_bump, n_z=n_z,
                                    smooth_tail_edge=smooth_tail_edge)
    ld = log_dN.log_dndm

    if tabulate_mass_function is None:
        tabulate_mass_function = not ld._z_dependent
    tabulate_mass_function = tabulate_mass_function and not ld._z_dependent

    if tabulate_mass_function:
        m_axis = _LogAxis(_mass_table_lo, _mass_table_hi, int(n_mass_table))
        n_tab = m_axis.n

        # -inf table entries (below mbh_min, above zmax) are floored so a lerp
        # between two dead nodes cannot form inf - inf = NaN.
        f_tab = jnp.maximum(ld(m_axis.grid, 0.0), _LOG_ZERO_FLOOR)
        log1p_tab = cosmo._log1p_z_table
        Jg_tab = jnp.maximum(
            cosmo._J_table + 2 * jnp.log(cosmo.dH)
            + log_dN.log_dndv.from_log1p(log1p_tab),
            _LOG_ZERO_FLOOR,
        )
        log_pair_ref = jnp.log(log_dN.mref * (1 + log_dN.qref))

        def _log_weights(log_m1s_det_, log_qs_, log1p_qs_, log_dls_, log_pdraw_):
            t = jnp.clip(
                (log_dls_ - jnp.log(cosmo.dH) - cosmo._u_lo) * cosmo._inv_du,
                0.0, cosmo._n_dl - 1.0,
            )
            log1p_zs_ = _lerp1d(log1p_tab, t, cosmo._n_dl)
            Jg = _lerp1d(Jg_tab, t, cosmo._n_dl)
            log_m1s_ = log_m1s_det_ - log1p_zs_
            f1 = _lerp1d(f_tab, m_axis.frac_index(log_m1s_), n_tab)
            f2 = _lerp1d(f_tab, m_axis.frac_index(log_m1s_ + log_qs_), n_tab)
            return (f1 + f2
                    + log_dN.beta * (log_m1s_ + log1p_qs_ - log_pair_ref)
                    + log_m1s_ + Jg - log_dN.log_norm - log_pdraw_)

        log_wts = _log_weights(log_m1s_det, log_qs, jnp.log1p(qs), log_dls, log_pdraw)
        log_sel_wts = _log_weights(log_m1s_det_sel, log_qs_sel, jnp.log1p(qs_sel),
                                   log_dls_sel, log_pdraw_sel)
    else:
        # --- detected events ---------------------------------------------
        log1p_zs, J = cosmo.z_and_log_jacobian(log_dls)
        opz = jnp.exp(log1p_zs)
        zs = opz - 1.0
        m1s = m1s_det / opz
        log_m1s = log_m1s_det - log1p_zs

        log_wts = log_dN.call_from_logs(m1s, log_m1s, log_qs, zs, log1p_zs) - log_pdraw + J

        # --- selection samples ---------------------------------------------
        log1p_zs_sel, J_sel = cosmo.z_and_log_jacobian(log_dls_sel)
        opz_sel = jnp.exp(log1p_zs_sel)
        zs_sel = opz_sel - 1.0
        m1s_sel = m1s_det_sel / opz_sel
        log_m1s_sel = log_m1s_det_sel - log1p_zs_sel

        log_sel_wts = (
            log_dN.call_from_logs(m1s_sel, log_m1s_sel, log_qs_sel, zs_sel, log1p_zs_sel)
            - log_pdraw_sel + J_sel
        )

    lse1, lse2, neff = _logsumexp_and_neff(log_wts, axis=1)
    # lse1 is already floored and gradient-safe, so no nan_to_num is needed.
    # CHANGED: the original mapped NaN to 0 here, i.e. silently treated an event
    # whose likelihood was NaN as having likelihood 1.
    log_like_per_event = lse1 - jnp.log(nsamp)
    if store_per_event:
        _ = numpyro.deterministic("loglik_array_dim", log_like_per_event)

    # Recentering: subtracting a constant per-event baseline *before* the sum
    # keeps the summed factor at O(variation over the posterior) ~ 10 nats
    # instead of O(16*nobs) ~ 1e5 nats, whose float32 ulp (1.6e-2 nats at
    # nobs=9000) was the dominant precision error at production scale.  A
    # constant shift of the potential is invisible to MCMC and to gradients.
    if loglike_ref is not None:
        log_like = jnp.sum(log_like_per_event - jnp.asarray(loglike_ref))
    else:
        log_like = jnp.sum(log_like_per_event)
    _ = numpyro.factor('loglike', log_like)

    # --- selection function ----------------------------------------------
    lse_sel, lse2_sel, _ = _logsumexp_and_neff(log_sel_wts[None, :], axis=1)
    # With log_pdraw_sel_scale = c, every selection weight (and thus this
    # scalar) is shifted by -c relative to the physical integral.  Keep the
    # scaled value for the float32-sensitive arithmetic below; report the
    # physical value as the deterministic.
    log_mu_sel_scaled = jnp.squeeze(lse_sel) - jnp.log(Ndraw)
    log_mu_sel = log_mu_sel_scaled + log_pdraw_sel_scale
    numpyro.deterministic('log_mu_sel', log_mu_sel)
    # CHANGED: if the selection integral underflows to zero, -nobs*log_mu_sel
    # becomes a huge *positive* log-factor (the original's nan_to_num turned it
    # into +1e38), i.e. a completely dead parameter region would look
    # infinitely attractive.  Penalize it instead.
    sel_dead = jnp.squeeze(lse_sel) <= _LOG_ZERO_FLOOR
    # Recentering of the selection factor uses the *scaled* scalar so that,
    # with log_pdraw_sel_scale chosen as the physical log_mu_sel at the ref
    # point, log_mu_sel_scaled sits near 0 and its float32 ulp is ~8x finer
    # than at magnitude ~14.  log_mu_sel_ref is then typically 0.
    if log_mu_sel_ref is not None:
        sel_log_factor = -nobs * (log_mu_sel_scaled - log_mu_sel_ref)
    else:
        sel_log_factor = -nobs * log_mu_sel_scaled
    _ = numpyro.factor('selfactor', jnp.where(sel_dead, _LOG_ZERO_FLOOR, sel_log_factor))

    # neff_sel is invariant under a constant weight shift, so the scaled
    # log_mu / log_mu2 pair is fine here.
    log_mu2 = jnp.squeeze(lse2_sel) - 2 * jnp.log(Ndraw)
    # 1 - exp(x) with x -> 0 from below is the dangerous case; -expm1 is the
    # accurate form and log(-expm1(x)) is finite for x < 0.
    x = 2 * log_mu_sel_scaled - jnp.log(Ndraw) - log_mu2
    log_s2 = log_mu2 + jnp.log(-jnp.expm1(jnp.minimum(x, -1e-7)))

    # --- n_eff guards ----------------------------------------------------
    min_neff = jnp.min(neff)
    # Per-event MC variance of the log likelihood is 1/n_eff.  An alive event
    # always has n_eff >= 1 (Cauchy-Schwarz); dead events carry neff = 0 and
    # are already handled by the _LOG_ZERO_FLOOR in their log likelihood, so
    # cap their contribution at the 1/n_eff = 1 maximum instead of inf.
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
        # min(0, -x) used by log_smooth_neff_boundary: ~0 well inside the
        # budget, asymptotically linear with slope -1/(0.05*budget) beyond it,
        # and -log(2) at the budget itself.  Smoothness matters here because a
        # gradient jump at the boundary causes HMC energy errors (divergences)
        # exactly when trajectories probe the guard.
        numpyro.factor(
            "neff_criteria",
            -jax.nn.softplus((mc_var - mc_variance_budget) / (0.05 * mc_variance_budget)),
        )
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

    _ = numpyro.deterministic(
        'mdNdmdVdt_fixed_qz',
        coords['m_grid'] * R * jnp.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)),
    )
    _ = numpyro.deterministic(
        'dNdqdVdt_fixed_mz',
        log_dN.mref * R * jnp.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)),
    )
    _ = numpyro.deterministic(
        'dNdVdt_fixed_mq',
        log_dN.mref * R * jnp.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])),
    )
    _ = numpyro.deterministic('hz', cosmo.h * cosmo.E(coords['z_grid']))


def recentering_baselines(model_args, ref_params, rng_seed=0, **model_kwargs):
    """Evaluate per-event log likelihoods and log_mu_sel once at a fixed
    reference point, for use as pop_cosmo_model kwargs.

    The baselines only need to be *near* typical posterior values -- whatever
    float32 numbers come out are exact constants once fixed, and the residual
    float32 error scales with how far the sampler wanders from them (O(10)
    nats over the posterior, up to O(1e3-1e4) during early warmup: still 100x
    less roundoff than no baseline at all).  So a plain float32 evaluation at
    the initialization point is entirely sufficient; no float64 pass needed.

    model_args: the positional arguments of pop_cosmo_model (data + prior).
    ref_params: dict of parameter values to condition on (e.g. the init/truth
        point).  Parameters not in the dict are drawn from the prior with
        `rng_seed`, so an empty dict still yields usable baselines.
    model_kwargs: forwarded to pop_cosmo_model (must match what the sampler
        will use, e.g. use_low_bump).  Any recentering / pdraw-scale kwargs
        in model_kwargs are stripped so this always evaluates the physical
        (unscaled) model.

    Returns a dict meant to be splatted into pop_cosmo_model:

      * ``loglike_ref`` (np.float64 (nobs,)): per-event baseline
      * ``log_pdraw_sel_scale`` (float): set to the physical ``log_mu_sel`` at
        the reference, so the scaled selection scalar sits at 0 there
      * ``log_mu_sel_ref`` (float): 0.0 -- the scaled selection recentering
        baseline after applying ``log_pdraw_sel_scale``
      * ``offset`` (float): ``sum(loglike_ref) - nobs*log_mu_sel_phys``;
        add to the centered potential to recover absolute log-likelihood
      * ``log_mu_sel_phys_ref`` (float): the unscaled ``log_mu_sel`` at the
        reference (same as ``log_pdraw_sel_scale``; kept under this name for
        clarity in logging)
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
        # A dead reference event carries a baseline of ~_LOG_ZERO_FLOOR, so its
        # residual is ~+1e6 whenever the event is alive -- which puts the huge
        # magnitude right back into the sum and defeats the recentering.
        import warnings
        warnings.warn(
            f"recentering_baselines: {n_dead} event(s) have (near-)zero "
            f"likelihood at the reference point; recentering will not help "
            f"until the reference point is moved inside the support."
        )
    nobs = loglike_ref.shape[0]
    return dict(
        loglike_ref=loglike_ref,
        log_pdraw_sel_scale=log_mu_sel_phys,
        log_mu_sel_ref=0.0,
        log_mu_sel_phys_ref=log_mu_sel_phys,
        offset=float(loglike_ref.sum() - nobs * log_mu_sel_phys),
    )

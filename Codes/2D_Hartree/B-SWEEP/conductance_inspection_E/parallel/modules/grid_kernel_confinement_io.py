from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from scipy.fft import rfftn, irfftn, next_fast_len


@dataclass
class ElectrostaticBackground2D:
    # Main simulation grid
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    dx: float
    dy: float

    # Enlarged surface-charge grid
    xS: np.ndarray
    yS: np.ndarray
    XS: np.ndarray
    YS: np.ndarray

    # Geometry / densities
    mask: np.ndarray
    nD: np.ndarray
    nS: np.ndarray
    top_gate_source_mask: np.ndarray

    # Electrostatic background
    U_ext: np.ndarray

    # Hartree callable
    hartree_from_density: object

    # Kernels
    K: np.ndarray
    K_ee_direct: np.ndarray
    K_ee_image: np.ndarray | None
    K_donor_direct: np.ndarray
    K_donor_image: np.ndarray | None


# ============================================================
# Grids and density profiles
# ============================================================

def make_grid_2d(p):
    x = np.linspace(-0.5 * p.Lx, 0.5 * p.Lx, p.Nx)
    y = np.linspace(-0.5 * p.Ly, 0.5 * p.Ly, p.Ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="xy")

    xS = np.arange(
        -0.5 * p.Lx - p.dS,
        0.5 * p.Lx + p.dS,
        dx,
    )
    yS = np.arange(
        -0.5 * p.Ly - p.dS,
        0.5 * p.Ly + p.dS,
        dy,
    )

    XS, YS = np.meshgrid(xS, yS, indexing="xy")

    return x, y, X, Y, dx, dy, xS, yS, XS, YS


def active_mask_2d(X, Y, p):
    if not p.use_elliptic_mask:
        return np.ones_like(X, dtype=bool)

    return (
        (X / (0.5 * p.Lx)) ** 2
        + (Y / (0.5 * p.Ly)) ** 2
        <= 1.0
    )


import numpy as np
from scipy.ndimage import gaussian_filter


def donor_density_profile_2d(X, Y, p):
    """
    Donor density profile

        n_D(x,y) = n_D^0(x,y) + envelope(x,y) * delta_n_D(x,y),

    where delta_n_D is a zero-mean Gaussian random field with approximately

        <delta_n_D(r) delta_n_D(r')>
            = donor_disorder_rms^2
              * exp[-|r-r'|^2 / (2 sigma^2)].

    The same disorder realization is obtained whenever the same seed
    and grid are used.
    """

    # ------------------------------------------------------------
    # 1. Smooth/abrupt donor envelope
    # ------------------------------------------------------------

    mx = 0.5 * p.Lx - p.donor_margin_nm
    my = 0.5 * p.Ly - p.donor_margin_nm

    if p.donor_smooth_nm <= 0:
        envelope = (
            (np.abs(X) <= mx) &
            (np.abs(Y) <= my)
        ).astype(float)

    else:
        s = p.donor_smooth_nm

        wx = 0.5 * (
            np.tanh((X + mx) / s)
            - np.tanh((X - mx) / s)
        )

        wy = 0.5 * (
            np.tanh((Y + my) / s)
            - np.tanh((Y - my) / s)
        )

        envelope = wx * wy

    # Clean donor profile
    n_clean = p.n_donor * envelope

    # ------------------------------------------------------------
    # 2. No disorder requested
    # ------------------------------------------------------------

    if (
        p.donor_disorder_rms <= 0
        or p.donor_disorder_sigma_nm <= 0
    ):
        return n_clean

    # ------------------------------------------------------------
    # 3. Grid spacing
    # ------------------------------------------------------------

    # Assuming meshgrid convention:
    # X varies along axis 1, Y along axis 0
    dx = np.mean(np.diff(X[0, :]))
    dy = np.mean(np.diff(Y[:, 0]))

    dx = abs(dx)
    dy = abs(dy)

    # ------------------------------------------------------------
    # 4. Generate white Gaussian noise
    # ------------------------------------------------------------

    rng = np.random.default_rng(p.donor_disorder_seed)

    white = rng.normal(
        loc=0.0,
        scale=1.0,
        size=X.shape
    )

    # ------------------------------------------------------------
    # 5. Gaussian filtering
    # ------------------------------------------------------------
    #
    # Important:
    #
    # If the FILTER itself has width s, its autocorrelation has width
    #
    #       sigma_corr = sqrt(2) * s.
    #
    # Therefore, to obtain
    #
    # C(r) ~ exp[-r^2/(2 sigma^2)]
    #
    # we filter with width
    #
    #       s = sigma / sqrt(2).
    #

    filter_sigma_y = (
        p.donor_disorder_sigma_nm /
        (np.sqrt(2.0) * dy)
    )

    filter_sigma_x = (
        p.donor_disorder_sigma_nm /
        (np.sqrt(2.0) * dx)
    )

    disorder = gaussian_filter(
        white,
        sigma=(filter_sigma_y, filter_sigma_x),
        mode="reflect"
    )

    # ------------------------------------------------------------
    # 6. Enforce zero mean and requested RMS
    # ------------------------------------------------------------
    #
    # Measure statistics well inside the donor region, avoiding the
    # smoothed boundaries.
    #

    bulk_mask = envelope > 0.99

    if np.any(bulk_mask):
        disorder -= np.mean(disorder[bulk_mask])

        rms = np.sqrt(
            np.mean(disorder[bulk_mask]**2)
        )
    else:
        disorder -= np.mean(disorder)
        rms = np.sqrt(np.mean(disorder**2))

    if rms > 0:
        disorder *= p.donor_disorder_rms / rms

    # ------------------------------------------------------------
    # 7. Apply donor envelope
    # ------------------------------------------------------------

    delta_nD = envelope * disorder

    n_total = n_clean + delta_nD

    return n_total

def clean_donor_density_profile_2d(X, Y, p):
    mx = 0.5 * p.Lx - p.donor_margin_nm
    my = 0.5 * p.Ly - p.donor_margin_nm

    if p.donor_smooth_nm <= 0:
        mask = (
            (np.abs(X) <= mx)
            & (np.abs(Y) <= my)
        )

        return p.n_donor * mask.astype(float)

    s = p.donor_smooth_nm

    wx = 0.5 * (
        np.tanh((X + mx) / s)
        - np.tanh((X - mx) / s)
    )

    wy = 0.5 * (
        np.tanh((Y + my) / s)
        - np.tanh((Y - my) / s)
    )

    return p.n_donor * wx * wy


def surf_density_profile_2d(XS, YS, dx, dy, p):
    mx = 0.5 * p.Lx + p.dS
    my = 0.5 * p.Ly + p.dS

    sigma = p.dS / 2.0

    norm_x = p.Lx / np.sqrt(2 * np.pi * sigma**2)
    norm_y = p.Ly / np.sqrt(2 * np.pi * sigma**2)

    n_L = (
        p.n_donor
        * np.exp(-0.5 * ((XS + mx) / sigma) ** 2)
        * norm_x
        / 2.0
    )

    n_R = (
        p.n_donor
        * np.exp(-0.5 * ((XS - mx + dx / 2.0) / sigma) ** 2)
        * norm_x
        / 2.0
    )

    n_bottom = (
        p.n_donor
        * np.exp(-0.5 * ((YS + my) / sigma) ** 2)
        * norm_y
        * 2.0
    )

    n_top = (
        p.n_donor
        * np.exp(-0.5 * ((YS - my + dy / 2.0) / sigma) ** 2)
        * norm_y
        * 2.0
    )

    return (n_L + n_R + n_bottom + n_top) / 4.0


# ============================================================
# Top-gate source masks
# ============================================================

def source_mask_first_half_x_2d(
    X,
    p,
    x_gate_edge_nm=None,
):
    """
    Source mask for a half-covered top gate.

    Image sources are included for

        x' <= x_gate_edge_nm.
    """

    if x_gate_edge_nm is None:
        x_gate_edge_nm = p.x_gate_edge_nm

    if x_gate_edge_nm is None:
        x_gate_edge_nm = 0.5 * (
            X.min() + X.max()
        )

    return (X <= x_gate_edge_nm).astype(float)


def source_mask_first_half_x_2d_2(
    X,
    p,
    x_gate_edge_nm=None,
):
    """
    Finite symmetric top-gate strip:

        -x_gate_edge_nm <= x <= +x_gate_edge_nm
    """

    if x_gate_edge_nm is None:
        x_gate_edge_nm = p.x_gate_edge_nm

    if x_gate_edge_nm is None:
        x_gate_edge_nm = 0.5 * (
            X.min() + X.max()
        )

    x_left = -x_gate_edge_nm
    x_right = +x_gate_edge_nm

    return (
        (X >= x_left)
        & (X <= x_right)
    ).astype(float)


def source_mask_first_half_x_2d_2_smooth(
    X,
    p,
    x_gate_edge_nm=None,
    fringe_nm=100.0,
):
    """
    Smooth finite top-gate source mask.

    Nominal gate region:

        -x_gate_edge_nm <= x <= +x_gate_edge_nm

    The gate edges are smoothed over the characteristic
    scale `fringe_nm`.
    """

    if x_gate_edge_nm is None:
        x_gate_edge_nm = p.x_gate_edge_nm

    if x_gate_edge_nm is None:
        x_gate_edge_nm = 0.5 * (
            X.min() + X.max()
        )

    x_left = -x_gate_edge_nm
    x_right = +x_gate_edge_nm

    mask = (
        np.arctan((X - x_left) / fringe_nm)
        - np.arctan((X - x_right) / fringe_nm)
    ) / np.pi

    return mask


# ============================================================
# Coulomb kernels
# ============================================================

def coulomb_kernel_2d_for_convolution(
    p,
    dx,
    dy,
    z_source_nm=0.0,
    z_image_nm=None,
):
    """
    Open-boundary Coulomb kernel for fftconvolve.

    Kernel shape:
        (2*Ny - 1, 2*Nx - 1)

    Kernel units:
        1 / nm
    """

    ix = np.arange(-(p.Nx - 1), p.Nx)
    iy = np.arange(-(p.Ny - 1), p.Ny)

    RX, RY = np.meshgrid(
        ix * dx,
        iy * dy,
        indexing="xy",
    )

    rho2 = RX**2 + RY**2

    if p.diagonal_cutoff_nm is None:
        a = np.sqrt(dx * dy / np.pi)
    else:
        a = p.diagonal_cutoff_nm

    z = (
        z_source_nm
        if z_image_nm is None
        else z_image_nm
    )

    R = np.sqrt(rho2 + z**2)

    if abs(z) < 1e-15:
        R = np.maximum(R, a)

    return 1.0 / R


def surf_coulomb_kernel_2d_for_convolution(
    p,
    Nx,
    Ny,
    dx,
    dy,
    z_source_nm=0.0,
    z_image_nm=None,
):
    ix = np.arange(-(Nx - 1), Nx)
    iy = np.arange(-(Ny - 1), Ny)

    RX, RY = np.meshgrid(
        ix * dx,
        iy * dy,
        indexing="xy",
    )

    rho2 = RX**2 + RY**2

    if p.diagonal_cutoff_nm is None:
        a = np.sqrt(dx * dy / np.pi)
    else:
        a = p.diagonal_cutoff_nm

    z = (
        z_source_nm
        if z_image_nm is None
        else z_image_nm
    )

    R = np.sqrt(rho2 + z**2)

    if abs(z) < 1e-15:
        R = np.maximum(R, a)

    return 1.0 / R


def convolve_open(field, K, dx, dy):
    return dx * dy * fftconvolve(
        field,
        K,
        mode="same",
    )


# ============================================================
# Full electrostatic background
# ============================================================

def make_kernel_and_background_2d(
    p,
    dist=None,
    fringe_nm=100.0,
    verbose=True,
):
    """
    Construct the electrostatic background required by
    the self-consistent Hartree calculation.

    Returns an ElectrostaticBackground2D object.
    """

    (
        x,
        y,
        X,
        Y,
        dx,
        dy,
        xS,
        yS,
        XS,
        YS,
    ) = make_grid_2d(p)

    mask = active_mask_2d(X, Y, p)

    nD = donor_density_profile_2d(
        X,
        Y,
        p,
    )

    nS = surf_density_profile_2d(
        XS,
        YS,
        dx,
        dy,
        p,
    )

    # --------------------------------------------------------
    # Direct kernels
    # --------------------------------------------------------

    K_ee_direct = (
        coulomb_kernel_2d_for_convolution(
            p,
            dx,
            dy,
            z_source_nm=0.0,
        )
    )

    K_donor_direct = (
        coulomb_kernel_2d_for_convolution(
            p,
            dx,
            dy,
            z_source_nm=p.donor_extra_dist_nm,
        )
    )

    K_donor_direct_down = (
        coulomb_kernel_2d_for_convolution(
            p,
            dx,
            dy,
            z_source_nm=-p.donor_extra_dist_nm,
        )
    )

    K_surf_direct = (
        surf_coulomb_kernel_2d_for_convolution(
            p,
            len(xS),
            len(yS),
            dx,
            dy,
            z_source_nm=0.0,
        )
    )

    # --------------------------------------------------------
    # Gate masks
    # --------------------------------------------------------

    top_gate_source_mask = (
        source_mask_first_half_x_2d_2_smooth(
            X,
            p,
            fringe_nm=fringe_nm,
        )
    )

    top_gate_source_mask_S = (
        source_mask_first_half_x_2d_2_smooth(
            XS,
            p,
            fringe_nm=fringe_nm,
        )
    )

    # --------------------------------------------------------
    # No top gate
    # --------------------------------------------------------

    if dist is None:

        K_ee_image = None
        K_donor_image = None

        U_donor = (
            -p.kappa0
            * convolve_open(
                nD,
                K_donor_direct,
                dx,
                dy,
            )
        )

        U_donor_down = (
            -p.kappa0
            * convolve_open(
                nD,
                K_donor_direct_down,
                dx,
                dy,
            )
        )

        U_surf = (
            p.kappa0
            * convolve_open(
                nS,
                K_surf_direct,
                dx,
                dy,
            )
        )

        def hartree_from_density(n):
            return (
                p.kappa0
                * convolve_open(
                    n,
                    K_ee_direct,
                    dx,
                    dy,
                )
            )

        K_effective_for_diagnostics = K_ee_direct

    # --------------------------------------------------------
    # Full top gate
    # --------------------------------------------------------

    elif not p.use_partial_top_gate:

        K_ee_image = (
            coulomb_kernel_2d_for_convolution(
                p,
                dx,
                dy,
                z_image_nm=2.0 * dist,
            )
        )

        K_donor_image = (
            coulomb_kernel_2d_for_convolution(
                p,
                dx,
                dy,
                z_image_nm=(
                    2.0 * dist
                    - p.donor_extra_dist_nm
                ),
            )
        )

        K_donor_image_down = (
            coulomb_kernel_2d_for_convolution(
                p,
                dx,
                dy,
                z_image_nm=(
                    2.0 * dist
                    + p.donor_extra_dist_nm
                ),
            )
        )

        K_surf_image = (
            surf_coulomb_kernel_2d_for_convolution(
                p,
                len(xS),
                len(yS),
                dx,
                dy,
                z_source_nm=2.0 * dist,
            )
        )

        U_donor = p.kappa0 * (
            -convolve_open(
                nD,
                K_donor_direct,
                dx,
                dy,
            )
            + convolve_open(
                nD,
                K_donor_image,
                dx,
                dy,
            )
        )

        U_donor_down = p.kappa0 * (
            -convolve_open(
                nD,
                K_donor_direct_down,
                dx,
                dy,
            )
            + convolve_open(
                nD,
                K_donor_image_down,
                dx,
                dy,
            )
        )

        U_surf = p.kappa0 * (
            convolve_open(
                nS,
                K_surf_direct,
                dx,
                dy,
            )
            - convolve_open(
                nS,
                K_surf_image,
                dx,
                dy,
            )
        )

        def hartree_from_density(n):
            return p.kappa0 * (
                convolve_open(
                    n,
                    K_ee_direct,
                    dx,
                    dy,
                )
                - convolve_open(
                    n,
                    K_ee_image,
                    dx,
                    dy,
                )
            )

        top_gate_source_mask = np.ones_like(
            X,
            dtype=float,
        )

        K_effective_for_diagnostics = (
            K_ee_direct
            - K_ee_image
        )

    # --------------------------------------------------------
    # Partial top gate
    # --------------------------------------------------------

    else:

        K_ee_image = (
            coulomb_kernel_2d_for_convolution(
                p,
                dx,
                dy,
                z_image_nm=2.0 * dist,
            )
        )

        K_donor_image = (
            coulomb_kernel_2d_for_convolution(
                p,
                dx,
                dy,
                z_image_nm=(
                    2.0 * dist
                    - p.donor_extra_dist_nm
                ),
            )
        )

        K_donor_image_down = (
            coulomb_kernel_2d_for_convolution(
                p,
                dx,
                dy,
                z_image_nm=(
                    2.0 * dist
                    + p.donor_extra_dist_nm
                ),
            )
        )

        K_surf_image = (
            surf_coulomb_kernel_2d_for_convolution(
                p,
                len(xS),
                len(yS),
                dx,
                dy,
                z_source_nm=2.0 * dist,
            )
        )

        U_surf = p.kappa0 * (
            convolve_open(
                nS,
                K_surf_direct,
                dx,
                dy,
            )
            - convolve_open(
                top_gate_source_mask_S * nS,
                K_surf_image,
                dx,
                dy,
            )
        )

        U_donor = p.kappa0 * (
            -convolve_open(
                nD,
                K_donor_direct,
                dx,
                dy,
            )
            + convolve_open(
                top_gate_source_mask * nD,
                K_donor_image,
                dx,
                dy,
            )
        )

        U_donor_down = p.kappa0 * (
            -convolve_open(
                nD,
                K_donor_direct_down,
                dx,
                dy,
            )
            + convolve_open(
                top_gate_source_mask * nD,
                K_donor_image_down,
                dx,
                dy,
            )
        )

        def hartree_from_density(n):
            return p.kappa0 * (
                convolve_open(
                    n,
                    K_ee_direct,
                    dx,
                    dy,
                )
                - convolve_open(
                    top_gate_source_mask * n,
                    K_ee_image,
                    dx,
                    dy,
                )
            )

        # Diagnostic only:
        # the masked interaction is not truly
        # translation invariant.
        K_effective_for_diagnostics = (
            K_ee_direct
            - K_ee_image
        )

    # --------------------------------------------------------
    # Crop enlarged surface grid to main simulation grid
    # --------------------------------------------------------

    indexS_xi = np.argmin(
        np.abs(xS - x[0])
    )

    indexS_xf = np.argmin(
        np.abs(xS - x[-1])
    )

    indexS_yi = np.argmin(
        np.abs(yS - y[0])
    )

    indexS_yf = np.argmin(
        np.abs(yS - y[-1])
    )

    U_surf_mask = U_surf[
        indexS_yi:indexS_yf + 1,
        indexS_xi:indexS_xf + 1,
    ]

    if verbose:
        print(
            "Surface crop:",
            indexS_xi,
            indexS_xf,
            indexS_yi,
            indexS_yf,
        )

        print(
            "grid:",
            len(x),
            len(xS),
            "U_surf_mask:",
            U_surf_mask.shape,
            "dx,dy:",
            dx,
            dy,
        )

    # --------------------------------------------------------
    # External confinement
    # --------------------------------------------------------

    U_ext = (
        0.5 * (
            U_donor
            + U_donor_down
        )
        + U_surf_mask
    )

    U_ext -= np.nanmin(
        U_ext[mask]
    )

    return ElectrostaticBackground2D(
        x=x,
        y=y,
        X=X,
        Y=Y,
        dx=dx,
        dy=dy,

        xS=xS,
        yS=yS,
        XS=XS,
        YS=YS,

        mask=mask,

        K=K_effective_for_diagnostics,

        nD=nD,
        nS=nS,

        U_ext=U_ext,

        top_gate_source_mask=top_gate_source_mask,

        hartree_from_density=hartree_from_density,

        K_ee_direct=K_ee_direct,
        K_ee_image=K_ee_image,

        K_donor_direct=K_donor_direct,
        K_donor_image=K_donor_image,
    )
    
    
# =============================================================================
# Reusable exact open-boundary FFT convolution
# =============================================================================

class PreparedFFTConvolver2D:
    """
    Exact linear convolution equivalent to
        dx*dy*fftconvolve(field, K, mode="same")
    but the FFT of K is computed only once.

    The crop convention matches scipy.signal.fftconvolve(..., mode="same").
    """
    def __init__(self, field_shape, kernel, dx, dy):
        self.field_shape = tuple(field_shape)
        self.kernel_shape = tuple(kernel.shape)
        self.scale = dx * dy

        full_shape = tuple(
            self.field_shape[i] + self.kernel_shape[i] - 1 for i in range(2)
        )
        self.fft_shape = tuple(next_fast_len(n) for n in full_shape)
        self.kernel_fft = rfftn(kernel, s=self.fft_shape)

        # scipy "same" crop: centered portion with shape of first input
        self.start = tuple((self.kernel_shape[i] - 1)//2 for i in range(2))
        self.stop = tuple(self.start[i] + self.field_shape[i] for i in range(2))

    def __call__(self, field):
        F = rfftn(field, s=self.fft_shape)
        full = irfftn(F * self.kernel_fft, s=self.fft_shape)

        # Only the mathematically non-zero full-convolution region matters.
        full_n0 = self.field_shape[0] + self.kernel_shape[0] - 1
        full_n1 = self.field_shape[1] + self.kernel_shape[1] - 1
        full = full[:full_n0, :full_n1]

        s0, s1 = self.start
        e0, e1 = self.stop
        return self.scale * full[s0:e0, s1:e1]


# =============================================================================
# Build ALL B-INDEPENDENT data once
# =============================================================================

def build_static_problem_2d(p, dist=None):
    x, y, X, Y, dx, dy, xS, yS, XS, YS = make_grid_2d(p)
    mask = active_mask_2d(X, Y, p)
    nD = donor_density_profile_2d(X, Y, p)
    nS = surf_density_profile_2d(XS, YS, dx, dy, p)

    K_ee_direct = coulomb_kernel_2d_for_convolution(
        p, dx, dy, z_source_nm=0.0
    )
    K_donor_direct = coulomb_kernel_2d_for_convolution(
        p, dx, dy, z_source_nm=p.donor_extra_dist_nm
    )
    K_surf_direct = surf_coulomb_kernel_2d_for_convolution(
        p, len(xS), len(yS), dx, dy, z_source_nm=0.0
    )
    K_donor_direct_down = coulomb_kernel_2d_for_convolution(
        p, dx, dy, z_source_nm=-p.donor_extra_dist_nm
    )

    # Same choice as the supplied cell.
    top_gate_source_mask = source_mask_first_half_x_2d_2_smooth(X, p)
    top_gate_source_mask_S = source_mask_first_half_x_2d_2_smooth(XS, p)

    # Prepare direct convolvers once.
    conv_ee_direct = PreparedFFTConvolver2D((p.Ny, p.Nx), K_ee_direct, dx, dy)
    conv_donor_direct = PreparedFFTConvolver2D((p.Ny, p.Nx), K_donor_direct, dx, dy)
    conv_donor_direct_down = PreparedFFTConvolver2D((p.Ny, p.Nx), K_donor_direct_down, dx, dy)
    conv_surf_direct = PreparedFFTConvolver2D(nS.shape, K_surf_direct, dx, dy)

    if dist is None:
        K_ee_image = None
        K_donor_image = None

        U_donor = -p.kappa0 * conv_donor_direct(nD)
        U_donor_down = -p.kappa0 * conv_donor_direct_down(nD)
        U_surf = p.kappa0 * conv_surf_direct(nS)

        def hartree_from_density(n):
            return p.kappa0 * conv_ee_direct(n)

        K_effective_for_diagnostics = K_ee_direct

    elif not p.use_partial_top_gate:
        K_ee_image = coulomb_kernel_2d_for_convolution(
            p, dx, dy, z_image_nm=2.0*dist
        )
        K_donor_image = coulomb_kernel_2d_for_convolution(
            p, dx, dy, z_image_nm=2.0*dist - p.donor_extra_dist_nm
        )
        K_donor_image_down = coulomb_kernel_2d_for_convolution(
            p, dx, dy, z_image_nm=2.0*dist + p.donor_extra_dist_nm
        )
        K_surf_image = surf_coulomb_kernel_2d_for_convolution(
            p, len(xS), len(yS), dx, dy, z_source_nm=2.0*dist
        )

        conv_ee_image = PreparedFFTConvolver2D((p.Ny, p.Nx), K_ee_image, dx, dy)
        conv_donor_image = PreparedFFTConvolver2D((p.Ny, p.Nx), K_donor_image, dx, dy)
        conv_donor_image_down = PreparedFFTConvolver2D((p.Ny, p.Nx), K_donor_image_down, dx, dy)
        conv_surf_image = PreparedFFTConvolver2D(nS.shape, K_surf_image, dx, dy)

        U_donor = -p.kappa0*conv_donor_direct(nD) + p.kappa0*conv_donor_image(nD)
        U_donor_down = -p.kappa0*conv_donor_direct_down(nD) + p.kappa0*conv_donor_image_down(nD)
        U_surf = p.kappa0*conv_surf_direct(nS) - p.kappa0*conv_surf_image(nS)

        def hartree_from_density(n):
            return p.kappa0 * (conv_ee_direct(n) - conv_ee_image(n))

        top_gate_source_mask = np.ones_like(X, dtype=float)
        top_gate_source_mask_S = np.ones_like(XS, dtype=float)
        K_effective_for_diagnostics = K_ee_direct - K_ee_image

    else:
        K_ee_image = coulomb_kernel_2d_for_convolution(
            p, dx, dy, z_image_nm=2.0*dist
        )
        K_donor_image = coulomb_kernel_2d_for_convolution(
            p, dx, dy, z_image_nm=2.0*dist - p.donor_extra_dist_nm
        )
        K_donor_image_down = coulomb_kernel_2d_for_convolution(
            p, dx, dy, z_image_nm=2.0*dist + p.donor_extra_dist_nm
        )
        K_surf_image = surf_coulomb_kernel_2d_for_convolution(
            p, len(xS), len(yS), dx, dy, z_source_nm=2.0*dist
        )

        conv_ee_image = PreparedFFTConvolver2D((p.Ny, p.Nx), K_ee_image, dx, dy)
        conv_donor_image = PreparedFFTConvolver2D((p.Ny, p.Nx), K_donor_image, dx, dy)
        conv_donor_image_down = PreparedFFTConvolver2D((p.Ny, p.Nx), K_donor_image_down, dx, dy)
        conv_surf_image = PreparedFFTConvolver2D(nS.shape, K_surf_image, dx, dy)

        U_surf = (
            p.kappa0 * conv_surf_direct(nS)
            - p.kappa0 * conv_surf_image(top_gate_source_mask_S*nS)
        )
        U_donor = (
            -p.kappa0 * conv_donor_direct(nD)
            + p.kappa0 * conv_donor_image(top_gate_source_mask*nD)
        )
        U_donor_down = (
            -p.kappa0 * conv_donor_direct_down(nD)
            + p.kappa0 * conv_donor_image_down(top_gate_source_mask*nD)
        )

        def hartree_from_density(n):
            return p.kappa0 * (
                conv_ee_direct(n)
                - conv_ee_image(top_gate_source_mask*n)
            )

        K_effective_for_diagnostics = K_ee_direct - K_ee_image

    indexS_xi = np.argmin(np.abs(xS-x[0]))
    indexS_xf = np.argmin(np.abs(xS-x[-1]))
    indexS_yi = np.argmin(np.abs(yS-y[0]))
    indexS_yf = np.argmin(np.abs(yS-y[-1]))
    U_surf_mask = U_surf[indexS_yi:indexS_yf+1, indexS_xi:indexS_xf+1]

    print(
        indexS_xi, indexS_xf, indexS_yi, indexS_yf,
        len(x), len(xS), U_surf_mask.shape, dx, dy
    )

    U_ext = .5*(U_donor.copy()+U_donor_down.copy()) + U_surf_mask.copy()
    U_ext -= np.nanmin(U_ext[mask])

    return {
        "x": x, "y": y, "X": X, "Y": Y,
        "dx": dx, "dy": dy,
        "xS": xS, "yS": yS, "XS": XS, "YS": YS,
        "mask": mask,
        "active_flat_idx": np.flatnonzero(mask.ravel()),
        "active_count": int(np.sum(mask)),
        "K": K_effective_for_diagnostics,
        "nD": nD,
        "nS": nS,
        "U_ext": U_ext,
        "top_gate_source_mask": top_gate_source_mask,
        "hartree_from_density": hartree_from_density,
        "K_ee_direct": K_ee_direct,
        "K_ee_image": K_ee_image,
        "K_donor_direct": K_donor_direct,
        "K_donor_image": K_donor_image,
        "dist": dist,
    }

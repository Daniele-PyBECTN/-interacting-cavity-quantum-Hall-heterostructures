import numpy as np

from scipy.special import erf
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter

from .grid_kernel_confinement_io import make_kernel_and_background_2d


# ============================================================
# Landau-level spectrum
# ============================================================

def landau_energies(p):
    """
    Spinful Landau-level energies.

    If g_spin = 0, each orbital Landau level is returned twice.
    """
    n = np.arange(p.n_LL, dtype=float)

    E = p.hbar_omega_c * (n + 0.5)
    EZ = p.g_spin * p.hbar_omega_c

    if abs(EZ) < 1e-15:
        return np.repeat(E, 2)

    return np.sort(
        np.r_[
            E - 0.5 * EZ,
            E + 0.5 * EZ,
        ]
    )


def spinless_landau_energies(p):
    """Spinless Landau-level energies."""
    n = np.arange(p.n_LL, dtype=float)

    return p.hbar_omega_c * (n + 0.5)

def prepare_landau_context(p):
    return {
        "E": spinless_landau_energies(p),
        "Gamma": p.Gamma,
        "degeneracy": 1.0/(2*np.pi*p.lB**2),
        "lB": p.lB,
        "hbar_omega_c": p.hbar_omega_c,
    }


def density_values_from_U(U_values, mu, ll):
    arg = (
        mu - U_values[..., None] - ll["E"]
    ) / (np.sqrt(2.0)*ll["Gamma"])
    occ = 0.5*(1.0 + erf(arg))
    return ll["degeneracy"] * np.sum(occ, axis=-1)

# ============================================================
# Density, filling, compressibility
# ============================================================
def density_from_potential_2d(
    U,
    mu,
    p,
    mask=None,
    ll=None,
):
    if ll is None:
        ll = prepare_landau_context(p)

    n = density_values_from_U(
        U,
        mu,
        ll,
    )

    if mask is not None:
        n = np.where(
            mask,
            n,
            0.0,
        )

    return n


def old_density_from_potential_2d(
    U,
    mu,
    p,
    mask=None,
):
    """
    Electron density for a broadened spinless Landau-level DOS.
    """

    E = spinless_landau_energies(p)

    arg = (
        mu
        - U[..., None]
        - E[None, None, :]
    ) / (
        np.sqrt(2.0)
        * p.Gamma
    )

    occ = 0.5 * (
        1.0
        + erf(arg)
    )

    degeneracy = 1.0 / (
        2.0
        * np.pi
        * p.lB**2
    )

    n = degeneracy * np.sum(
        occ,
        axis=-1,
    )

    if mask is not None:
        n = np.where(
            mask,
            n,
            0.0,
        )

    return n

def compressibility_from_potential_2d(
    U,
    mu,
    p,
    mask=None,
    ll=None,
):
    if ll is None:
        ll = prepare_landau_context(p)

    z = (
        mu
        - U[..., None]
        - ll["E"]
    ) / ll["Gamma"]

    gaussian_dos = (
        np.exp(-0.5 * z*z)
        /
        (
            np.sqrt(2*np.pi)
            * ll["Gamma"]
        )
    )

    comp = (
        ll["degeneracy"]
        * np.sum(
            gaussian_dos,
            axis=-1,
        )
    )

    if mask is not None:
        comp = np.where(
            mask,
            comp,
            0.0,
        )

    return comp

def old_compressibility_from_potential_2d(
    U,
    mu,
    p,
    mask=None,
):
    """
    Local thermodynamic compressibility dn/dmu.
    """

    E = spinless_landau_energies(p)

    z = (
        mu
        - U[..., None]
        - E[None, None, :]
    ) / p.Gamma

    gaussian_dos = (
        np.exp(-0.5 * z**2)
        /
        (
            np.sqrt(2.0 * np.pi)
            * p.Gamma
        )
    )

    degeneracy = 1.0 / (
        2.0
        * np.pi
        * p.lB**2
    )

    comp = degeneracy * np.sum(
        gaussian_dos,
        axis=-1,
    )

    if mask is not None:
        comp = np.where(
            mask,
            comp,
            0.0,
        )

    return comp


def filling_from_density_2d(n, p):
    """Local filling factor."""
    return (
        2.0
        * np.pi
        * p.lB**2
        * n
    )


# ============================================================
# Density normalization / chemical potential
# ============================================================

def active_area(mask, dx, dy):
    """Physical area of the active simulation region."""
    return (
        dx
        * dy
        * np.sum(mask)
    )


def average_density_2d(
    n,
    mask,
    dx,
    dy,
):
    """Average electron density inside the active region."""

    return (
        dx
        * dy
        * np.sum(n[mask])
        / active_area(
            mask,
            dx,
            dy,
        )
    )


def find_mu_for_target_2d(
    U,
    p,
    mask,
    dx,
    dy,
    verbose=False,
):
    """
    Determine the chemical potential such that the spatially
    averaged density equals p.n_target.
    """

    E = spinless_landau_energies(p)

    U_active = U[mask]

    lo = (
        np.min(U_active)
        + np.min(E)
        - 20.0 * p.Gamma
        - 5.0 * p.hbar_omega_c
    )

    hi = (
        np.max(U_active)
        + np.max(E)
        + 20.0 * p.Gamma
        + 5.0 * p.hbar_omega_c
    )

    def f(mu):
        n = density_from_potential_2d(
            U,
            mu,
            p,
            mask=mask,
        )

        return (
            average_density_2d(
                n,
                mask,
                dx,
                dy,
            )
            - p.n_target
        )

    flo = f(lo)
    fhi = f(hi)

    if verbose:
        print(
            "mu bracket:",
            lo,
            hi,
            flo,
            fhi,
        )

    if flo > 0 or fhi < 0:
        raise RuntimeError(
            "Cannot bracket mu: "
            f"f(lo)={flo:g}, "
            f"f(hi)={fhi:g}. "
            "Increase n_LL or check target density."
        )

    return brentq(
        f,
        lo,
        hi,
        xtol=1e-12,
        rtol=1e-12,
        maxiter=100,
    )

def find_mu_for_target_2d_optimized(
    U,
    p,
    static,
    ll,
    verbose=False,
):
    """
    Optimized chemical-potential search for B sweeps.

    Density is evaluated only on active grid points during Brent's
    root search, avoiding repeated full-grid allocations.
    """

    U_active = U.ravel()[static["active_flat_idx"]]
    E = ll["E"]

    lo = (
        np.min(U_active)
        + np.min(E)
        - 20.0 * ll["Gamma"]
        - 5.0 * ll["hbar_omega_c"]
    )

    hi = (
        np.max(U_active)
        + np.max(E)
        + 20.0 * ll["Gamma"]
        + 5.0 * ll["hbar_omega_c"]
    )

    def f(mu):
        n_active = density_values_from_U(
            U_active,
            mu,
            ll,
        )

        return (
            np.mean(n_active)
            - p.n_target
        )

    flo = f(lo)
    fhi = f(hi)

    if verbose:
        print(
            "mu bracket",
            lo,
            hi,
            flo,
            fhi,
        )

    if flo > 0 or fhi < 0:
        raise RuntimeError(
            f"Cannot bracket mu: "
            f"f(lo)={flo:g}, "
            f"f(hi)={fhi:g}. "
            "Increase n_LL or check target density."
        )

    return brentq(
        f,
        lo,
        hi,
        xtol=1e-12,
        rtol=1e-12,
        maxiter=100,
    )

# ============================================================
# Numerical utilities
# ============================================================

def gaussian_smooth_2d(
    U,
    sigma_nm,
    dx,
    dy,
):
    """Gaussian smooth using a physical width specified in nm."""

    if sigma_nm <= 0:
        return U

    sigma_pix = (
        sigma_nm / dy,
        sigma_nm / dx,
    )

    return gaussian_filter(
        U,
        sigma=sigma_pix,
        mode="nearest",
    )


# ============================================================
# Self-consistent Hartree solver
# ============================================================

def solve_self_consistent_2d(
    p,
    dist=None,
    U_initial=None,
    verbose=False,
    background=None,
):
    """
    Solve the self-consistent 2D Hartree problem.

    Parameters
    ----------
    p
        Parameter dataclass.

    dist : float or None
        Top-gate distance in nm.

    U_initial : ndarray or None
        Optional initial potential.

    verbose : bool
        Print convergence information.

    background : ElectrostaticBackground2D or None
        Optional precomputed electrostatic background.
        If None, it is constructed internally.

    Returns
    -------
    dict
        Self-consistent electrostatic and electronic quantities.
    """

    # --------------------------------------------------------
    # Electrostatic background
    # --------------------------------------------------------

    if background is None:
        bg = make_kernel_and_background_2d(
            p,
            dist=dist,
            verbose=False,
        )
    else:
        bg = background

    x = bg.x
    y = bg.y
    X = bg.X
    Y = bg.Y

    dx = bg.dx
    dy = bg.dy

    mask = bg.mask

    U_ext = bg.U_ext
    hartree_from_density = bg.hartree_from_density

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    if U_initial is None:
        U = U_ext.copy()
    else:
        U = U_initial.copy()

    U -= np.nanmin(
        U[mask]
    )

    mu = find_mu_for_target_2d(
        U,
        p,
        mask,
        dx,
        dy,
    )

    n = density_from_potential_2d(
        U,
        mu,
        p,
        mask=mask,
    )

    # --------------------------------------------------------
    # Self-consistent iteration
    # --------------------------------------------------------

    history = []

    for it in range(p.max_iter):

        # Hartree potential generated by current density
        U_H = hartree_from_density(n)

        # New self-consistent potential
        U_new = (
            U_ext
            + U_H
        )

        U_new = gaussian_smooth_2d(
            U_new,
            p.smooth_sigma_nm,
            dx,
            dy,
        )

        U_new -= np.nanmin(
            U_new[mask]
        )

        # Linear mixing
        U_mixed = (
            (1.0 - p.mix) * U
            + p.mix * U_new
        )

        # Keep the inactive region finite
        U_mixed = np.where(
            mask,
            U_mixed,
            np.nanmax(U_mixed[mask]),
        )

        # Recalculate chemical potential
        mu_new = find_mu_for_target_2d(
            U_mixed,
            p,
            mask,
            dx,
            dy,
        )

        # Updated density
        n_new = density_from_potential_2d(
            U_mixed,
            mu_new,
            p,
            mask=mask,
        )

        # Density convergence criterion
        dn = np.sqrt(
            np.mean(
                (
                    n_new[mask]
                    - n[mask]
                ) ** 2
            )
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        if (
            it % 25 == 0
            or it == p.max_iter - 1
        ):

            n_avg = average_density_2d(
                n_new,
                mask,
                dx,
                dy,
            )

            history.append(
                (
                    it,
                    mu_new,
                    n_avg,
                    dn,
                )
            )

            if verbose:
                print(
                    f"it={it:4d}, "
                    f"mu={mu_new:.6g}, "
                    f"n_avg={n_avg:.6g}, "
                    f"dn={dn:.3e}"
                )

        # Accept iteration
        U = U_mixed
        mu = mu_new
        n = n_new

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        if dn < p.tol:

            history.append(
                (
                    it,
                    mu,
                    average_density_2d(
                        n,
                        mask,
                        dx,
                        dy,
                    ),
                    dn,
                )
            )

            if verbose:
                print(
                    f"Converged after {it + 1} iterations."
                )

            break

    # --------------------------------------------------------
    # Final observables
    # --------------------------------------------------------

    comp = compressibility_from_potential_2d(
        U,
        mu,
        p,
        mask=mask,
    )

    nu = filling_from_density_2d(
        n,
        p,
    )

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,

        "dx": dx,
        "dy": dy,

        "mask": mask,

        "dist": dist,

        "K": bg.K,

        "K_ee_direct": bg.K_ee_direct,
        "K_ee_image": bg.K_ee_image,

        "K_donor_direct": bg.K_donor_direct,
        "K_donor_image": bg.K_donor_image,

        "top_gate_source_mask": (
            bg.top_gate_source_mask
        ),

        "n_donor_profile": bg.nD,
        "n_surface_profile": bg.nS,

        "U_ext": U_ext,

        "U": U,
        "mu": mu,
        "n": n,
        "nu": nu,

        "compressibility": comp,

        "history": np.asarray(history),
    }
    



def solve_self_consistent_2d_optimized(
    p,
    static,
    U_initial=None,
    verbose=False,
):
    """
    Same fixed-density Hartree iteration as the supplied cell.

    Difference: all B-independent electrostatics and kernel FFTs come from
    `static`; B-dependent LL quantities are prepared once here.
    """
    mask = static["mask"]
    dx, dy = static["dx"], static["dy"]
    U_ext = static["U_ext"]
    hartree_from_density = static["hartree_from_density"]
    ll = prepare_landau_context(p)

    U = U_ext.copy() if U_initial is None else U_initial.copy()
    U -= np.nanmin(U[mask])

    mu = find_mu_for_target_2d_optimized(U, p, static, ll)
    n = density_from_potential_2d(U, mu, p, mask=mask, ll=ll)

    history = []
    for it in range(p.max_iter):
        U_H = hartree_from_density(n)
        U_new = U_ext + U_H
        U_new = gaussian_smooth_2d(U_new, p.smooth_sigma_nm, dx, dy)
        U_new -= np.nanmin(U_new[mask])

        U_mixed = (1.0 - p.mix)*U + p.mix*U_new
        U_mixed = np.where(mask, U_mixed, np.nanmax(U_mixed[mask]))

        mu_new = find_mu_for_target_2d_optimized(U_mixed, p, static, ll)
        n_new = density_from_potential_2d(
            U_mixed, mu_new, p, mask=mask, ll=ll
        )

        dn = np.sqrt(np.mean((n_new[mask] - n[mask])**2))

        if it % 25 == 0 or it == p.max_iter - 1:
            n_avg = average_density_2d(n_new, mask, dx, dy)
            history.append((it, mu_new, n_avg, dn))
            if verbose:
                print(
                    f"it={it:4d}, mu={mu_new:.6g}, "
                    f"n_avg={n_avg:.6g}, dn={dn:.3e}"
                )

        U, mu, n = U_mixed, mu_new, n_new

        if dn < p.tol:
            history.append(
                (it, mu, average_density_2d(n, mask, dx, dy), dn)
            )
            break

    comp = compressibility_from_potential_2d(
        U, mu, p, mask=mask, ll=ll
    )

    return {
        "x": static["x"], "y": static["y"],
        "X": static["X"], "Y": static["Y"],
        "dx": dx, "dy": dy, "mask": mask,
        "dist": static["dist"], "K": static["K"],
        "K_ee_direct": static["K_ee_direct"],
        "K_ee_image": static["K_ee_image"],
        "K_donor_direct": static["K_donor_direct"],
        "K_donor_image": static["K_donor_image"],
        "top_gate_source_mask": static["top_gate_source_mask"],
        "n_donor_profile": static["nD"],
        "U_ext": U_ext,
        "U": U,
        "mu": mu,
        "n": n,
        "nu": filling_from_density_2d(n, p),
        "compressibility": comp,
        "history": np.array(history),
    }
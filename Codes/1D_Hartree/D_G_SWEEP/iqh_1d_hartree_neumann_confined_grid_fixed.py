"""Utilities for the effective-1D Thomas-Fermi/Hartree IQH calculation.

Refactored from gpt_TF_confinement_gate_sweep_compressibility_upgraded(2).ipynb.

Kernel choices
--------------
- 'top_gate': original translationally-invariant top-gate image kernel.
- 'top_open_box': grounded top gate + grounded (Dirichlet) sidewalls.
- 'top_gate_neumann': grounded top gate + Neumann sidewalls.

All lengths are nm, energies meV, and areal densities nm^-2.
The dimensionless kernel convention is K = 2*pi*G, so locally K ~ -log(r).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import brentq
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


VALID_KERNELS = ("top_gate", "top_open_box", "top_gate_neumann")


@dataclass
class Params:
    """Simulation parameters.

    Deliberately has no user-facing defaults: instantiate this class explicitly
    in the notebook so that all physical/numerical choices are visible there.
    """
    # geometry
    d: float
    Nx: int
    margin: float
    d_S: float

    # material / electrostatics
    eps_r: float
    n_donor: float
    n_s: float
    n_target: float
    donor_b_fraction: float
    donor_smooth_nm: float
    donor_z_offset_nm: float

    # magnetic field / LLs
    B: float
    g_spin: float
    n_LL: int
    gamma_rel: float
    use_B_scaling: bool

    # self-consistency
    max_iter: int
    mix: float
    tol: float
    smooth_sigma_nm: float

    @property
    def kappa0(self):
        return 1440.0 / self.eps_r

    @property
    def hbar_omega_c(self):
        return 1.76 * self.B

    @property
    def lB(self):
        return 25.658 / np.sqrt(self.B)

    @property
    def Gamma(self):
        if self.use_B_scaling:
            return self.gamma_rel * self.hbar_omega_c * np.sqrt(10.0 / self.B)
        return self.gamma_rel * self.hbar_omega_c



def make_grid(p: Params, include_surface_grid=True,
              electron_halfwidth_nm=None,
              include_donor_grid=False):
    """Build simulation grids.

    Parameters
    ----------
    electron_halfwidth_nm : float or None
        If None, use the original electronic half-width p.d.
        For the Neumann geometry this should be x_N = x_N0 - W_depl,
        so the 2DEG grid itself is confined inside the Neumann walls.
    include_surface_grid : bool
        Build the larger surface-charge grid. This is False for Neumann.
    include_donor_grid : bool
        Build an independent donor grid spanning the original physical
        donor/2DEG width p.d. This is used in the Neumann branch because
        donors remain extended over the original width while electrons
        are confined to |x| < x_N.
    """
    d_e = p.d if electron_halfwidth_nm is None else float(electron_halfwidth_nm)
    if d_e <= 0:
        raise ValueError("electron_halfwidth_nm must be positive.")

    x = np.linspace(-p.margin*d_e, p.margin*d_e, p.Nx)
    dx = x[1] - x[0]

    x_S = np.arange(-p.d_S, p.d_S, dx) if include_surface_grid else None

    if include_donor_grid:
        x_D = np.linspace(-p.margin*p.d, p.margin*p.d, p.Nx)
        dx_D = x_D[1] - x_D[0]
    else:
        x_D = x
        dx_D = dx

    return x, dx, x_S, x_D, dx_D

def csg_kernel(x, dextra, d, diagonal_cutoff=None):
    """Original dimensionless CSG logarithmic kernel on [-d,d]."""
    x = np.asarray(x)
    dx_grid = abs(x[1] - x[0]) if diagonal_cutoff is None else diagonal_cutoff
    xi = x[:, None]
    xj = x[None, :]
    d2 = d*d
    ai = np.maximum(d2 - xi*xi, 0.0)
    aj = np.maximum(d2 - xj*xj, 0.0)
    num = np.sqrt(ai*aj) + d2 - xi*xj
    den = d*np.sqrt((xi - xj)**2 + dextra**2)
    den[den == 0.0] = d*dx_grid
    return np.log(np.abs(num/den))


def bare_kernel(x, dextra, diagonal_cutoff=None):
    """Original free logarithmic line-charge kernel, including its cutoff convention."""
    x = np.asarray(x)
    dx_grid = abs(x[1] - x[0]) if diagonal_cutoff is None else diagonal_cutoff
    xi = x[:, None]
    xj = x[None, :]
    num = np.sqrt((xi - xj)**2 + dextra**2)/dx_grid
    num[num == 0.0] = 2
    return -np.log(np.abs(num))


def kernel_top_gate(x, d, dist, dextra, diagonal_cutoff=None):
    """Original notebook top-gate kernel, intentionally preserved exactly."""
    if dist is None:
        raise ValueError("kernel_top_gate requires a finite gate distance 'dist'.")
    x = np.asarray(x)
    dx_grid = abs(x[1] - x[0]) if diagonal_cutoff is None else diagonal_cutoff
    xi = x[:, None]
    xj = x[None, :]
    delta2 = (xi - xj)**2
    delta2[delta2 == 0.0] = dx_grid**2
    K0 = bare_kernel(x, dextra, diagonal_cutoff=diagonal_cutoff)
    K_image_correction = 0.5*np.log(delta2 + (2.0*dist + dextra)**2)
    return K0 + K_image_correction



def kernel_top_gate_cross(x_obs, x_src, dist, dextra, reference_length_nm):
    """Standard single-top-gate kernel from an independent source grid to
    an observation grid.

    This is the rectangular-grid analogue of ``kernel_top_gate``.  The
    additive constant ``log(reference_length_nm)`` matches the convention
    used by the original notebook kernel; it is spatially uniform and is
    removed when U_ext is shifted by its minimum.

    Parameters
    ----------
    x_obs, x_src : array_like
        Observation and source coordinates [nm].
    dist : float
        Distance to the grounded top gate [nm].
    dextra : float
        Signed vertical source offset used by the original notebook [nm].
    reference_length_nm : float
        Positive length entering the same dimensionless-log convention as
        the original square-grid kernel.
    """
    if dist is None:
        raise ValueError("kernel_top_gate_cross requires finite 'dist'.")
    x_obs = np.asarray(x_obs, dtype=float)
    x_src = np.asarray(x_src, dtype=float)
    ell = float(reference_length_nm)
    if ell <= 0:
        raise ValueError("reference_length_nm must be positive.")

    delta = x_obs[:, None] - x_src[None, :]
    direct2 = delta**2 + float(dextra)**2
    image2 = delta**2 + (2.0*float(dist) + float(dextra))**2

    # For donor planes dextra != 0 in the present model, so direct2 is
    # nonsingular. Keep a defensive cutoff nevertheless.
    direct2 = np.maximum(direct2, np.finfo(float).tiny)
    image2 = np.maximum(image2, np.finfo(float).tiny)

    return np.log(ell) + 0.5*np.log(image2/direct2)

def _log_cosh_minus_cos(s, h, L):
    """Stable log[cosh(pi*h/L)-cos(pi*s/L)] for scalar h and array s."""
    s = np.asarray(s, dtype=float)
    t = np.pi * abs(float(h)) / L
    theta = np.pi * s / L
    c = np.cos(theta)
    if t > 40.0:
        # cosh(t)-cos(theta) = exp(t)/2 * [1 - 2 cos(theta)e^-t + e^-2t]
        em = np.exp(-t)
        return t - np.log(2.0) + np.log1p(-2.0*c*em + em*em)
    val = np.cosh(t) - c
    return np.log(np.maximum(val, np.finfo(float).tiny))


def _validate_bounded_geometry(x, halfwidth_nm):
    xmax = float(np.max(np.abs(x)))
    if halfwidth_nm <= xmax:
        raise ValueError(
            f"Sidewall half-width {halfwidth_nm:.6g} nm must be strictly larger than "
            f"all source/observation coordinates (max |x| = {xmax:.6g} nm)."
        )


def kernel_top_open_box(x, dist, dextra=0.0, halfwidth_nm=None, diagonal_cutoff=None,
                        stabilize_constant=False):
    r"""Grounded top gate plus grounded (Dirichlet) sidewalls x=+-halfwidth_nm.

    The source plane is z'=-dextra and the observation plane is z=0, matching the
    sign convention of the original kernel_top_gate.  The returned dimensionless
    kernel has the same local normalization K ~ -log(r).
    """
    if dist is None:
        raise ValueError("top_open_box requires a finite top-gate distance.")
    x = np.asarray(x, dtype=float)
    if halfwidth_nm is None:
        halfwidth_nm = float(np.max(np.abs(x))) + abs(x[1]-x[0])
    halfwidth_nm = float(halfwidth_nm)
    _validate_bounded_geometry(x, halfwidth_nm)
    L = 2.0*halfwidth_nm
    dx_grid = abs(x[1]-x[0]) if diagonal_cutoff is None else float(diagonal_cutoff)

    xi = x[:, None]
    xj = x[None, :]
    s_minus = xi - xj
    if abs(dextra) < 1e-15:
        s_minus = s_minus.copy()
        np.fill_diagonal(s_minus, dx_grid)
    s_plus = xi + xj + 2.0*halfwidth_nm

    a = abs(float(dextra))
    b = 2.0*float(dist) + float(dextra)
    if b <= 0:
        raise ValueError("Image distance 2*dist+dextra must be positive.")

    la_d = _log_cosh_minus_cos(s_minus, a, L)
    lb_d = _log_cosh_minus_cos(s_plus, a, L)
    la_i = _log_cosh_minus_cos(xi-xj, b, L)
    lb_i = _log_cosh_minus_cos(s_plus, b, L)

    # 2*pi*G_D = 1/2 log[A_plus(direct) A_minus(image) /
    #                         (A_minus(direct) A_plus(image))]
    K = 0.5*(lb_d + la_i - la_d - lb_i)
    if stabilize_constant:
        K -= K[K.shape[0]//2, 0]
    return K


def kernel_top_gate_neumann(x, dist, dextra=0.0, halfwidth_nm=None,
                            diagonal_cutoff=None, stabilize_constant=True):
    r"""Grounded top gate plus homogeneous Neumann sidewalls x=+-halfwidth_nm.

    This is the full Neumann Green function including the n=0 transverse mode.
    In closed form the zero mode is automatically contained in the logarithmic
    expression.  The dimensionless normalization is K=2*pi*G ~ -log(r).
    """
    if dist is None:
        raise ValueError("top_gate_neumann requires a finite top-gate distance.")
    x = np.asarray(x, dtype=float)
    if halfwidth_nm is None:
        halfwidth_nm = float(np.max(np.abs(x))) + abs(x[1]-x[0])
    halfwidth_nm = float(halfwidth_nm)
    _validate_bounded_geometry(x, halfwidth_nm)
    L = 2.0*halfwidth_nm
    dx_grid = abs(x[1]-x[0]) if diagonal_cutoff is None else float(diagonal_cutoff)

    xi = x[:, None]
    xj = x[None, :]
    s_minus_direct = xi-xj
    if abs(dextra) < 1e-15:
        s_minus_direct = s_minus_direct.copy()
        np.fill_diagonal(s_minus_direct, dx_grid)
    s_minus_image = xi-xj
    s_plus = xi+xj+2.0*halfwidth_nm

    a = abs(float(dextra))
    b = 2.0*float(dist) + float(dextra)
    if b <= 0:
        raise ValueError("Image distance 2*dist+dextra must be positive.")

    la_d = _log_cosh_minus_cos(s_minus_direct, a, L)
    lb_d = _log_cosh_minus_cos(s_plus, a, L)
    la_i = _log_cosh_minus_cos(s_minus_image, b, L)
    lb_i = _log_cosh_minus_cos(s_plus, b, L)

    # Full Neumann solution (including n=0):
    # 2*pi*G_N = 1/2 log[A_minus(image) A_plus(image) /
    #                         (A_minus(direct) A_plus(direct))].
    K = 0.5*(la_i + lb_i - la_d - lb_d)
    if stabilize_constant:
        # Neumann zero mode contains a large x-independent contribution ~ dist/L.
        # Removing one scalar from the full matrix changes only the arbitrary
        # additive potential constant and greatly improves conditioning for huge dist.
        K -= K[K.shape[0]//2, 0]
    return K


def resolve_sidewall_halfwidth(p: Params, sidewall_halfwidth_nm=None, depletion_width_nm=0.0):
    """Resolve x_w or x_N.  By default the wall is just outside the x_S domain.

    depletion_width_nm shifts an explicitly supplied/reference wall inward:
        x_boundary = base_halfwidth - depletion_width_nm.
    The result is validated later against every grid on which the bounded kernel is used.
    """
    base = p.d_S if sidewall_halfwidth_nm is None else float(sidewall_halfwidth_nm)
    hw = base - float(depletion_width_nm)
    if hw <= 0:
        raise ValueError("Resolved sidewall half-width must be positive.")
    return hw


def build_kernel(x, p: Params, dist, dextra=0.0, kernel_kind="top_gate",
                 sidewall_halfwidth_nm=None, depletion_width_nm=0.0,
                 diagonal_cutoff=None):
    kernel_kind = str(kernel_kind).lower()
    if kernel_kind not in VALID_KERNELS:
        raise ValueError(f"kernel_kind must be one of {VALID_KERNELS}, got {kernel_kind!r}")
    if kernel_kind == "top_gate":
        return kernel_top_gate(x, p.d, dist, dextra, diagonal_cutoff=diagonal_cutoff)

    hw = resolve_sidewall_halfwidth(p, sidewall_halfwidth_nm, depletion_width_nm)
    if kernel_kind == "top_open_box":
        return kernel_top_open_box(x, dist, dextra, hw, diagonal_cutoff)
    return kernel_top_gate_neumann(x, dist, dextra, hw, diagonal_cutoff)


def donor_density_profile(x, p: Params):
    b = p.donor_b_fraction * p.d
    if p.donor_smooth_nm <= 0:
        return p.n_donor * (np.abs(x) <= b).astype(float)
    s = p.donor_smooth_nm
    return 0.5*p.n_donor*(np.tanh((x+b)/s)-np.tanh((x-b)/s))


def s_density_profile(x_S, p: Params):
    b = abs(p.d_S-p.d)/2.0
    return p.n_s*p.d_S/np.sqrt(2*np.pi*b**2) * (
        np.exp(-0.5*np.abs((x_S-p.d_S)/b)**2) +
        np.exp(-0.5*np.abs((x_S+p.d_S)/b)**2)
    )


def donor_potential_from_kernel(K, n_donor_x, dx, p: Params):
    return -p.kappa0 * dx * (K @ n_donor_x)


def gate_potential_zero(x):
    return np.zeros_like(x)


def donor_potential_analytic(x, p: Params):
    E0 = np.pi*p.kappa0*2*p.n_target*p.d
    return -E0*np.sqrt(np.maximum(1.0-(x/p.d)**2, 0.0))



def make_kernel_and_background(p: Params, dist=None, kernel_kind="top_gate",
                               sidewall_halfwidth_nm=None, depletion_width_nm=0.0,
                               verbose=False):
    """Build grids, Coulomb kernels, charge profiles, and external potential.

    Conventions
    -----------
    top_gate
        Original notebook geometry. Electron, donor, and surface-charge
        electrostatics use the standard top-gate kernel.

    top_open_box
        Electron, donor, and surface-charge electrostatics use the
        Dirichlet top-open-box kernel.

    top_gate_neumann
        * The actual 2DEG grid is confined to |x| < x_N, where
          x_N = sidewall_halfwidth_nm - depletion_width_nm.
        * The Neumann+top-gate Green function is used ONLY for the
          electron-electron Hartree interaction.
        * Donors remain distributed over the original physical interval
          set by p.d and are integrated on an independent donor grid x_D.
        * Donor -> electron electrostatics uses the ordinary single-top-gate
          kernel, evaluated between x_D (source) and x (observation).
        * Surface charges are completely omitted and x_S is not built.
    """
    kernel_kind = str(kernel_kind).lower()
    if kernel_kind not in VALID_KERNELS:
        raise ValueError(f"kernel_kind must be one of {VALID_KERNELS}, got {kernel_kind!r}")

    dist0_d = -abs(float(p.donor_z_offset_nm))

    if kernel_kind == "top_gate_neumann":
        xN = resolve_sidewall_halfwidth(
            p, sidewall_halfwidth_nm, depletion_width_nm
        )
        # The physical Neumann boundary is at +/-xN.  The grid itself
        # stays slightly inside through p.margin, exactly as in the
        # original notebook.
        x, dx, x_S, x_D, dx_D = make_grid(
            p,
            include_surface_grid=False,
            electron_halfwidth_nm=xN,
            include_donor_grid=True,
        )
        use_surface_charges = False
    else:
        x, dx, x_S, x_D, dx_D = make_grid(
            p,
            include_surface_grid=True,
            electron_halfwidth_nm=p.d,
            include_donor_grid=False,
        )
        use_surface_charges = True
        
    if p.n_s == None and kernel_kind != "top_gate_neumann":
        x, dx, x_S, x_D, dx_D = make_grid(
            p,
            include_surface_grid=False,
            electron_halfwidth_nm=p.d,
            include_donor_grid=False,
        )
        use_surface_charges = False

    if dist is None:
        if kernel_kind != "top_gate":
            raise ValueError("Bounded kernels require a finite top-gate distance.")
        K = csg_kernel(x, 0.0, p.d)
        K_donor = csg_kernel(x, dist0_d, p.d)
        K_donor_down = csg_kernel(x, -dist0_d, p.d)
        if use_surface_charges:
            K_s = csg_kernel(x_S, 0.0, p.d_S)
        else:
            K_s = None
    else:
        # Electron-electron Hartree kernel follows the selected geometry.
        K = build_kernel(
            x, p, dist, 0.0, kernel_kind,
            sidewall_halfwidth_nm, depletion_width_nm, dx
        )

        if kernel_kind == "top_gate_neumann":
            # Donors remain on the original physical donor grid x_D,
            # while the potential is evaluated on the smaller electron grid x.
            K_donor = kernel_top_gate_cross(
                x, x_D, dist, dist0_d, reference_length_nm=dx_D
            )
            K_donor_down = kernel_top_gate_cross(
                x, x_D, dist, -dist0_d, reference_length_nm=dx_D
            )
            K_s = None
        else:
            K_donor = build_kernel(
                x, p, dist, dist0_d, kernel_kind,
                sidewall_halfwidth_nm, depletion_width_nm, dx
            )
            K_donor_down = build_kernel(
                x, p, dist, -dist0_d, kernel_kind,
                sidewall_halfwidth_nm, depletion_width_nm, dx
            )
            if use_surface_charges:
                #K_s = build_kernel(
                #    x_S, p, dist, 0.0, kernel_kind,
                #    sidewall_halfwidth_nm, depletion_width_nm, dx
                #)
                K_s = build_kernel(
                    x_S, p, dist, 0.0, kernel_kind,
                    sidewall_halfwidth_nm, depletion_width_nm, dx
                )
            else:
                K_s = None

    # IMPORTANT: in the Neumann branch the donor density is defined on x_D,
    # not on the confined electron grid x.
    nD = donor_density_profile(x_D, p)
    U_donor = donor_potential_from_kernel(K_donor, nD, dx_D, p)
    U_donor_down = donor_potential_from_kernel(K_donor_down, nD, dx_D, p)

    if use_surface_charges:
        nS = s_density_profile(x_S, p)
        U_s_full = donor_potential_from_kernel(K_s, -nS, dx, p)
        indexS_i = np.argmin(np.abs(x_S-x[0]))
        indexS_f = np.argmin(np.abs(x_S-x[-1]))
        U_s = U_s_full[indexS_i:indexS_f+1]
        if len(U_s) != len(x):
            raise RuntimeError(
                f"Surface-potential crop length mismatch: "
                f"len(U_s)={len(U_s)}, len(x)={len(x)}."
            )
        if verbose:
            print(indexS_i, indexS_f, len(x), len(x_S), len(U_s), dx)
    else:
        if kernel_kind == "top_gate_neumann":
            nS = None
            U_s = np.zeros_like(x)
            if verbose:
                print("Neumann geometry:")
                print(f"  electron half-width x_N = {xN:.6g} nm")
                print(f"  electron grid max |x| = {np.max(np.abs(x)):.6g} nm")
                print(f"  donor-grid half-width = {p.d:.6g} nm")
                print(f"  donor grid max |x_D| = {np.max(np.abs(x_D)):.6g} nm")
                print("  surface charges disabled; no x_S grid constructed.")
                print("  e-e kernel: Neumann + top gate.")
                print("  donor -> electron kernel: ordinary single top gate.")
        else:
            nS = None
            U_s = np.zeros_like(x)
            if verbose:
                print("No Surface charges")
                print(f"  electron grid max |x| = {np.max(np.abs(x)):.6g} nm")
                print(f"  donor-grid half-width = {p.d:.6g} nm")
                print(f"  donor grid max |x_D| = {np.max(np.abs(x_D)):.6g} nm")
                print("  surface charges disabled; no x_S grid constructed.")

    U_ext = 0.5*(U_donor+U_donor_down) + gate_potential_zero(x) + U_s
    U_ext -= np.min(U_ext)

    # Keep the historical return values first, and append x_D/dx_D so
    # diagnostics can distinguish the donor and electron grids.
    return x, dx, x_S, K, nD, nS, U_ext, x_D, dx_D

def landau_energies(p: Params):
    n = np.arange(p.n_LL, dtype=float)
    E = p.hbar_omega_c*(n+0.5)
    EZ = p.g_spin*p.hbar_omega_c
    if abs(EZ) < 1e-15:
        return np.repeat(E, 2)
    return np.sort(np.r_[E-0.5*EZ, E+0.5*EZ])


def density_from_potential(U, mu, p: Params):
    E = landau_energies(p)
    arg = (mu-U[:, None]-E[None, :])/(np.sqrt(2.0)*p.Gamma)
    occ = 0.5*(1.0+erf(arg))
    degeneracy_per_spin = 1.0/(2*np.pi*p.lB**2)
    return degeneracy_per_spin*np.sum(occ, axis=1)


def compressibility_from_potential(U, mu, p: Params):
    E = landau_energies(p)
    z = (mu-U[:, None]-E[None, :])/p.Gamma
    gaussian_dos = np.exp(-0.5*z*z)/(np.sqrt(2*np.pi)*p.Gamma)
    degeneracy_per_spin = 1.0/(2*np.pi*p.lB**2)
    return degeneracy_per_spin*np.sum(gaussian_dos, axis=1)


def filling_from_density(n, p: Params):
    return 2*np.pi*p.lB**2*n


def average_density(n, dx, p: Params):
    return dx*np.sum(n)/(2*p.d)


def find_mu_for_target(U, p: Params, dx, verbose=False):
    E = landau_energies(p)
    lo = np.min(U)+np.min(E)-20*p.Gamma-5*p.hbar_omega_c
    hi = np.max(U)+np.max(E)+20*p.Gamma+5*p.hbar_omega_c

    def f(mu):
        return average_density(density_from_potential(U, mu, p), dx, p)-p.n_target

    flo, fhi = f(lo), f(hi)
    if verbose:
        print("mu bracket", lo, hi, flo, fhi)
    if flo > 0 or fhi < 0:
        raise RuntimeError(
            f"Cannot bracket mu: f(lo)={flo:g}, f(hi)={fhi:g}. Increase n_LL or check target density."
        )
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100)


def gaussian_smooth_realspace(y, sigma_nm, dx):
    if sigma_nm <= 0:
        return y
    k = 2*np.pi*np.fft.fftfreq(y.size, d=dx)
    return np.fft.ifft(np.fft.fft(y)*np.exp(-0.5*(sigma_nm*k)**2)).real


def solve_self_consistent(p: Params, dist=None, U_initial=None, verbose=False,
                          kernel_kind="top_gate", sidewall_halfwidth_nm=None,
                          depletion_width_nm=0.0):
    x, dx_local, x_S, K, nD, nS, U_ext, x_D, dx_D = make_kernel_and_background(
        p, dist=dist, kernel_kind=kernel_kind,
        sidewall_halfwidth_nm=sidewall_halfwidth_nm,
        depletion_width_nm=depletion_width_nm,
        verbose=verbose,
    )
    U = U_ext.copy() if U_initial is None else U_initial.copy()
    U -= np.min(U)
    mu = find_mu_for_target(U, p, dx_local)
    n = density_from_potential(U, mu, p)

    history = []
    for it in range(p.max_iter):
        U_H = p.kappa0*dx_local*(K @ n)
        U_new = U_ext+U_H
        U_new = gaussian_smooth_realspace(U_new, p.smooth_sigma_nm, dx_local)
        U_new -= np.min(U_new)
        U_mixed = (1-p.mix)*U+p.mix*U_new
        mu_new = find_mu_for_target(U_mixed, p, dx_local)
        n_new = density_from_potential(U_mixed, mu_new, p)
        dn = np.sqrt(dx_local*np.mean((n_new-n)**2))
        if it % 25 == 0 or it == p.max_iter-1:
            history.append((it, mu_new, average_density(n_new, dx_local, p), dn))
            if verbose:
                print(f"it={it:4d}, mu={mu_new:.6g}, n_avg={average_density(n_new, dx_local, p):.6g}, dn={dn:.3e}")
        U, mu, n = U_mixed, mu_new, n_new
        if dn < p.tol:
            history.append((it, mu, average_density(n, dx_local, p), dn))
            break

    return {
        "x": x, "x_S": x_S, "dx": dx_local, "dist": dist,
        "kernel_kind": kernel_kind,
        "sidewall_halfwidth_nm": None if kernel_kind == "top_gate" else resolve_sidewall_halfwidth(p, sidewall_halfwidth_nm, depletion_width_nm),
        "K": K, "n_donor_profile": nD,
        "x_donor": x_D,
        "dx_donor": dx_D, "n_surface_charges": nS,
        "surface_charges_enabled": (kernel_kind != "top_gate_neumann"),
        "donor_kernel_kind": ("top_gate" if kernel_kind == "top_gate_neumann" else kernel_kind),
        "U_ext": U_ext, "U": U, "mu": mu, "n": n,
        "nu": filling_from_density(n, p),
        "compressibility": compressibility_from_potential(U, mu, p),
        "history": np.array(history),
    }


def plot_solution(res, p: Params):
    x, U, mu, n, nu = res["x"], res["U"], res["mu"], res["n"], res["nu"]
    hwc = p.hbar_omega_c
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(x*1e-3, (U-np.amin(U))/hwc, color="tab:red", lw=2, label=r"$U/\hbar\omega_c$")
    ax.axhline(mu/hwc, color="tab:green", lw=1.5, label=r"$\mu/\hbar\omega_c$")
    ax.set_xlabel(r"$x$ [$\mu$m]"); ax.set_ylabel(r"energy / $\hbar\omega_c$"); ax.grid(True); ax.set_ylim(0, 10)
    ax2 = ax.twinx(); ax2.plot(x*1e-3, n/1e-3, color="tab:blue", lw=2)
    ax2.set_ylabel(r"$n$ [$10^{11}$ cm$^{-2}$]", color="tab:blue"); ax.legend(loc="upper left"); fig.tight_layout()

    plt.figure(figsize=(11, 3.2)); plt.plot(x*1e-3, nu, lw=2)
    for m in range(0, int(np.nanmax(nu))+2): plt.axhline(m, color="0.7", lw=0.8)
    plt.xlabel(r"$x$ [$\mu$m]"); plt.ylabel(r"$\nu(x)=2\pi\ell_B^2 n(x)$"); plt.grid(True); plt.tight_layout()

    plt.figure(figsize=(11, 3.2)); plt.plot(x*1e-3, nu, lw=2)
    for m in range(0, int(np.nanmax(nu))+2): plt.axhline(m, color="0.7", lw=0.8)
    plt.xlabel(r"$x$ [$\mu$m]"); plt.ylabel(r"$\nu(x)=2\pi\ell_B^2 n(x)$")
    plt.xlim(-p.d*1e-3, -p.d*1e-3*.7); plt.grid(True); plt.tight_layout()


def save_animation(anim, mp4_path, gif_path, fps=2):
    mp4_path, gif_path = Path(mp4_path), Path(gif_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True); gif_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is not None:
        try:
            anim.save(str(mp4_path), writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=140)
            print("saved", mp4_path); return str(mp4_path)
        except Exception as exc:
            print("MP4 failed, falling back to GIF:", exc)
    else:
        print("ffmpeg not found; saving GIF instead.")
    anim.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=110)
    print("saved", gif_path); return str(gif_path)


def make_density_potential_animation(sweep, p, out_prefix="density_potential_vs_dist", outdir="results"):
    x_um = sweep[0]["x"]*1e-3; hwc = p.hbar_omega_c
    fig, axU = plt.subplots(figsize=(9, 4.2)); axn = axU.twinx()
    U_all = np.array([(r["U"]-np.amin(r["U"])) for r in sweep])/hwc
    Uext_all = np.array([r["U_ext"] for r in sweep])/hwc
    n_all = np.array([r["n"] for r in sweep])/1e-3
    mu_all = np.array([(r["mu"]-np.amin(r["U"])) for r in sweep])/hwc
    lineU, = axU.plot([], [], lw=2, label=r"$U_{\rm eff}/\hbar\omega_c$", color="red")
    lineUext, = axU.plot([], [], lw=1.2, ls=":", label=r"$U_{\rm donor}/\hbar\omega_c$")
    lineMu = axU.axhline(0, lw=1.4, ls="--", label=r"$\mu/\hbar\omega_c$", color="green")
    linen, = axn.plot([], [], lw=2, label=r"$n$", color="blue")
    axU.set_xlim(x_um.min(), x_um.max()); axU.set_ylim(0, 1000); axn.set_ylim(np.nanmin(n_all)*.95, np.nanmax(n_all)*1.05)
    axU.set_xlabel(r"$x$ [$\mu$m]"); axU.set_ylabel(r"energy / $\hbar\omega_c$"); axn.set_ylabel(r"$n$ [$10^{11}$ cm$^{-2}$]"); axU.grid(True)
    title = axU.set_title(""); axU.legend(loc="upper left"); axn.legend(loc="upper right")
    def update(i):
        r=sweep[i]; lineU.set_data(x_um,U_all[i]); lineUext.set_data(x_um,Uext_all[i]); lineMu.set_ydata([mu_all[i],mu_all[i]]); linen.set_data(x_um,n_all[i]); title.set_text(f"dist = {r['dist']:.1f} nm")
        return lineU,lineUext,lineMu,linen,title
    anim=FuncAnimation(fig,update,frames=len(sweep),interval=700,blit=False)
    path=save_animation(anim,Path(outdir)/f"{out_prefix}.mp4",Path(outdir)/f"{out_prefix}.gif"); plt.close(fig); return anim,path


def make_filling_animation(sweep, p, out_prefix="local_filling_vs_dist", outdir="results"):
    x_um=sweep[0]["x"]*1e-3; nu_all=np.array([r["nu"] for r in sweep])
    fig,ax=plt.subplots(figsize=(9,4.2)); line,=ax.plot([],[],lw=2)
    ax.set_xlim(x_um.min(),x_um.max()); ax.set_ylim(0,np.nanmax(nu_all)*1.08); ax.set_xlabel(r"$x$ [$\mu$m]"); ax.set_ylabel(r"$\nu(x)=2\pi\ell_B^2 n(x)$"); ax.grid(True)
    for m in range(0,int(np.nanmax(nu_all))+2): ax.axhline(m,color="0.75",lw=.7)
    title=ax.set_title("")
    def update(i):
        r=sweep[i]; line.set_data(x_um,nu_all[i]); title.set_text(f"dist = {r['dist']:.1f} nm"); return line,title
    anim=FuncAnimation(fig,update,frames=len(sweep),interval=700,blit=False)
    path=save_animation(anim,Path(outdir)/f"{out_prefix}.mp4",Path(outdir)/f"{out_prefix}.gif"); plt.close(fig); return anim,path

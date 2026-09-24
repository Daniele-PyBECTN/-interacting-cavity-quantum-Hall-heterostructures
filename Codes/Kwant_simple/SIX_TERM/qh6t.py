"""Minimal six-terminal integer-QH transport tools for Kwant.

Units: length nm, energy meV, magnetic field tesla.
Lead order: 0 source(left), 1 drain(right), 2 bottom-left, 3 top-left,
            4 bottom-right, 5 top-right.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import math
import numpy as np
from scipy.ndimage import gaussian_filter

PHI0_T_NM2 = 4135.667696          # h/e in T nm^2
HBAR2_OVER_2ME_NM2_MEV = 38.0998212
KB_MEV_K = 0.08617333262
H_OVER_E2_OHM = 25812.80745


def hopping_meV(a_nm, m_eff=0.067):
    return HBAR2_OVER_2ME_NM2_MEV / (m_eff * a_nm**2)


def cyclotron_meV(B_T, m_eff=0.067):
    return 0.11576764 * B_T / m_eff


@dataclass(frozen=True)
class Geometry:
    Lx_nm: float = 1000.0
    Ly_nm: float = 400.0
    arm_width_nm: float = 120.0
    arm_length_nm: float = 180.0
    arm_centers_nm: Tuple[float, float] = (-250.0, 250.0)
    a_nm: float = 10.0


@dataclass(frozen=True)
class Disorder:
    amplitude_meV: float = 0.02
    correlation_nm: float = 30.0
    seed: int = 12345


@dataclass(frozen=True)
class Barriers:
    A: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    width_nm: float = 80.0
    smooth_nm: float = 15.0
    height_cyclotron: float = 1.0


@dataclass(frozen=True)
class Model:
    geometry: Geometry = field(default_factory=Geometry)
    disorder: Disorder = field(default_factory=Disorder)
    barriers: Barriers = field(default_factory=Barriers)
    B_T: float = 1.0
    m_eff: float = 0.067


def _inside(x, y, g: Geometry):
    if abs(x) <= g.Lx_nm/2 and abs(y) <= g.Ly_nm/2:
        return True
    for xc in g.arm_centers_nm:
        if abs(x-xc) <= g.arm_width_nm/2 and g.Ly_nm/2 <= abs(y) <= g.Ly_nm/2 + g.arm_length_nm:
            return True
    return False


def _lead_windows(g: Geometry):
    # side, transverse center, width
    x1, x2 = g.arm_centers_nm
    return (("left", 0., g.Ly_nm), ("right", 0., g.Ly_nm),
            ("bottom", x1, g.arm_width_nm), ("top", x1, g.arm_width_nm),
            ("bottom", x2, g.arm_width_nm), ("top", x2, g.arm_width_nm))


def _distance_from_contact(x, y, side, g):
    if side == "left": return x + g.Lx_nm/2
    if side == "right": return g.Lx_nm/2 - x
    if side == "bottom": return y + g.Ly_nm/2 + g.arm_length_nm
    return g.Ly_nm/2 + g.arm_length_nm - y


def _smooth_box(d, width, smooth):
    """Box on 0<d<width with tanh-smoothed edges."""
    if width <= 0: return 0.0
    s = max(float(smooth), 1e-9)
    return 0.5*(np.tanh( (d-100)/s) - np.tanh(((d-100)-width)/s))


def barrier_is_on(nu, A, eps=1e-12):
    """Signed version of the requested interval around each positive integer.

    A>0: [N(1-A), N]; A<0: [N, N(1-A)]; A=0: off.
    Thus the literal requested rule is recovered for A>0, while negative A
    naturally places the window on the opposite side of the integer.
    """
    if abs(A) < eps or nu <= 0: return False
    # Check nearby integers rather than floor(nu), which fails just above N for A<0.
    n0 = max(1, int(math.floor(nu))-1)
    n1 = int(math.ceil(nu))+1
    for N in range(n0, n1+1):
        lo, hi = sorted((float(N), (float(N))*(1.0-A)))
        if lo-eps <= nu <= hi+eps:
            return True
    return False


def barrier_switches(nu, A_tuple):
    return np.asarray([barrier_is_on(nu, a) for a in A_tuple], dtype=float)


def make_disorder(model: Model):
    g, d = model.geometry, model.disorder
    # integer-coordinate bounding box, including arms
    nx = int(round(g.Lx_nm/g.a_nm)) + 1
    ny = int(round((g.Ly_nm + 2*g.arm_length_nm)/g.a_nm)) + 1
    rng = np.random.default_rng(d.seed)
    z = rng.normal(size=(ny, nx))
    if d.correlation_nm > 0:
        z = gaussian_filter(z, d.correlation_nm/g.a_nm, mode="reflect")
    z -= z.mean(); std = z.std()
    if std > 0: z /= std
    return d.amplitude_meV*z


def _disorder_at(site, disorder, g):
    x, y = site.pos
    ix = int(round((x + g.Lx_nm/2)/g.a_nm))
    iy = int(round((y + g.Ly_nm/2 + g.arm_length_nm)/g.a_nm))
    if 0 <= iy < disorder.shape[0] and 0 <= ix < disorder.shape[1]:
        return float(disorder[iy, ix])
    return 0.0


def build_system(model: Model, disorder=None):
    """Build once. nu is a runtime parameter controlling contact barriers."""
    import kwant
    g, bc = model.geometry, model.barriers
    a = g.a_nm; t = hopping_meV(a, model.m_eff)
    lat = kwant.lattice.square(a, norbs=1)
    disorder = make_disorder(model) if disorder is None else disorder
    windows = _lead_windows(g)
    barrier_height = bc.height_cyclotron * cyclotron_meV(model.B_T, model.m_eff)

    def shape(pos): return _inside(pos[0], pos[1], g)

    def onsite(site, nu):
        x, y = site.pos
        v = _disorder_at(site, disorder, g)
        sw = barrier_switches(float(nu), bc.A)
        for j, (side, center, width) in enumerate(windows):
            transverse = y if side in ("left", "right") else x
            if abs(transverse-center) <= width/2 + 2*bc.smooth_nm:
                dist = _distance_from_contact(x, y, side, g)
                v += sw[j] * barrier_height * _smooth_box(dist, bc.width_nm, bc.smooth_nm)
        return 4*t + v

    syst = kwant.Builder()
    syst[lat.shape(shape, (0, 0))] = onsite
    syst[lat.neighbors()] = -t

    # Leads are ideal continuations and carry the same B through Peierls phases.
    for side, center, width in windows:
        if side == "left": sym = (-a, 0); start = (-g.Lx_nm/2, center); lead_shape = lambda p, c=center,w=width: abs(p[1]-c) <= w/2
        elif side == "right": sym = (a, 0); start = (g.Lx_nm/2, center); lead_shape = lambda p, c=center,w=width: abs(p[1]-c) <= w/2
        elif side == "bottom": sym = (0, -a); start = (center, -g.Ly_nm/2-g.arm_length_nm); lead_shape = lambda p, c=center,w=width: abs(p[0]-c) <= w/2
        else: sym = (0, a); start = (center, g.Ly_nm/2+g.arm_length_nm); lead_shape = lambda p, c=center,w=width: abs(p[0]-c) <= w/2
        lead = kwant.Builder(kwant.TranslationalSymmetry(sym))
        lead[lat.shape(lead_shape, start)] = 4*t
        lead[lat.neighbors()] = -t
        syst.attach_lead(lead)
    return syst, disorder


def finalize_with_B(builder, B_T, nleads=6):
    import kwant
    fsyst, gauge = kwant.builder.add_peierls_phase(builder)
    b = 2*B_T / PHI0_T_NM2
    #params = gauge(b, *([b]*nleads))
    params = gauge(b, *([0.0] * nleads))
    return fsyst, params


def build_dos_system(model: Model, disorder=None):
    """Barrier-free central rectangle for bulk DOS -> filling calibration."""
    import kwant
    g = model.geometry; a=g.a_nm; t=hopping_meV(a, model.m_eff)
    lat=kwant.lattice.square(a, norbs=1)
    disorder = make_disorder(model) if disorder is None else disorder
    def shape(pos): return abs(pos[0]) <= g.Lx_nm/2 and abs(pos[1]) <= g.Ly_nm/2
    def onsite(site): return 4*t + _disorder_at(site, disorder, g)
    s=kwant.Builder(); s[lat.shape(shape,(0,0))]=onsite; s[lat.neighbors()]=-t
    return s


def compute_dos(model: Model, disorder, energy_resolution_meV=None, num_vectors=80):
    import kwant
    builder = build_dos_system(model, disorder)
    fs, gauge = kwant.builder.add_peierls_phase(builder)
    b = 2*model.B_T / PHI0_T_NM2
    params = gauge(b)
    res = energy_resolution_meV or cyclotron_meV(model.B_T, model.m_eff)/80
    if num_vectors == None:
        sd = kwant.kpm.SpectralDensity(fs, params=params, energy_resolution=res)
    else:
        sd = kwant.kpm.SpectralDensity(fs, params=params, num_vectors=num_vectors, energy_resolution=res)
    E=np.asarray(sd.energies).real; rho=np.asarray(sd.densities).real
    order=np.argsort(E)
    return E[order], rho[order], fs.graph.num_nodes


def filling_from_dos(mu, dos_E, dos_rho, B_T, area_nm2, T_K):
    """nu(mu)=N_e/N_phi using KPM total DOS and finite-T Fermi occupation."""
    mu=np.atleast_1d(mu).astype(float); E=np.asarray(dos_E,float); rho=np.asarray(dos_rho,float)
    kT=KB_MEV_K*float(T_K)
    out=np.empty_like(mu)
    for i,m in enumerate(mu):
        if kT <= 0:
            f=(E <= m).astype(float)
        else:
            z=np.clip((E-m)/kT, -700, 700); f=1/(np.exp(z)+1)
        Ne=np.trapz(rho*f, E)
        Nphi=area_nm2*B_T/PHI0_T_NM2
        out[i]=Ne/Nphi
    return out


def transmission_matrix(sm):
    n=len(sm.lead_info); T=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j: T[i,j]=sm.transmission(i,j)
    return T


def conductance_matrix(T):
    # I_i=(e^2/h) sum_j [T_{j i} V_i - T_{i j} V_j]
    G=-T.copy()
    for i in range(len(T)): G[i,i]=np.sum(T[:,i])
    return G


def voltages(T, source=0, drain=1, ground=1):
    G=conductance_matrix(T); I=np.zeros(len(T)); I[source]=1.; I[drain]=-1.
    keep=[i for i in range(len(T)) if i != ground]
    V=np.zeros(len(T)); V[keep]=np.linalg.solve(G[np.ix_(keep,keep)], I[keep])
    return V


def resistances(T):
    V=voltages(T)
    return dict(Rxy_left=V[3]-V[2], Rxy_right=V[5]-V[4],
                Rxx_top=V[3]-V[5], Rxx_bottom=V[2]-V[4],
                Rxy=0.5*((V[3]-V[2])+(V[5]-V[4])),
                Rxx=0.5*((V[3]-V[5])+(V[2]-V[4])))


def _fermi_derivative(E, mu, T_K):
    """Return -df/dE in 1/meV, evaluated stably."""
    kT = KB_MEV_K * float(T_K)
    if kT <= 0:
        raise ValueError("T_K must be > 0 for thermal averaging")
    z = np.clip((np.asarray(E, float) - float(mu)) / kT, -700, 700)
    ez = np.exp(z)
    return ez / (kT * (1.0 + ez)**2)


def transport_sweep(fsyst, magnetic_params, energies_meV, fillings, show_progress=True,
                    T_K=0.0, thermal_window_kT=5.0):
    """Six-terminal transport sweep, with optional finite-T averaging.

    The expensive zero-T transmission matrices T(E) are computed once on the
    supplied energy grid.  For T_K > 0 the corresponding conductance matrices
    are convolved with -df/dE around each chemical potential, then the
    Landauer-Buettiker equations are solved.  Thus temperature adds essentially
    no Kwant solves.

    Barrier switching follows the filling associated with each sampled energy.
    This is natural for the precomputed E <-> nu sweep and keeps the calculation
    reusable.  At T=0 the result is exactly the original calculation.
    """
    import kwant
    energies=np.asarray(energies_meV,float); fillings=np.asarray(fillings,float)
    if energies.ndim != 1 or fillings.shape != energies.shape:
        raise ValueError("energies_meV and fillings must be 1D arrays of equal length")
    if np.any(np.diff(energies) <= 0):
        raise ValueError("energies_meV must be strictly increasing")

    keys=("Rxy_left","Rxy_right","Rxx_top","Rxx_bottom","Rxy","Rxx")
    transmissions=[]

    # Expensive part: one Kwant solve per energy, exactly as before.
    for i,(E,nu) in enumerate(zip(energies,fillings)):
        if show_progress and (i % max(1,len(energies)//20)==0):
            print(f"{i+1}/{len(energies)}  E={E:.4f} meV  nu={nu:.3f}")
        try:
            sm=kwant.smatrix(fsyst, energy=float(E),
                             params={**magnetic_params,"nu":float(nu)})
            transmissions.append(transmission_matrix(sm))
        except Exception as exc:
            print(f"warning at i={i}, E={E:.6g}: {exc}")
            transmissions.append(np.full((6,6),np.nan))

    T0=np.asarray(transmissions)
    out={k:np.full(len(energies),np.nan) for k in keys}

    if T_K <= 0:
        T_used=T0.copy()
    else:
        # Thermal averaging is linear in the conductance/transmission matrix.
        # Restrict to +/- thermal_window_kT for speed and to avoid distant NaNs.
        T_used=np.full_like(T0, np.nan)
        kT=KB_MEV_K*float(T_K)
        for i,mu in enumerate(energies):
            mask=(np.abs(energies-mu) <= thermal_window_kT*kT)
            mask &= np.all(np.isfinite(T0), axis=(1,2))
            idx=np.flatnonzero(mask)
            if len(idx) < 2:
                continue
            Ewin=energies[idx]
            w=_fermi_derivative(Ewin, mu, T_K)
            norm=np.trapz(w, Ewin)
            if norm <= 0:
                continue
            # Integral T(E)(-df/dE)dE, normalized for the finite/truncated grid.
            T_used[i]=np.trapz(T0[idx]*w[:,None,None], Ewin, axis=0)/norm

    # Only after thermal averaging do we solve the voltage-probe problem.
    for i,Tmat in enumerate(T_used):
        if not np.all(np.isfinite(Tmat)):
            continue
        try:
            r=resistances(Tmat)
            for k in keys: out[k][i]=r[k]
        except Exception as exc:
            print(f"warning solving LB equations at i={i}: {exc}")

    return {"energy_meV":energies, "nu":fillings,
            "T":T0, "T_thermal":T_used, "T_K":float(T_K), **out}

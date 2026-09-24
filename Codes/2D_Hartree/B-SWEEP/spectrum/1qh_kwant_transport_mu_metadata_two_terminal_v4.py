"""
Flexible Kwant transport scaffold for magnetic-field sweeps of Hartree-generated
integer-QH effective potentials.

Design principles
-----------------
1. The Hartree grid is the Kwant grid: no resampling/down-grading of lengths.
2. Energies are physical meV. The hopping is t = hbar^2 / (2 m* a^2).
3. The magnetic field enters through Kwant's gauge-fixed Peierls machinery,
   which is safer than hand-written gauge phases for multi-terminal devices.
4. The potential can be smoothly reshaped under rectangular lead mouths to
   reduce artificial entrance barriers.
5. The multi-terminal conductance matrix is converted to four-terminal
   resistances by solving the Landauer-Buettiker current-voltage equations.

The module is written so that importing it does not require kwant. Kwant is only
required when calling build/finalize/smatrix routines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union
import json
import math

import numpy as np

# ---------- physical constants ----------
HBAR2_OVER_2ME_NM2_MEV = 38.0998212  # hbar^2/(2 m_e) in meV nm^2
PHI0_T_NM2 = 4135.667696             # h/e in T nm^2
HBAR_OMEGA_C_MEV_PER_T_MSTAR1 = 0.11576764  # hbar e B / m_e in meV/T
MU_B_MEV_PER_T = 0.05788381806


def lB_nm(B_T: float) -> float:
    """Magnetic length in nm."""
    return 25.658206 / math.sqrt(float(B_T))


def alpha_flux_per_plaquette(B_T: float, a_nm: float) -> float:
    """Dimensionless flux alpha = B a^2 / Phi0 = a^2/(2 pi lB^2)."""
    return float(B_T) * float(a_nm) ** 2 / PHI0_T_NM2


def hopping_meV(a_nm: float, m_eff: float = 0.067) -> float:
    """Nearest-neighbour finite-difference hopping for GaAs-like effective mass."""
    return HBAR2_OVER_2ME_NM2_MEV / (float(m_eff) * float(a_nm) ** 2)


def cyclotron_meV(B_T: float, m_eff: float = 0.067) -> float:
    return HBAR_OMEGA_C_MEV_PER_T_MSTAR1 * float(B_T) / float(m_eff)


def zeeman_meV(B_T: float, g_factor: float = -0.44) -> float:
    return g_factor * MU_B_MEV_PER_T * float(B_T)


# ---------- data loading ----------
@dataclass(frozen=True)
class PotentialFrame:
    B_T: float
    U_meV: np.ndarray
    metadata: dict
    run_dir: Optional[Path] = None


@dataclass(frozen=True)
class PotentialSequence:
    B_values_T: np.ndarray
    U_stack_meV: np.ndarray  # shape (nB, ny, nx)
    x_nm: np.ndarray
    y_nm: np.ndarray
    metadata: List[dict]
    run_dirs: List[Optional[Path]] = field(default_factory=list)
    potential_key: str = "U"

    @property
    def nx(self) -> int: return int(self.x_nm.size)
    @property
    def ny(self) -> int: return int(self.y_nm.size)
    @property
    def dx_nm(self) -> float: return float(np.mean(np.diff(self.x_nm)))
    @property
    def dy_nm(self) -> float: return float(np.mean(np.diff(self.y_nm)))
    @property
    def a_nm(self) -> float:
        if not np.allclose(np.diff(self.x_nm), self.dx_nm, rtol=1e-4, atol=1e-9):
            raise ValueError("x grid is not uniform enough for a square Kwant lattice")
        if not np.allclose(np.diff(self.y_nm), self.dy_nm, rtol=1e-4, atol=1e-9):
            raise ValueError("y grid is not uniform enough for a square Kwant lattice")
        if not np.isclose(abs(self.dx_nm), abs(self.dy_nm), rtol=2e-3, atol=1e-9):
            raise ValueError(f"Need dx ~= dy. Got dx={self.dx_nm} nm, dy={self.dy_nm} nm")
        return abs(self.dx_nm)

    def frame(self, index: int) -> PotentialFrame:
        return PotentialFrame(
            B_T=float(self.B_values_T[index]),
            U_meV=np.asarray(self.U_stack_meV[index]),
            metadata=self.metadata[index],
            run_dir=self.run_dirs[index] if self.run_dirs else None,
        )


def find_simulation_dirs(sweep_dir: Union[str, Path]) -> List[Path]:
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Cannot find sweep directory {sweep_dir}")
    out = [p for p in sweep_dir.rglob("*") if (p / "data.npz").exists() and (p / "metadata.json").exists()]
    if not out:
        raise FileNotFoundError(f"No folders with data.npz and metadata.json below {sweep_dir}")
    return sorted(out)


def read_B_from_metadata(run_dir: Union[str, Path]) -> Tuple[float, dict]:
    run_dir = Path(run_dir)
    with open(run_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    try:
        B = float(meta["params"]["B"])
    except KeyError as exc:
        raise KeyError(f"Missing metadata['params']['B'] in {run_dir / 'metadata.json'}") from exc
    return B, meta


def load_effective_potential_sequence(
    sweep_dir_or_run_dirs: Union[str, Path, Sequence[Union[str, Path]]],
    potential_key: str = "U",
    ground_each_frame: bool = False,
    dtype=np.float64,
) -> PotentialSequence:
    """Load the B-sorted 2D potential sequence from the Hartree sweep format.

    The expected format is the one used by load_effective_potential_video.ipynb:
    each run folder contains metadata.json and data.npz, with arrays U, x, y.
    """
    if isinstance(sweep_dir_or_run_dirs, (str, Path)):
        run_dirs = find_simulation_dirs(sweep_dir_or_run_dirs)
    else:
        run_dirs = [Path(p) for p in sweep_dir_or_run_dirs]

    records = []
    x_ref = y_ref = None
    shape_ref = None
    for run_dir in run_dirs:
        B, meta = read_B_from_metadata(run_dir)
        with np.load(run_dir / "data.npz", allow_pickle=False) as data:
            if potential_key not in data.files:
                raise KeyError(f"{potential_key!r} not in {run_dir/'data.npz'}; available={data.files}")
            U = np.asarray(data[potential_key], dtype=dtype)
            if U.ndim != 2:
                raise ValueError(f"Potential {potential_key!r} must be 2D, got shape {U.shape}")
            if ground_each_frame:
                U = U - np.nanmin(U)
            x = np.asarray(data["x"], dtype=dtype) if "x" in data.files else np.arange(U.shape[1], dtype=dtype)
            y = np.asarray(data["y"], dtype=dtype) if "y" in data.files else np.arange(U.shape[0], dtype=dtype)
        if shape_ref is None:
            shape_ref = U.shape
            x_ref, y_ref = x, y
        else:
            if U.shape != shape_ref:
                raise ValueError(f"Grid shape mismatch in {run_dir}: {U.shape} != {shape_ref}")
            if not (np.allclose(x, x_ref) and np.allclose(y, y_ref)):
                raise ValueError(f"x/y grid mismatch in {run_dir}")
        records.append((B, run_dir, U, meta))

    records.sort(key=lambda r: r[0])
    return PotentialSequence(
        B_values_T=np.asarray([r[0] for r in records], dtype=dtype),
        U_stack_meV=np.stack([r[2] for r in records], axis=0),
        x_nm=x_ref,
        y_nm=y_ref,
        metadata=[r[3] for r in records],
        run_dirs=[r[1] for r in records],
        potential_key=potential_key,
    )


def load_cached_sequence(cache_file: Union[str, Path], potential_key: str = "U") -> PotentialSequence:
    """Load a lightweight cache produced by the video loader notebook."""
    cache_file = Path(cache_file)
    with np.load(cache_file, allow_pickle=False) as z:
        B = np.asarray(z["B_values"], dtype=float)
        U = np.asarray(z["U_stack"], dtype=float)
        x = np.asarray(z["x"], dtype=float)
        y = np.asarray(z["y"], dtype=float)
    return PotentialSequence(B, U, x, y, metadata=[{} for _ in B], run_dirs=[None for _ in B], potential_key=potential_key)




# ---------- Hartree metadata energy selection ----------
def _flatten_metadata(meta: dict, prefix: str = "") -> Dict[str, object]:
    """Return a flattened view of nested metadata using dot-separated keys."""
    out: Dict[str, object] = {}
    if not isinstance(meta, dict):
        return out
    for key, val in meta.items():
        k = f"{prefix}.{key}" if prefix else str(key)
        out[k] = val
        if isinstance(val, dict):
            out.update(_flatten_metadata(val, k))
    return out


def _as_float_or_none(value) -> Optional[float]:
    """Convert scalar metadata values to float when possible."""
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
    except Exception:
        pass
    return None


def chemical_potential_from_metadata(
    metadata: dict,
    key_candidates: Optional[Sequence[str]] = None,
    unit: Literal["meV", "eV", "J", "auto"] = "auto",
    strict: bool = True,
) -> float:
    """Extract the Hartree self-consistent chemical potential from metadata.

    The loader keeps the full ``metadata.json`` for every B-frame. This helper
    recursively searches common key names and returns the value in meV, suitable
    to be passed as Kwant's scattering energy.

    Parameters
    ----------
    metadata:
        One frame metadata dictionary, usually ``seq.frame(i).metadata``.
    key_candidates:
        Optional ordered list of exact flattened metadata keys or leaf-key names.
        Examples: ``"mu"``, ``"mu_meV"``, ``"params.mu"``.
    unit:
        Unit of the stored value. ``"auto"`` treats keys containing ``meV`` as
        meV, keys containing ``eV`` as eV, keys containing ``J``/``joule`` as J,
        and otherwise assumes meV because the transport module uses meV.
    strict:
        If True, raise a helpful KeyError when no candidate is found.
    """
    flat = _flatten_metadata(metadata)

    default_candidates = [
        "mu_meV", "chemical_potential_meV", "fermi_energy_meV", "EF_meV", "E_F_meV",
        "mu", "chemical_potential", "fermi_energy", "EF", "E_F",
        "params.mu_meV", "params.chemical_potential_meV", "params.fermi_energy_meV",
        "params.mu", "params.chemical_potential", "params.fermi_energy",
        "results.mu_meV", "results.chemical_potential_meV",
        "self_consistency.mu_meV", "self_consistency.chemical_potential_meV",
    ]
    candidates = list(key_candidates) if key_candidates is not None else default_candidates

    # First pass: exact flattened-key matches.
    for cand in candidates:
        if cand in flat:
            val = _as_float_or_none(flat[cand])
            if val is not None:
                key = cand
                break
    else:
        # Second pass: leaf-name matches, e.g. cand='mu' matches 'params.mu'.
        val = None
        key = None
        leaf_candidates = set(candidates)
        for k, v in flat.items():
            leaf = k.split(".")[-1]
            if leaf in leaf_candidates:
                fv = _as_float_or_none(v)
                if fv is not None:
                    key, val = k, fv
                    break
        # Third pass: conservative fuzzy search for metadata names containing
        # chemical potential / Fermi energy.
        if val is None:
            for k, v in flat.items():
                kl = k.lower()
                looks_like_mu = (
                    "chemical" in kl and "potential" in kl
                ) or "fermi" in kl or kl.endswith(".mu") or kl == "mu"
                if looks_like_mu:
                    fv = _as_float_or_none(v)
                    if fv is not None:
                        key, val = k, fv
                        break
        if val is None:
            if strict:
                available = ", ".join(sorted(flat.keys())[:80])
                raise KeyError(
                    "Could not find the Hartree chemical potential in metadata. "
                    "Pass key_candidates=[...] to chemical_potential_from_metadata, "
                    "or inspect seq.frame(i).metadata. Available flattened keys start with: "
                    + available
                )
            return float("nan")

    key_lower = str(key).lower()
    chosen_unit = unit
    if unit == "auto":
        if "mev" in key_lower:
            chosen_unit = "meV"
        elif "ev" in key_lower and "mev" not in key_lower:
            chosen_unit = "eV"
        elif "joule" in key_lower or key_lower.endswith("_j") or key_lower.endswith(".j"):
            chosen_unit = "J"
        else:
            chosen_unit = "meV"

    if chosen_unit == "meV":
        return float(val)
    if chosen_unit == "eV":
        return 1e3 * float(val)
    if chosen_unit == "J":
        return float(val) / 1.602176634e-22
    raise ValueError(f"Unknown unit={unit!r}")


def hartree_energy_selector(
    key_candidates: Optional[Sequence[str]] = None,
    unit: Literal["meV", "eV", "J", "auto"] = "auto",
) -> Callable[[PotentialFrame], float]:
    """Return an energy_selector(frame) that reads mu from frame.metadata."""
    def selector(frame: PotentialFrame) -> float:
        return chemical_potential_from_metadata(frame.metadata, key_candidates=key_candidates, unit=unit, strict=True)
    return selector


# ---------- potential preparation ----------
@dataclass(frozen=True)
class DisorderConfig:
    amplitude_meV: float = 0.0
    correlation_nm: float = 0.0
    seed: int = 12345


def correlated_disorder(shape: Tuple[int, int], a_nm: float, cfg: DisorderConfig) -> np.ndarray:
    """Gaussian-correlated zero-mean disorder in meV, FFT implementation."""
    ny, nx = shape
    if cfg.amplitude_meV == 0:
        return np.zeros(shape, dtype=float)
    rng = np.random.default_rng(cfg.seed)
    white = rng.normal(size=(ny, nx))
    if cfg.correlation_nm <= 0:
        dis = white
    else:
        kk = np.fft.fft2(white)
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=a_nm)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=a_nm)
        KX, KY = np.meshgrid(kx, ky)
        filt = np.exp(-0.5 * cfg.correlation_nm ** 2 * (KX ** 2 + KY ** 2))
        dis = np.fft.ifft2(kk * filt).real
    dis -= dis.mean()
    std = dis.std()
    return cfg.amplitude_meV * dis / std if std > 0 else dis


LeadSide = Literal["left", "right", "bottom", "top"]


@dataclass(frozen=True)
class LeadSpec:
    name: str
    side: LeadSide
    center_nm: Optional[float] = None       # along the boundary: y for left/right, x for top/bottom
    width_nm: Optional[float] = None        # mouth width along boundary; None means full side
    depth_nm: float = 150.0                 # how far the mouth extends into the sample
    lead_potential_meV: float = 0.0         # target flat potential in the lead/mouth
    smooth_nm: float = 50.0                 # Gaussian sigma for lead-mouth taper, in nm
    smooth_profile: Literal[
        "gaussian_boundary",
        "constant_inner",
        "gaussian_inner",
        "gaussian",
        "smoothstep",
    ] = "gaussian_boundary"
    # "gaussian_boundary" replaces the full lead-mouth region by an explicit
    # boundary-value interpolation:
    #     U(contact edge) = lead_potential_meV,
    #     U(inner mouth at depth_nm) = original U sampled at depth_nm.
    # No pointwise multiplication of the violent original edge potential is used.
    #
    # "constant_inner" replaces the whole lead mouth by the original potential
    # sampled at the inner mouth boundary, i.e. U(x_edge...x_edge+depth)=U(depth).
    # This is useful as a diagnostic/robust option when any ramp still creates
    # an unwanted barrier.
    #
    # "gaussian_inner" is kept as a backward-compatible alias of
    # "gaussian_boundary". "gaussian" keeps the previous pointwise Gaussian
    # interpolation, and "smoothstep" restores the old polynomial behavior.


@dataclass(frozen=True)
class DeviceConfig:
    m_eff: float = 0.067
    g_factor: float = -0.44
    norbs: int = 1                          # 1 = spinless, 2 = spinful Zeeman split
    disorder: DisorderConfig = DisorderConfig()
    lead_specs: Tuple[LeadSpec, ...] = (LeadSpec("L", "left"), LeadSpec("R", "right"),)
    reshape_under_leads: bool = True
    lead_B_mode: Literal["same", "zero"] = "same"  # passed to Kwant magnetic_gauge


def _smoothstep(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, 0.0, 1.0)
    return z * z * (3 - 2 * z)


def _normalized_gaussian_taper(dist_nm: np.ndarray, depth_nm: float, sigma_nm: float) -> np.ndarray:
    """Weight for pointwise interpolation to the lead value.

    This legacy taper is 1 at the contacted edge and 0 at the inner mouth.
    It is used only by ``smooth_profile='gaussian'``.
    """
    depth = max(float(depth_nm), 1e-12)
    sigma = max(float(sigma_nm), 1e-12)
    g = np.exp(-0.5 * (dist_nm / sigma) ** 2)
    g_depth = math.exp(-0.5 * (depth / sigma) ** 2)
    denom = max(1.0 - g_depth, 1e-15)
    return np.clip((g - g_depth) / denom, 0.0, 1.0)


def _normalized_gaussian_boundary_ramp(dist_nm: np.ndarray, depth_nm: float, sigma_nm: float) -> np.ndarray:
    """Ramp from 0 at the contact edge to 1 at the inner mouth boundary.

    The reshaped potential for the robust lead-mouth mode is

        U_new = U_lead + ramp * (U_inner_boundary - U_lead).

    Thus the original violent edge potential inside the mouth is completely
    discarded. The Gaussian is centered at the inner boundary, so the potential
    approaches the sampled inner value smoothly as ``dist_nm -> depth_nm``.
    """
    depth = max(float(depth_nm), 1e-12)
    sigma = max(float(sigma_nm), 1e-12)
    dist = np.clip(dist_nm, 0.0, depth)
    g = np.exp(-0.5 * ((depth - dist) / sigma) ** 2)
    g_edge = math.exp(-0.5 * (depth / sigma) ** 2)
    denom = max(1.0 - g_edge, 1e-15)
    return np.clip((g - g_edge) / denom, 0.0, 1.0)


def lead_mask_and_weight(x_nm: np.ndarray, y_nm: np.ndarray, spec: LeadSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Return rectangular lead-mouth mask and interpolation weight.

    The weight multiplies the lead target potential in

        U_new = (1 - weight) * U_old + weight * U_lead.

    Therefore weight=1 means a fully flattened lead/contact value, while
    weight=0 means the original Hartree potential is untouched. By default the
    decay is a normalized Gaussian: weight=1 at the contacted edge and weight=0
    at ``depth_nm``.
    """
    X, Y = np.meshgrid(x_nm, y_nm)
    xmin, xmax = float(x_nm.min()), float(x_nm.max())
    ymin, ymax = float(y_nm.min()), float(y_nm.max())
    along = Y if spec.side in ("left", "right") else X
    center = float(spec.center_nm) if spec.center_nm is not None else float(0.5 * (along.min() + along.max()))
    width = float(spec.width_nm) if spec.width_nm is not None else float(along.max() - along.min() + max(abs(np.diff(x_nm).mean()), abs(np.diff(y_nm).mean())))
    in_width = np.abs(along - center) <= 0.5 * width

    if spec.side == "left":
        dist = X - xmin
    elif spec.side == "right":
        dist = xmax - X
    elif spec.side == "bottom":
        dist = Y - ymin
    elif spec.side == "top":
        dist = ymax - Y
    else:
        raise ValueError(spec.side)

    depth = max(float(spec.depth_nm), 1e-12)
    in_depth = (dist >= -1e-9) & (dist <= depth)
    mask = in_width & in_depth

    if spec.smooth_profile in ("gaussian_boundary", "gaussian_inner", "constant_inner"):
        # For robust non-pointwise modes, the returned weight is the boundary
        # ramp: 0 at contact, 1 at inner mouth. For constant_inner the weight is
        # not used by reshape_potential_under_leads, but returning the mask keeps
        # this helper useful for diagnostics.
        edge_weight = _normalized_gaussian_boundary_ramp(dist, depth_nm=depth, sigma_nm=float(spec.smooth_nm))
    elif spec.smooth_profile == "gaussian":
        edge_weight = _normalized_gaussian_taper(dist, depth_nm=depth, sigma_nm=float(spec.smooth_nm))
    elif spec.smooth_profile == "smoothstep":
        # Old polynomial behavior, retained for direct comparison.
        edge_weight = 1.0 - _smoothstep(dist / depth)
    else:
        raise ValueError(f"Unknown lead smooth_profile={spec.smooth_profile!r}")

    return mask, mask * edge_weight


def _sample_inner_mouth_potential(U: np.ndarray, x_nm: np.ndarray, y_nm: np.ndarray, spec: LeadSpec) -> np.ndarray:
    """Return U sampled at the inner boundary of a lead mouth, broadcast as U.shape.

    For left/right leads, the sampled value depends on y and is taken at the
    column closest to xmin+depth or xmax-depth. For top/bottom leads, it depends
    on x and is taken at the row closest to ymin+depth or ymax-depth.
    """
    xmin, xmax = float(x_nm.min()), float(x_nm.max())
    ymin, ymax = float(y_nm.min()), float(y_nm.max())
    depth = max(float(spec.depth_nm), 0.0)

    if spec.side == "left":
        ix = int(np.argmin(np.abs(x_nm - (xmin + depth))))
        return np.broadcast_to(U[:, ix][:, None], U.shape)
    if spec.side == "right":
        ix = int(np.argmin(np.abs(x_nm - (xmax - depth))))
        return np.broadcast_to(U[:, ix][:, None], U.shape)
    if spec.side == "bottom":
        iy = int(np.argmin(np.abs(y_nm - (ymin + depth))))
        return np.broadcast_to(U[iy, :][None, :], U.shape)
    if spec.side == "top":
        iy = int(np.argmin(np.abs(y_nm - (ymax - depth))))
        return np.broadcast_to(U[iy, :][None, :], U.shape)
    raise ValueError(spec.side)


def reshape_potential_under_leads(U_meV: np.ndarray, x_nm: np.ndarray, y_nm: np.ndarray, lead_specs: Sequence[LeadSpec]) -> np.ndarray:
    """Smoothly lower/flatten U in rectangular lead mouths to improve impedance matching.

    For the default ``smooth_profile='gaussian_boundary'``, the reshaped mouth is
    not obtained by multiplying the original edge potential point by point.
    Instead it is built as a Gaussian ramp from the lead value at the contact to
    the original potential sampled at the inner mouth boundary. This is more
    robust when the confinement rises sharply at the physical edge.
    """
    U_original = np.array(U_meV, dtype=float, copy=True)
    U = U_original.copy()
    for spec in lead_specs:
        mask, w = lead_mask_and_weight(x_nm, y_nm, spec)
        target = float(spec.lead_potential_meV)
        if spec.smooth_profile in ("gaussian_boundary", "gaussian_inner"):
            U_inner = _sample_inner_mouth_potential(U_original, x_nm, y_nm, spec)
            # w=0 at contact -> target. w=1 at inner mouth -> U_inner.
            # The original U inside the mouth is deliberately not used.
            U[mask] = target + w[mask] * (U_inner[mask] - target)
        elif spec.smooth_profile == "constant_inner":
            U_inner = _sample_inner_mouth_potential(U_original, x_nm, y_nm, spec)
            # Flat mouth: U(edge...inner boundary) = U(inner boundary).
            U[mask] = U_inner[mask]
        else:
            # Backward-compatible pointwise interpolation. Here w=1 at contact
            # and w=0 at the inner mouth.
            U[mask] = (1.0 - w[mask]) * U[mask] + w[mask] * target
    return U


# ---------- Kwant builder ----------
@dataclass
class PreparedFrame:
    B_T: float
    x_nm: np.ndarray
    y_nm: np.ndarray
    U_meV: np.ndarray
    U_for_transport_meV: np.ndarray
    t_meV: float
    a_nm: float
    alpha: float
    lB_nm: float
    omega_c_meV: float
    zeeman_meV: float


def prepare_frame(seq: PotentialSequence, index: int, cfg: DeviceConfig) -> PreparedFrame:
    frame = seq.frame(index)
    a = seq.a_nm
    dis = correlated_disorder(frame.U_meV.shape, a, cfg.disorder)
    U = np.asarray(frame.U_meV, dtype=float) + dis
    Utr = reshape_potential_under_leads(U, seq.x_nm, seq.y_nm, cfg.lead_specs) if cfg.reshape_under_leads else U.copy()
    B = frame.B_T
    return PreparedFrame(
        B_T=B, x_nm=seq.x_nm, y_nm=seq.y_nm, U_meV=U, U_for_transport_meV=Utr,
        t_meV=hopping_meV(a, cfg.m_eff), a_nm=a, alpha=alpha_flux_per_plaquette(B, a),
        lB_nm=lB_nm(B), omega_c_meV=cyclotron_meV(B, cfg.m_eff), zeeman_meV=zeeman_meV(B, cfg.g_factor),
    )


def _sigma_matrices(norbs: int, zeeman: float):
    if norbs == 1:
        return np.array([[1.0]]), np.array([[0.0]])
    if norbs == 2:
        s0 = np.eye(2, dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        return s0, 0.5 * zeeman * sz
    raise ValueError("norbs must be 1 or 2")


def build_kwant_builder(prep: PreparedFrame, cfg: DeviceConfig):
    """Build an unf inalized Kwant Builder with rectangular leads.

    Requires kwant installed. Uses add_peierls_phase later, so hoppings here are
    real; the magnetic phase is supplied by Kwant's gauge fixer.
    """
    try:
        import kwant
    except ImportError as exc:
        raise ImportError("This function requires kwant. Install it in the environment where you run transport.") from exc

    nx, ny = prep.x_nm.size, prep.y_nm.size
    lat = kwant.lattice.square(a=prep.a_nm, norbs=cfg.norbs)
    s0, hz = _sigma_matrices(cfg.norbs, prep.zeeman_meV)
    t = prep.t_meV
    U = prep.U_for_transport_meV

    def onsite(site):
        ix, iy = site.tag
        return (4 * t + float(U[iy, ix])) * s0 + hz

    def hop(site1, site2):
        return -t * s0

    syst = kwant.Builder()
    syst[(lat(ix, iy) for ix in range(nx) for iy in range(ny))] = onsite
    syst[lat.neighbors()] = hop

    def add_rect_lead(spec: LeadSpec):
        # Determine the sites on the boundary covered by the mouth.
        if spec.side in ("left", "right"):
            coord = prep.y_nm
            center = spec.center_nm if spec.center_nm is not None else 0.5 * (coord.min() + coord.max())
            width = spec.width_nm if spec.width_nm is not None else coord.max() - coord.min() + prep.a_nm
            inds = [i for i, yy in enumerate(coord) if abs(yy - center) <= 0.5 * width]
            if not inds: raise ValueError(f"Lead {spec.name} has no sites")
            direction = (-1, 0) if spec.side == "left" else (1, 0)
            x0 = 0 if spec.side == "left" else nx - 1
            sym = kwant.TranslationalSymmetry((direction[0] * prep.a_nm, 0))
            lead = kwant.Builder(sym)
            V_lead = float(spec.lead_potential_meV)

            def lead_onsite(site):
                return (4 * t + V_lead) * s0 + hz

            lead[(lat(x0, iy) for iy in inds)] = lead_onsite
            lead[lat.neighbors()] = hop
        else:
            coord = prep.x_nm
            center = spec.center_nm if spec.center_nm is not None else 0.5 * (coord.min() + coord.max())
            width = spec.width_nm if spec.width_nm is not None else coord.max() - coord.min() + prep.a_nm
            inds = [i for i, xx in enumerate(coord) if abs(xx - center) <= 0.5 * width]
            if not inds: raise ValueError(f"Lead {spec.name} has no sites")
            direction = (0, -1) if spec.side == "bottom" else (0, 1)
            y0 = 0 if spec.side == "bottom" else ny - 1
            sym = kwant.TranslationalSymmetry((0, direction[1] * prep.a_nm))
            lead = kwant.Builder(sym)
            V_lead = float(spec.lead_potential_meV)

            def lead_onsite(site):
                return (4 * t + V_lead) * s0 + hz

            lead[(lat(ix, y0) for ix in inds)] = lead_onsite
            lead[lat.neighbors()] = hop
        syst.attach_lead(lead)

    for spec in cfg.lead_specs:
        add_rect_lead(spec)
    return syst


def finalize_with_magnetic_field(builder, B_T: float, nleads: int, lead_B_mode: str = "same"):
    """Finalize builder and attach Kwant gauge-fixed Peierls phases.

    Kwant's add_peierls_phase returns (fsyst, gauge). The smatrix params must
    include **gauge(B_syst, B_lead0, B_lead1, ...). Here B is in units of flux
    quantum per nm^2 because the lattice coordinates are in nm.
    """
    import kwant
    fsyst, gauge = kwant.builder.add_peierls_phase(builder)
    B_flux_per_nm2 = float(B_T) / PHI0_T_NM2
    lead_field = B_flux_per_nm2 if lead_B_mode == "same" else 0.0
    phase_params = gauge(B_flux_per_nm2, *([lead_field] * nleads))
    return fsyst, phase_params


# ---------- transport and multi-terminal resistance ----------
def transmission_matrix(smatrix) -> np.ndarray:
    """T[i,j] = transmission from lead j into lead i."""
    n = len(smatrix.lead_info)
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                T[i, j] = smatrix.transmission(i, j)
    return T


def conductance_matrix_e2h(T: np.ndarray) -> np.ndarray:
    """Dimensionless Landauer-Buettiker matrix I_i=(e^2/h) sum_j G_ij V_j."""
    n = T.shape[0]
    G = np.zeros_like(T, dtype=float)
    for i in range(n):
        G[i, i] = np.sum(T[:, i])  # total leaving reservoir i into all other leads
        for j in range(n):
            if i != j:
                G[i, j] = -T[i, j]
    return G


def solve_voltages_dimensionless(G: np.ndarray, currents_e2h_times_V: np.ndarray, ground: int = 0) -> np.ndarray:
    """Solve G V = I with one lead grounded. Units are dimensionless voltages."""
    n = G.shape[0]
    keep = [i for i in range(n) if i != ground]
    V = np.zeros(n, dtype=float)
    V[keep] = np.linalg.solve(G[np.ix_(keep, keep)], currents_e2h_times_V[keep])
    return V


def four_terminal_resistance_h_over_e2(T: np.ndarray, source: int, drain: int, vplus: int, vminus: int, current: float = 1.0, ground: Optional[int] = None) -> float:
    """Return R=(V_vplus - V_vminus)/I in units h/e^2.

    Current is injected at source and removed at drain. Voltage probes have zero current.
    """
    G = conductance_matrix_e2h(T)
    I = np.zeros(T.shape[0], dtype=float)
    I[source] = current
    I[drain] = -current
    if ground is None:
        ground = drain
    V = solve_voltages_dimensionless(G, I, ground=ground)
    return float((V[vplus] - V[vminus]) / current)



# ---------- two-terminal convenience helpers ----------
def two_terminal_conductance_e2h(T: np.ndarray, source: int = 0, drain: int = 1) -> float:
    """Return the two-terminal Landauer conductance in units e^2/h.

    With the standard ordering used here, lead 0 is source and lead 1 is drain.
    The returned quantity is T[drain, source], i.e. transmission from source
    into drain.
    """
    return float(T[drain, source])


def two_terminal_resistance_h_e2(T: np.ndarray, source: int = 0, drain: int = 1) -> float:
    """Return the ideal two-terminal resistance 1/G in units h/e^2.

    This is only the inverse Landauer conductance. It is useful for plotting,
    but it is not a four-terminal resistance.
    """
    G = two_terminal_conductance_e2h(T, source=source, drain=drain)
    return float(np.inf if G == 0 else 1.0 / G)

def run_one_frame(
    seq: PotentialSequence,
    index: int,
    cfg: DeviceConfig,
    energy_meV: Optional[float] = None,
    energy_selector: Optional[Callable[[PotentialFrame], float]] = None,
):
    """Build, finalize, and compute the scattering matrix for one B frame.

    If ``energy_meV`` is not supplied, the default is now to read the
    self-consistent Hartree chemical potential from ``metadata.json`` for this
    frame. This is the physically correct Fermi energy for the Kwant scattering
    calculation, provided the Hartree potential and chemical potential use the
    same energy zero.
    """
    import kwant
    frame = seq.frame(index)
    if energy_meV is None:
        if energy_selector is None:
            E = chemical_potential_from_metadata(frame.metadata)
        else:
            E = float(energy_selector(frame))
    else:
        E = float(energy_meV)
    prep = prepare_frame(seq, index, cfg)
    builder = build_kwant_builder(prep, cfg)
    fsyst, phase_params = finalize_with_magnetic_field(builder, prep.B_T, len(cfg.lead_specs), cfg.lead_B_mode)
    smat = kwant.smatrix(fsyst, energy=E, params=phase_params)
    return prep, smat, transmission_matrix(smat)


def run_B_sweep(seq: PotentialSequence, cfg: DeviceConfig, energy_selector: Optional[Callable[[PotentialFrame], float]] = None, max_frames: Optional[int] = None) -> List[dict]:
    """Run the Kwant sweep over B. Returns a list of dictionaries per frame.

    By default the scattering energy is the chemical potential stored in each
    frame's Hartree metadata. Pass ``energy_selector`` only to override the
    metadata key/unit logic.
    """
    results = []
    nframes = len(seq.B_values_T) if max_frames is None else min(max_frames, len(seq.B_values_T))
    for i in range(nframes):
        frame = seq.frame(i)
        energy = chemical_potential_from_metadata(frame.metadata) if energy_selector is None else float(energy_selector(frame))
        prep, smat, T = run_one_frame(seq, i, cfg, energy_meV=energy)
        results.append({"index": i, "B_T": prep.B_T, "energy_meV": energy, "T": T})
    return results


# ---------- useful preset lead geometries ----------
def two_terminal_leads(depth_nm=150.0, lead_potential_meV=0.0) -> Tuple[LeadSpec, ...]:
    return (
        LeadSpec("source", "left", depth_nm=depth_nm, lead_potential_meV=lead_potential_meV),
        LeadSpec("drain", "right", depth_nm=depth_nm, lead_potential_meV=lead_potential_meV),
    )


def six_terminal_hall_bar(y_probe_nm: float, probe_width_nm: float, depth_nm=150.0, lead_potential_meV=0.0) -> Tuple[LeadSpec, ...]:
    """A common ordering: 0 left, 1 right, 2 bottom-left, 3 top-left, 4 bottom-right, 5 top-right."""
    return (
        LeadSpec("source", "left", depth_nm=depth_nm, lead_potential_meV=lead_potential_meV),
        LeadSpec("drain", "right", depth_nm=depth_nm, lead_potential_meV=lead_potential_meV),
        LeadSpec("VBL", "bottom", center_nm=-y_probe_nm, width_nm=probe_width_nm, depth_nm=depth_nm, lead_potential_meV=lead_potential_meV),
        LeadSpec("VTL", "top", center_nm=-y_probe_nm, width_nm=probe_width_nm, depth_nm=depth_nm, lead_potential_meV=lead_potential_meV),
        LeadSpec("VBR", "bottom", center_nm=+y_probe_nm, width_nm=probe_width_nm, depth_nm=depth_nm, lead_potential_meV=lead_potential_meV),
        LeadSpec("VTR", "top", center_nm=+y_probe_nm, width_nm=probe_width_nm, depth_nm=depth_nm, lead_potential_meV=lead_potential_meV),
    )

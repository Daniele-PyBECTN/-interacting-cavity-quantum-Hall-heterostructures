from pathlib import Path
from dataclasses import replace, fields, asdict
import json

import numpy as np
from datetime import datetime


def json_safe_value(v):
    """Convert common NumPy/Path values to JSON-safe Python values."""
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


def params_to_json_dict(p):
    """Convert a dataclass parameter object to a JSON-safe dictionary."""
    return {
        k: json_safe_value(v)
        for k, v in asdict(p).items()
    }


def latest_saved_simulation_dir(results_dir):
    """
    Return the folder of the latest saved simulation,
    or None if no saved simulation is found.
    """
    results_dir = Path(results_dir)

    latest_file = results_dir / "latest_simulation.json"

    if latest_file.exists():
        with open(latest_file, "r") as f:
            info = json.load(f)

        run_dir = Path(info["run_dir"])

        if not run_dir.is_absolute():
            run_dir = results_dir / run_dir

        if run_dir.exists():
            return run_dir

    if not results_dir.exists():
        return None

    candidates = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "data.npz").exists()
    )

    return candidates[-1] if candidates else None


def load_saved_simulation(run_dir):
    """Load metadata and numerical arrays from a saved simulation directory."""
    run_dir = Path(run_dir)

    with open(run_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    data = np.load(run_dir / "data.npz", allow_pickle=False)

    return metadata, data


def apply_parameter_overrides(p, overrides):
    """Return p with the requested parameter overrides applied."""
    if not overrides:
        return p

    valid_fields = {f.name for f in fields(p)}
    bad = sorted(set(overrides) - valid_fields)

    if bad:
        raise ValueError(
            f"Unknown parameter fields in PARAMETER_OVERRIDES: {bad}"
        )

    return replace(p, **overrides)


def prepare_simulation(
    p,
    results_dir,
    parameter_overrides=None,
    use_latest_saved_simulation=False,
    use_saved_U_as_initial_guess=True,
    verbose=True,
):
    """
    Prepare a simulation.

    Operations:
      1. optionally load parameters from latest saved simulation,
      2. apply user parameter overrides,
      3. optionally recover saved U as initial guess.

    Returns
    -------
    p
        Updated parameter dataclass.

    U_initial
        Saved converged potential if compatible, otherwise None.

    resumed_run_dir
        Directory from which the simulation was resumed, or None.

    resumed_metadata
        Metadata from previous simulation, or None.

    resumed_data
        Loaded npz object, or None.
    """

    results_dir = Path(results_dir)

    resumed_run_dir = None
    resumed_metadata = None
    resumed_data = None
    U_initial = None

    # ---------------------------------------------------------
    # Load previous simulation
    # ---------------------------------------------------------
    if use_latest_saved_simulation:

        resumed_run_dir = latest_saved_simulation_dir(results_dir)

        if resumed_run_dir is None:
            if verbose:
                print(
                    f"No saved simulation found in {results_dir}. "
                    "Starting from the current parameters."
                )

        else:
            resumed_metadata, resumed_data = load_saved_simulation(
                resumed_run_dir
            )

            saved_params = resumed_metadata.get("params", {})

            valid_fields = {f.name for f in fields(p)}

            saved_params = {
                k: v
                for k, v in saved_params.items()
                if k in valid_fields
            }

            p = replace(p, **saved_params)

            if verbose:
                print(
                    f"Loaded parameters and data from: "
                    f"{resumed_run_dir}"
                )

    # ---------------------------------------------------------
    # User overrides
    # ---------------------------------------------------------
    p = apply_parameter_overrides(
        p,
        parameter_overrides,
    )

    if parameter_overrides and verbose:
        print(
            "Applied PARAMETER_OVERRIDES:",
            parameter_overrides,
        )

    # ---------------------------------------------------------
    # Saved potential as initial guess
    # ---------------------------------------------------------
    if (
        use_latest_saved_simulation
        and resumed_data is not None
        and use_saved_U_as_initial_guess
    ):

        saved_U = resumed_data["U"]

        if saved_U.shape == (p.Ny, p.Nx):

            U_initial = saved_U.copy()

            if verbose:
                print("Using saved U as initial guess.")

        else:

            if verbose:
                print(
                    "Saved U shape does not match the current grid; "
                    "starting from U_ext.\n"
                    f"saved_U.shape={saved_U.shape}, "
                    f"current={(p.Ny, p.Nx)}"
                )

    return (
        p,
        U_initial,
        resumed_run_dir,
        resumed_metadata,
        resumed_data,
    )


def print_simulation_info(p):
    """Print a compact summary of the active simulation parameters."""

    print("Active parameters:")
    print(p)

    print(
        f"dx, dy = "
        f"{p.Lx/(p.Nx-1):.3f} nm, "
        f"{p.Ly/(p.Ny-1):.3f} nm"
    )

    print(f"grid points = {p.Nx*p.Ny}")

    print(
        f"lB = {p.lB:.3f} nm, "
        f"hbar omega_c = {p.hbar_omega_c:.3f} meV, "
        f"Gamma = {p.Gamma:.4f} meV"
    )

    print(
        "Expected average filling = "
        f"{2*np.pi*p.lB**2*p.n_target:.4f}"
    )
    
def save_static_problem_2d(static, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "static_data.npz"

    np.savez_compressed(
        path,
        x=static["x"], y=static["y"],
        X=static["X"], Y=static["Y"],
        mask=static["mask"],
        n_donor_profile=static["nD"],
        top_gate_source_mask=static["top_gate_source_mask"],
        U_ext=static["U_ext"],
        K=static["K"],
        K_ee_direct=static["K_ee_direct"],
        K_ee_image=static["K_ee_image"],
        K_donor_direct=static["K_donor_direct"],
        K_donor_image=static["K_donor_image"],
    )
    return path

def save_simulation_2d(
    res,
    p,
    resumed_run_dir,
    parameter_overrides,
    results_dir,
    run_name=None,
    extra_metadata=None,
    static_data_path=None,
    save_static_arrays_each_run=True,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"sim_B_{p.B:.5g}_dist_{p.dist_fixed}_"
            f"xedge_{p.x_gate_edge_nm}_{stamp}"
        )

    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    data_path = run_dir / "data.npz"
    meta_path = run_dir / "metadata.json"

    if save_static_arrays_each_run:
        # Same keys as the supplied cell.
        arrays_to_save = {
            "x": res["x"],
            "y": res["y"],
            "X": res["X"],
            "Y": res["Y"],
            "mask": res["mask"],
            "n": res["n"],
            "nu": res["nu"],
            "U": res["U"],
            "U_ext": res["U_ext"],
            "compressibility": res["compressibility"],
            "n_donor_profile": res["n_donor_profile"],
            "top_gate_source_mask": res["top_gate_source_mask"],
            "K": res["K"],
            "K_ee_direct": res["K_ee_direct"],
            "K_ee_image": res["K_ee_image"],
            "K_donor_direct": res["K_donor_direct"],
            "K_donor_image": res["K_donor_image"],
            "history": res["history"],
        }
    else:
        arrays_to_save = {
            "n": res["n"],
            "nu": res["nu"],
            "U": res["U"],
            "compressibility": res["compressibility"],
            "history": res["history"],
        }

    np.savez_compressed(data_path, **arrays_to_save)

    summary = {
        "mu": json_safe_value(res["mu"]),
        "dist": json_safe_value(res["dist"]),
        "dx": json_safe_value(res["dx"]),
        "dy": json_safe_value(res["dy"]),
        "average_density": json_safe_value(np.mean(res["n"][res["mask"]])),
        "target_density": json_safe_value(p.n_target),
        "center_filling": json_safe_value(
            res["nu"][p.Ny//2, p.Nx//2]
        ),
        "nu_min": json_safe_value(np.nanmin(res["nu"])),
        "nu_max": json_safe_value(np.nanmax(res["nu"])),
        "converged_iterations_recorded": (
            int(res["history"][-1, 0]) if len(res["history"]) else None
        ),
        "final_dn": (
            json_safe_value(res["history"][-1, 3])
            if len(res["history"]) else None
        ),
    }

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "params": params_to_json_dict(p),
        "summary": summary,
        "resumed_from": (
            str(resumed_run_dir) if resumed_run_dir is not None else None
        ),
        "parameter_overrides": {
            k: json_safe_value(v)
            for k, v in parameter_overrides.items()
        },
    }

    if not save_static_arrays_each_run and static_data_path is not None:
        metadata["static_data"] = str(static_data_path)

    if extra_metadata is not None:
        metadata["extra_metadata"] = extra_metadata

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    latest_path = results_dir / "latest_simulation.json"
    with open(latest_path, "w") as f:
        json.dump(
            {
                "run_dir": run_dir.name,
                "created_at": metadata["created_at"],
            },
            f,
            indent=2,
        )

    print(f"Saved simulation to: {run_dir}")
    print(f"  data:     {data_path}")
    print(f"  metadata: {meta_path}")
    print(f"Updated latest pointer: {latest_path}")
    return run_dir
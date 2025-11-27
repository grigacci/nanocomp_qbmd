"""Run the Fortran QBMD program with convenient helpers.
Example (basic - returns only output directory):
    out_dir = run_qbmd(5, 2e-9, 7e-9, 2e-9, 1, 2e-9, 7e-9,
                       thickness_units="m", create_param_folder=True)
    print("Outputs in:", out_dir)

Example (with results - returns tuple of output directory and SimulationResult):
    out_dir, result = run_qbmd(5, 2e-9, 7e-9, 2e-9, 1, 2e-9, 7e-9,
                       thickness_units="m", create_param_folder=True,
                       return_results=True)
    print("Outputs in:", out_dir)
    print("Simulation result:", result)
"""

import subprocess
import os
import shutil
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from .extract import getSimulationResult
from .models import SimulationResult


# -----------------------
# Utilities
# -----------------------
def nm_str_from_value(val, as_meters=False, fmt="{:.3f}d0"):
    """Return Fortran-style value like '2.000d0'.
       val: if as_meters True, val is in meters -> convert to nm.
    """
    if as_meters:
        val = val * 1e9
    return fmt.format(val).replace("e","d")  # ensure d0

def _fmt_slug_value_nm(x: float, decimals: int = 1) -> str:
    """Format a thickness value in nm for folder slug, e.g., 2.0 -> '02.0'."""
    return f"{x:.{decimals}f}"


def _build_param_slug(
    RW: int,
    RQWt_nm: float,
    RQBt_nm: float,
    MQWt_nm: float,
    LW: int,
    LQWt_nm: float,
    LQBt_nm: float,
    decimals: int = 1,
) -> str:
    """Return a descriptive folder slug like:
    '05x02.0_07.0__02.0__01x02.0_07.0'
    """
    f = lambda v: _fmt_slug_value_nm(v, decimals)
    return (
        f"{int(RW):02d}x{f(RQWt_nm)}_{f(RQBt_nm)}__{f(MQWt_nm)}__{int(LW):02d}x{f(LQWt_nm)}_{f(LQBt_nm)}"
    )


def _resolve_executable(fortran_dir: Path) -> Path:
    """Try to find the Fortran executable inside fortran_dir.
    Raises FileNotFoundError if none found.
    """
    candidates = [
        fortran_dir / "prog"
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Fortran executable not found in {fortran_dir}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )

def run_fortran_sim(exe_path, 
                    RW, RQWt_nm, RQBt_nm, MQWt_nm, 
                    LW, LQWt_nm, LQBt_nm,
                    out_folder, 
                    timeout=120):
    """
    Run Fortran executable with properly formatted args (absolute paths).
    Returns the output folder path used (string) or raises on error.
    """
    out_folder = os.path.abspath(out_folder)
    os.makedirs(out_folder, exist_ok=True)
    # Format args: ints and nm->Fortran d0 format
    args = [
        str(int(RW)),
        nm_str_from_value(RQWt_nm, as_meters=False),
        nm_str_from_value(RQBt_nm, as_meters=False),
        nm_str_from_value(MQWt_nm, as_meters=False),
        str(int(LW)),
        nm_str_from_value(LQWt_nm, as_meters=False),
        nm_str_from_value(LQBt_nm, as_meters=False),
        str(os.path.abspath(out_folder)) + "/"  # trailing slash, pass as plain arg (no extra quotes)
    ]
    # Build command - avoid shell=True for safety; use executable path directly
    cmd = [os.path.abspath(exe_path)] + args
    # On some systems the exe may need chmod +x first
    if not os.access(exe_path, os.X_OK):
        os.chmod(exe_path, 0o755)
    try:
        # Run without shell to pass argv exactly (avoid shell quoting rules)
        print("Running command:", cmd)
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
    except Exception as e:
        raise RuntimeError("Erro ao executar exe: " + str(e))
    if res.returncode != 0:
        raise RuntimeError(f"Executável retornou código {res.returncode}. stderr: {res.stderr}")
    return out_folder

def run_qbmd(
    RW: int,
    RQWt: float,
    RQBt: float,
    MQWt: float,
    LW: int,
    LQWt: float,
    LQBt: float,
    *,
    thickness_units: Literal["nm", "m"] = "m",
    create_param_folder: bool = True,
    out_root: Optional[str] = None,
    timeout: int = 120,
    return_results: bool = False,
    peak_threshold: float = 0.1,
    simulation_id: Optional[int] = None,
) -> Union[str, Tuple[str, SimulationResult]]:
    """Run the QBMD Fortran program with the provided parameters.

    Parameters
    - RW, LW: integers (well counts)
    - RQWt, RQBt, MQWt, LQWt, LQBt: thickness values (float)
    - thickness_units: 'm' if values are in meters (e.g., 2e-9), 'nm' if already in nanometers
    - create_param_folder: if True, create a uniquely named folder under fortran_files/ based on parameters.
                           if False, reuse a stable folder (for GA runs) under fortran_files/GA_tmp
    - out_root: optional override for the root output directory. Defaults to linux_executable/fortran_files
    - timeout: seconds to wait for the executable
    - return_results: if True, also extract and return SimulationResult from photocurrent file
    - peak_threshold: threshold for peak detection (used when return_results=True)
    - simulation_id: optional simulation identifier for GA runs (used for folder naming)

    Returns
    - If return_results=False: Absolute path to the output folder used (string)
    - If return_results=True: Tuple of (output_folder_path, SimulationResult)
    """
    # Resolve root dirs
    repo_root = Path(__file__).resolve().parents[2]  # .../nano.compQBMD
    default_fortran_root = (repo_root / "linux_executable").resolve()
    fortran_root = Path(out_root).resolve() if out_root else default_fortran_root

    # Convert thicknesses to nm if needed
    if thickness_units not in ("nm", "m"):
        raise ValueError("thickness_units must be 'nm' or 'm'")
    factor = 1e9 if thickness_units == "m" else 1.0
    RQWt_nm = RQWt * factor
    RQBt_nm = RQBt * factor
    MQWt_nm = MQWt * factor
    LQWt_nm = LQWt * factor
    LQBt_nm = LQBt * factor

    # Determine output folder
    if create_param_folder:
        slug = _build_param_slug(RW, RQWt_nm, RQBt_nm, MQWt_nm, LW, LQWt_nm, LQBt_nm)
        if simulation_id is not None:
            # For GA runs, include simulation ID in path
            out_dir = (fortran_root / "GA_runs" / f"sim_{simulation_id:05d}" / slug).resolve()
        else:
            out_dir = (fortran_root / "manual_runs" / slug).resolve()
    else:
        out_dir = (fortran_root / "temp_files").resolve()

    preexisting = out_dir.exists()
    # Ensure folder exists for the executable; underlying run_fortran_sim may also create it
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find executable
    exe_path = _resolve_executable(fortran_root)

    # Run the simulation (run_fortran_sim formats numbers and ensures quotes/trailing slash)
    try:
        used_out = run_fortran_sim(
            str(exe_path),
            RW, RQWt_nm, RQBt_nm, MQWt_nm,
            LW, LQWt_nm, LQBt_nm,
            str(out_dir),
            timeout=timeout,
        )
    except Exception as exc:
        # If the folder did not exist before this call, remove it to avoid leaving empty/partial outputs.
        try:
            if not preexisting and out_dir.exists():
                shutil.rmtree(out_dir)
        except Exception:
            # Don't mask original exception; just attempt best-effort cleanup.
            pass
        
        if return_results:
            # Return default result for failed simulations
            return str(out_dir), _default_result()
        raise

    # Return results based on return_results flag
    if return_results:
        # Extract simulation results from photocurrent file
        photocurrent_file = Path(used_out) / "Photocurrent_SL.txt"
        if photocurrent_file.exists():
            try:
                simulation_result = getSimulationResult(
                    str(photocurrent_file),
                    peak_threshold
                )
            except Exception as e:
                print(f"  Error extracting results: {e}")
                simulation_result = _default_result()
        else:
            print(f"  Photocurrent file not found: {photocurrent_file}")
            simulation_result = _default_result()
        return used_out, simulation_result
    
    return used_out


def _default_result() -> SimulationResult:
    """Return default SimulationResult for failed simulations"""
    return SimulationResult(
        max_energy=0.0,
        max_photocurrent=0.0,
        prominence=0.0,
        quality_factor=0.0
    )


__all__ = ["run_qbmd"]
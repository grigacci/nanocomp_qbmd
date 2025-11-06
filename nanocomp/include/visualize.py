"""Visualization helpers for QBMD simulation outputs.

Place common plotting and summary utilities here so the notebook can import
`include.visualize` and call `visualize_all(out_dir)` to render the standard
figures and return computed metrics.

This module intentionally does lazy imports of matplotlib so it can be used in
headless environments for non-plotting tasks.
"""
from pathlib import Path
from typing import Dict, Any

import numpy as np
import os

from .extract import (
    _load_photocurrent_file,
    extract_metrics_from_photocurrent,
)


def get_paths(out_dir) -> Dict[str, Path]:
    """Return a dict of expected output file paths inside out_dir."""
    p = Path(out_dir)
    return {
        "Energy_SL": p / "Energy_SL.txt",
        "FimPrograma": p / "FimPrograma.txt",
        "OscStr_SL": p / "OscStr_SL.txt",
        "Photocurrent_SL": p / "Photocurrent_SL.txt",
        "Transmission_SL": p / "Transmission_SL.txt",
        "Potencial_SL": p / "Potencial_SL.txt",
        "Wavefunction_SL": p / "wavefunction_SL.txt",
    }


def _pretty_print(title: str, d: Dict[str, Any]):
    print("─" * 60)
    print(title)
    print("─" * 60)
    for k, v in d.items():
        print(f"{k:25s} : {v}")
    print()


def visualize_all(out_dir: str, show: bool = True) -> Dict[str, Any]:
    """Load standard files from `out_dir`, plot the common figures and
    return a small metrics dict.

    Returns a dict containing the photocurrent metrics (via extract.extract_metrics...)
    and the filepath map used.
    """
    out = Path(out_dir)
    paths = get_paths(out)

    # Lazy import matplotlib to avoid import in non-plotting runs
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    # Print existence summary
    exists = {k: p.exists() for k, p in paths.items()}
    _pretty_print(f"Files in {out}", exists)

    # Use the dedicated helpers to avoid duplicating logic
    results: Dict[str, Any] = {"paths": paths, "exists": exists, "metrics": {}}

    # call FimPrograma printer if present
    try:
        if exists.get("FimPrograma"):
            # print and also return the raw/parsed value
            try:
                print_fimprogram(out)
                results["fimprogram"] = True
            except Exception:
                # if printing fails, still continue
                results["fimprogram"] = False
    except Exception as e:
        print("visualize_all: error handling FimPrograma:", e)

    # Energy
    try:
        if exists.get("Energy_SL"):
            e_res = plot_energy(out, show=show)
            results["energy"] = e_res
    except Exception as e:
        print("visualize_all: energy helper failed:", e)

    # Photocurrent (also returns metrics)
    try:
        if exists.get("Photocurrent_SL"):
            fpc = plot_photocurrent(out, show=show)
            if isinstance(fpc, tuple) and len(fpc) == 3:
                results["photocurrent_fig"] = (fpc[0], fpc[1])
                results["metrics"]["photocurrent"] = fpc[2]
            else:
                results["photocurrent_data"] = fpc
    except Exception as e:
        print("visualize_all: photocurrent helper failed:", e)

    # OscStr
    try:
        if exists.get("OscStr_SL"):
            results["oscstr"] = plot_oscstr(out, show=show)
    except Exception as e:
        print("visualize_all: oscstr helper failed:", e)

    # Transmission
    try:
        if exists.get("Transmission_SL"):
            results["transmission"] = plot_transmission(out, show=show)
    except Exception as e:
        print("visualize_all: transmission helper failed:", e)

    # Potential
    try:
        if exists.get("Potencial_SL"):
            results["potential"] = plot_potential(out, show=show)
    except Exception as e:
        print("visualize_all: potential helper failed:", e)

    # Wavefunction (may return multiple figures)
    try:
        if exists.get("Wavefunction_SL"):
            wf = plot_wavefunction(out, show=show)
            # plot_wavefunction may return (fig, ax) or (fig1, ax1, fig2, ax2)
            results["wavefunction"] = wf
    except Exception as e:
        print("visualize_all: wavefunction helper failed:", e)

    return results


if __name__ == "__main__":
    # quick smoke test if executed directly (won't run in CI but is handy locally)
    import sys
    if len(sys.argv) > 1:
        visualize_all(sys.argv[1])

# -----------------------
# Individual plot helpers
# -----------------------
def _get_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def get_metrics(out_dir: str) -> Dict[str, Any]:
    """Return computed metrics (photocurrent metrics) and file existence map."""
    paths = get_paths(out_dir)
    exists = {k: p.exists() for k, p in paths.items()}
    metrics = {}
    if paths["Photocurrent_SL"].exists():
        metrics["photocurrent"] = extract_metrics_from_photocurrent(str(paths["Photocurrent_SL"]))
    return {"paths": paths, "exists": exists, "metrics": metrics}


def plot_energy(out_dir: str, show: bool = True):
    p = get_paths(out_dir)["Energy_SL"]
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.loadtxt(p)
    # print summary statistics similar to the notebook
    stats = {
        "num_points": int(data.size),
        "min (eV?)": float(np.min(data)),
        "max (eV?)": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
    }
    _pretty_print("Energy_SL.txt summary", stats)
    plt = _get_plt()
    if plt is None:
        return data
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(np.arange(data.size), data)
    ax.set_title("Energy_SL (index vs energy)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Energy (units in file)")
    ax.grid(True)
    if show:
        plt.show()
    return fig, ax


def plot_photocurrent(out_dir: str, show: bool = True):
    p = get_paths(out_dir)["Photocurrent_SL"]
    if not p.exists():
        raise FileNotFoundError(p)
    energy, photocurrent = _load_photocurrent_file(str(p))
    pc_metrics = extract_metrics_from_photocurrent(str(p))
    # print metrics like the notebook
    _pretty_print("Photocurrent_SL metrics", pc_metrics)
    plt = _get_plt()
    if plt is None:
        return energy, photocurrent, pc_metrics
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(energy, photocurrent, label="photocurrent")
    if pc_metrics.get("peak_energy") is not None:
        ax.axvline(pc_metrics["peak_energy"], linestyle="--", color="tab:orange", label="peak")
    ax.set_title("Photocurrent")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Photocurrent")
    ax.grid(True)
    ax.legend()
    if show:
        plt.show()
    return fig, ax, pc_metrics


def plot_oscstr(out_dir: str, show: bool = True):
    p = get_paths(out_dir)["OscStr_SL"]
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.loadtxt(p)
    # print summary similar to the notebook
    if data.ndim == 1:
        shape = (data.size,)
    else:
        shape = data.shape
    osc_summary = {"shape": shape}
    try:
        if data.ndim == 1:
            x = np.arange(data.size)
            y = data
        else:
            x = data[:, 0]
            y = data[:, 1]
        osc_summary.update({
            "x min/max": (float(np.min(x)), float(np.max(x))),
            "y min/max": (float(np.min(y)), float(np.max(y)))
        })
    except Exception:
        pass
    _pretty_print("OscStr_SL.txt summary", osc_summary)
    plt = _get_plt()
    if plt is None:
        return data
    if data.ndim == 1:
        x = np.arange(data.size)
        y = data
    else:
        x = data[:, 0]
        y = data[:, 1]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(x, y)
    ax.set_title("OscStr_SL")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    if show:
        plt.show()
    return fig, ax


def plot_transmission(out_dir: str, show: bool = True):
    p = get_paths(out_dir)["Transmission_SL"]
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.loadtxt(p)
    if data.ndim == 1:
        x = np.arange(data.size)
        y = data
    else:
        x = data[:, 0]
        y = data[:, 1]
    trans_summary = {
        "shape": data.shape,
        "x min/max": (float(np.min(x)), float(np.max(x))),
        "y min/max": (float(np.min(y)), float(np.max(y)))
    }
    _pretty_print("Transmission_SL.txt summary", trans_summary)
    plt = _get_plt()
    if plt is None:
        return data
    if data.ndim == 1:
        x = np.arange(data.size)
        y = data
    else:
        x = data[:, 0]
        y = data[:, 1]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(x, y)
    ax.set_title("Transmission_SL")
    ax.grid(True)
    if show:
        plt.show()
    return fig, ax


def plot_potential(out_dir: str, show: bool = True):
    p = get_paths(out_dir)["Potencial_SL"]
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.loadtxt(p)
    # print summary like the notebook
    if data.ndim == 1:
        x = np.arange(data.size)
        y = data
    else:
        x = data[:, 0]
        y = data[:, 1]
    pot_summary = {
        "shape": data.shape,
        "x min/max": (float(np.min(x)), float(np.max(x))),
        "y min/max": (float(np.min(y)), float(np.max(y)))
    }
    _pretty_print("Potencial_SL.txt summary", pot_summary)
    plt = _get_plt()
    if plt is None:
        return data
    if data.ndim == 1:
        x = np.arange(data.size)
        y = data
    else:
        x = data[:, 0]
        y = data[:, 1]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(x, y)
    ax.set_title("Potencial_SL")
    ax.grid(True)
    if show:
        plt.show()
    return fig, ax


def plot_wavefunction(out_dir: str, show: bool = True):
    p = get_paths(out_dir)["Wavefunction_SL"]
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.loadtxt(p)
    plt = _get_plt()
    if plt is None:
        return data
    # 1) plot first few columns as line plots (amplitude vs index)
    if data.ndim == 1:
        fig1, ax1 = plt.subplots(figsize=(10,5))
        x = np.arange(data.size)
        ax1.plot(x, data, label="col 0")
        ax1.set_title("Wavefunction - first columns (amplitude vs index)")
        ax1.set_xlabel("index (discrete position)")
        ax1.set_ylabel("amplitude")
        ax1.legend()
        ax1.grid(True)
        if show:
            plt.show()
        return fig1, ax1
    else:
        cols_to_plot = min(6, data.shape[1])
        fig1, ax1 = plt.subplots(figsize=(10,5))
        x = np.arange(data.shape[0])
        for c in range(cols_to_plot):
            ax1.plot(x, data[:, c], label=f"col {c}")
        ax1.set_title("Wavefunction - first columns (amplitude vs index)")
        ax1.set_xlabel("index (discrete position)")
        ax1.set_ylabel("amplitude")
        ax1.legend()
        ax1.grid(True)
        if show:
            plt.show()

        # 2) heatmap of the whole matrix (rows x cols) - use imshow
        fig2, ax2 = plt.subplots(figsize=(8,5))
        im = ax2.imshow(data.T, aspect='auto', origin='lower')
        ax2.set_title("Wavefunction heatmap (columns across x-axis)")
        ax2.set_ylabel("column index")
        ax2.set_xlabel("row index (position)")
        fig2.colorbar(im, ax=ax2, label="amplitude")
        if show:
            plt.show()
        return (fig1, ax1, fig2, ax2)


def plot_xy(x, y, title=None, xlabel=None, ylabel=None, show=True):
    """Small helper used in the notebook: plot x vs y with labels.
    Returns (fig, ax) or raw arrays when matplotlib isn't available.
    """
    plt = _get_plt()
    if plt is None:
        return x, y
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True)
    if show:
        plt.show()
    return fig, ax


def print_fimprogram(out_dir: str):
    """Read `FimPrograma.txt` and pretty-print its content (tries to coerce to float)."""
    p = get_paths(out_dir)["FimPrograma"]
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r") as f:
        text = f.read().strip()
    try:
        val = float(text)
        _pretty_print("FimPrograma.txt value", {"raw": text, "as_float": val})
    except Exception:
        _pretty_print("FimPrograma.txt (raw)", {"raw": text})


__all__ = [
    "get_paths",
    "visualize_all",
    "get_metrics",
    "plot_energy",
    "plot_photocurrent",
    "plot_oscstr",
    "plot_transmission",
    "plot_potential",
    "plot_wavefunction",
]

import os
import subprocess
import numpy as np
from math import isclose
from hashlib import sha1
import json
import time

# Optional: use scipy for robust peak detection if installed
try:
    from scipy.signal import find_peaks, peak_widths
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

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

def run_fortran_sim(exe_path, RW, RQWt_nm, RQBt_nm, MQWt_nm, LW, LQWt_nm, LQBt_nm,
                    out_folder, timeout=120):
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
        f'"{out_folder}/"'  # ensure trailing slash and quotes (prog might expect quotes)
    ]
    # Build command - avoid shell=True for safety; use executable path directly
    cmd = [os.path.abspath(exe_path)] + args
    # On some systems the exe may need chmod +x first
    if not os.access(exe_path, os.X_OK):
        os.chmod(exe_path, 0o755)
    # Join as single string only if necessary by your exe; we'll call via subprocess.run with shell=False
    # Many Fortran exes accept argv normally.
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
    except Exception as e:
        raise RuntimeError("Erro ao executar exe: " + str(e))
    if res.returncode != 0:
        raise RuntimeError(f"Executável retornou código {res.returncode}. stderr: {res.stderr}")
    return out_folder

# -----------------------
# Metrics extraction
# -----------------------
def _load_photocurrent_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Formato inesperado em Photocurrent_SL.txt")
    energy = data[:,0]
    photocurrent = data[:,1]
    return energy, photocurrent

def detect_peaks_numpy(energy, photocurrent, rel_threshold=0.05, min_distance=1):
    """Simple numpy peak finder: a point is peak if greater than neighbours.
       Returns list of (index, value).
    """
    peak_idxs = []
    N = len(photocurrent)
    for i in range(1, N-1):
        if photocurrent[i] > photocurrent[i-1] and photocurrent[i] > photocurrent[i+1]:
            peak_idxs.append(i)
    # filter by threshold relative to max
    if len(photocurrent) == 0:
        return []
    maxv = np.max(photocurrent)
    thr = rel_threshold * maxv
    filtered = [(i, photocurrent[i]) for i in peak_idxs if photocurrent[i] >= thr]
    # sort by value descending
    filtered.sort(key=lambda x: -x[1])
    return filtered

def extract_metrics_from_photocurrent(filepath, peak_rel_threshold=0.05, half_max_fraction=0.5):
    energy, photocurrent = _load_photocurrent_file(filepath)
    # basic stats
    total_abs_area = np.trapz(np.abs(photocurrent), energy)
    if SCIPY_AVAILABLE:
        # robust peak detection
        peaks, props = find_peaks(photocurrent, height=np.max(photocurrent)*peak_rel_threshold, distance=1)
        peak_heights = props['peak_heights'] if 'peak_heights' in props else photocurrent[peaks]
        order = np.argsort(-peak_heights)
        peak_indices = peaks[order]
    else:
        peak_list = detect_peaks_numpy(energy, photocurrent, rel_threshold=peak_rel_threshold)
        peak_indices = [p[0] for p in peak_list]
    if len(peak_indices) == 0:
        # no peaks above threshold
        return {
            "num_peaks": 0,
            "peak_energy": None,
            "peak_value": 0.0,
            "peak_area": 0.0,
            "total_abs_area": total_abs_area,
            "prominence_ratio": 0.0,
            "Q_factor": 0.0,
            "secondary_peaks_ratio": 0.0,
            "integrated_QE": 0.0
        }
    # main peak = first in peak_indices (highest)
    peak_idx = int(peak_indices[0])
    peak_energy = float(energy[peak_idx])
    peak_value = float(photocurrent[peak_idx])
    # define peak region via threshold or FWHM method
    thr = peak_rel_threshold * peak_value
    above_thr = photocurrent >= thr
    # ensure contiguous region around the peak
    # find left boundary
    left = peak_idx
    while left>0 and above_thr[left]:
        left -= 1
    # if we stopped while the point at left is below thr, take left+1
    left = max(0, left+1)
    right = peak_idx
    while right < len(photocurrent)-1 and above_thr[right]:
        right += 1
    right = min(len(photocurrent)-1, right-1)
    # area of main peak
    if right <= left:
        peak_area = 0.0
    else:
        peak_area = np.trapz(photocurrent[left:right+1], energy[left:right+1])
    # integrated QE = area above thr
    integrated_QE = np.trapz(photocurrent[above_thr], energy[above_thr]) if np.any(above_thr) else 0.0
    # FWHM and Q-factor (approx)
    if SCIPY_AVAILABLE:
        results_half = peak_widths(photocurrent, np.array([peak_idx]), rel_height=0.5)
        # results_half[0] is widths in samples; convert indexes to energy
        left_ip = results_half[2][0]; right_ip = results_half[3][0]
        # interpolate indices to energies
        # clamp
        left_idx = int(max(0, np.floor(left_ip)))
        right_idx = int(min(len(energy)-1, np.ceil(right_ip)))
        fwhm = energy[right_idx] - energy[left_idx] if right_idx>left_idx else 0.0
    else:
        half_max = peak_value * half_max_fraction
        above_half = photocurrent >= half_max
        inds = np.where(above_half)[0]
        if inds.size < 2:
            fwhm = 0.0
        else:
            fwhm = energy[inds[-1]] - energy[inds[0]]
    Q_factor = (peak_energy / fwhm) if fwhm > 1e-12 else 0.0
    # secondary peaks metric
    other_peak_sum = 0.0
    for idx in peak_indices[1:]:
        other_peak_sum += photocurrent[int(idx)]
    secondary_peaks_ratio = (other_peak_sum / peak_value) if peak_value!=0 else 0.0
    prominence_ratio = (peak_area / total_abs_area) if total_abs_area!=0 else 0.0
    return {
        "num_peaks": len(peak_indices),
        "peak_energy": peak_energy,
        "peak_value": peak_value,
        "peak_area": peak_area,
        "total_abs_area": total_abs_area,
        "prominence_ratio": prominence_ratio,
        "Q_factor": Q_factor,
        "secondary_peaks_ratio": secondary_peaks_ratio,
        "integrated_QE": integrated_QE
    }

# -----------------------
# High-level: run sim + extract metrics with caching
# -----------------------
def params_to_hash(params):
    """Deterministic hash for caching"""
    s = json.dumps(params, sort_keys=True).encode('utf-8')
    return sha1(s).hexdigest()

CACHE_DIR = "./sim_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def eval_simulation_with_cache(exe_path, params, base_outdir="./outputs", timeout=120):
    """
    params: dict with keys RW,RQWt_nm,RQBt_nm,MQWt_nm,LW,LQWt_nm,LQBt_nm
    Returns metrics dict
    """
    h = params_to_hash(params)
    cache_file = os.path.join(CACHE_DIR, f"{h}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    # build an output folder name unique to this hash
    out_folder = os.path.join(os.path.abspath(base_outdir), h[:10])
    # run simulation (may raise)
    run_fortran_sim(exe_path,
                    params['RW'], params['RQWt_nm'], params['RQBt_nm'],
                    params['MQWt_nm'], params['LW'], params['LQWt_nm'], params['LQBt_nm'],
                    out_folder, timeout=timeout)
    # read Photocurrent_SL.txt
    pc_file = os.path.join(out_folder, "Photocurrent_SL.txt")
    metrics = extract_metrics_from_photocurrent(pc_file)
    # store extra metadata
    metrics['params'] = params
    metrics['out_folder'] = out_folder
    with open(cache_file, "w") as f:
        json.dump(metrics, f, default=float)
    return metrics

# -----------------------
# Example fitness (weighted, normalized by sample min/max)
# -----------------------
def compute_fitness_from_metrics(metrics, normalization_stats=None, weights=None):
    """Compute scalar fitness. normalization_stats should be dict with min/max observed for each metric.
       If None, we use heuristic scalers.
    """
    # metrics used: peak_value (higher), Q_factor(higher), prominence_ratio(higher), integrated_QE(lower), secondary_peaks_ratio(lower)
    # fallback normalization:
    def norm(v, vmin, vmax):
        if vmax==vmin:
            return 0.0
        return (v - vmin) / (vmax - vmin)
    # heuristic ranges (tune on real data)
    defaults = {
        "peak_value": (0.0, 1.0),
        "Q_factor": (0.0, 100.0),
        "prominence_ratio": (0.0, 1.0),
        "integrated_QE": (0.0, 10.0),
        "secondary_peaks_ratio": (0.0, 5.0)
    }
    if normalization_stats is None:
        normalization_stats = {k:defaults[k] for k in defaults}
    if weights is None:
        weights = {"peak_value":0.4, "Q_factor":0.3, "prominence_ratio":0.2,
                   "integrated_QE":0.05, "secondary_peaks_ratio":0.05}
    pv = metrics.get("peak_value",0.0)
    qf = metrics.get("Q_factor",0.0)
    pr = metrics.get("prominence_ratio",0.0)
    iq = metrics.get("integrated_QE",0.0)
    spr = metrics.get("secondary_peaks_ratio",0.0)
    n_pv = norm(pv, *normalization_stats["peak_value"])
    n_qf = norm(qf, *normalization_stats["Q_factor"])
    n_pr = norm(pr, *normalization_stats["prominence_ratio"])
    n_iq = norm(iq, *normalization_stats["integrated_QE"])  # higher means worse
    n_spr = norm(spr, *normalization_stats["secondary_peaks_ratio"])
    # fitness: maximize is better
    fitness = weights["peak_value"]*n_pv + weights["Q_factor"]*n_qf + weights["prominence_ratio"]*n_pr \
              - weights["integrated_QE"]*n_iq - weights["secondary_peaks_ratio"]*n_spr
    # optional: penalize multiple peaks if you want single-peak
    if metrics.get("num_peaks",0) > 1:
        fitness *= 0.9  # small penalty (tune)
    return float(fitness)

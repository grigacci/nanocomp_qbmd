import numpy as np
import os
from .models import SimulationResult

# NumPy compatibility: trapezoid was renamed from trapz in NumPy 2.0
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz

def getSimulationResult(photocurrent_file, peak_detection_threshold=0.05):
    max_energy, max_photocurrent = getMaxPhotocurrent(photocurrent_file)
    promience = getGetProminenceMetric(photocurrent_file, max_photocurrent, peak_detection_threshold)
    quality_factor = getQualityFactorMetric(photocurrent_file, max_photocurrent)

    simuResultObj = SimulationResult(
        max_energy=max_energy,
        max_photocurrent=max_photocurrent,
        prominence=promience,
        quality_factor=quality_factor
    )

    return simuResultObj

def getMaxPhotocurrent(photocurrent_file):
    data = np.loadtxt(photocurrent_file)
    energy = data[:, 0]  # eV 
    photocurrent = data[:, 1]  # normalized units
    
    max_photocurrent = np.max(photocurrent)
    max_index = np.argmax(photocurrent)  
    max_energy = energy[max_index]
    
    return max_energy, max_photocurrent

def getQualityFactorMetric(photocurrent_file, peak_value):
    data = np.loadtxt(photocurrent_file)
    energy = data[:, 0]  # eV
    photocurrent = data[:, 1]  # normalized units
    
    peak_index = np.argmax(photocurrent)
    peak_energy = energy[peak_index]
    
    half_max = peak_value * 0.5
    above_half = photocurrent >= half_max

    if np.sum(above_half) < 2:
        return 0  # Not enough points to define FWHM

    indexes_above_half = np.where(above_half)[0]

    fwhm = energy[indexes_above_half[-1]] - energy[indexes_above_half[0]]
    
    Q_factor = peak_energy / fwhm if fwhm != 0 else 0
    
    return Q_factor

def getGetProminenceMetric(photocurrent_file, peak_value, peak_detection_threshold=0.05):
    data = np.loadtxt(photocurrent_file)
    energy = data[:, 0]  # eV
    photocurrent = data[:, 1]  # normalized units
    
    threshold_value = peak_detection_threshold * peak_value

    above_threshold =  photocurrent >= threshold_value

    peak_energy_integral = _trapezoid(photocurrent[above_threshold], energy[above_threshold])

    total_energy_integral = _trapezoid(np.abs(photocurrent), energy)

    # Calculate average photocurrent below threshold
    prominence = peak_energy_integral / total_energy_integral if total_energy_integral != 0 else 0
    
    return prominence

# -----------------------
# Metrics extraction (used by viz)
# -----------------------

try:
    from scipy.signal import find_peaks, peak_widths
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


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


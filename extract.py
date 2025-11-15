import numpy as np
from scipy.signal import find_peaks, peak_prominences
from models import SimulationResult

# Physical constants
H_C = 1240.0  # eV·nm (Planck constant × speed of light)

def getSimulationResult(photocurrent_file, peak_detection_threshold=0.05):
    """Extract metrics from photocurrent vs wavelength data."""
    
    # Load and validate data
    data = np.loadtxt(photocurrent_file)
    if data.size == 0:
        return _default_result()
    
    # Convert wavelength (nm) to energy (eV)
    wavelength_nm = data[:, 0]
    energy_ev = H_C / wavelength_nm
    
    # Handle signed photocurrent (take absolute value if all negative)
    photocurrent_raw = data[:, 1]
    if np.all(photocurrent_raw < 0):
        photocurrent = np.abs(photocurrent_raw)
    else:
        photocurrent = photocurrent_raw
    
    # Normalize photocurrent for consistent metric calculation
    photocurrent = photocurrent / np.max(photocurrent) if np.max(photocurrent) > 0 else photocurrent
    
    # Find the global peak
    peak_idx = np.argmax(photocurrent)
    
    # Calculate robust metrics
    prominence = calculate_prominence(energy_ev, photocurrent, peak_idx)
    q_factor = calculate_q_factor(energy_ev, photocurrent, peak_idx)
    
    return SimulationResult(
        max_energy=energy_ev[peak_idx],
        max_photocurrent=np.max(photocurrent_raw),  # Store raw intensity
        prominence=prominence,
        quality_factor=q_factor
    )

def calculate_q_factor(energy, photocurrent, peak_idx):
    """Calculate Q-factor with proper interpolation at boundaries."""
    
    peak_value = photocurrent[peak_idx]
    half_max = peak_value * 0.5
    
    # Find left boundary with interpolation
    left_idx = peak_idx
    while left_idx > 0 and photocurrent[left_idx] >= half_max:
        left_idx -= 1
    
    if left_idx == peak_idx or left_idx == len(energy) - 1:
        return 0.0  # No width found
    
    # Linear interpolation for precise left crossing point
    x1, x2 = energy[left_idx], energy[left_idx + 1]
    y1, y2 = photocurrent[left_idx], photocurrent[left_idx + 1]
    left_energy = x2 - (x2 - x1) * (y2 - half_max) / (y2 - y1)
    
    # Find right boundary with interpolation
    right_idx = peak_idx
    while right_idx < len(photocurrent) - 1 and photocurrent[right_idx] >= half_max:
        right_idx += 1
    
    if right_idx == peak_idx or right_idx == 0:
        return 0.0  # No width found
    
    # Linear interpolation for precise right crossing point
    x1, x2 = energy[right_idx - 1], energy[right_idx]
    y1, y2 = photocurrent[right_idx - 1], photocurrent[right_idx]
    right_energy = x1 + (x2 - x1) * (half_max - y1) / (y2 - y1)
    
    fwhm = right_energy - left_energy
    
    if fwhm <= 0:
        return 0.0
    
    return energy[peak_idx] / fwhm

def calculate_prominence(energy, photocurrent, peak_idx):
    """Calculate true prominence using scipy for accuracy."""
    
    # Find all peaks
    peaks, _ = find_peaks(photocurrent)
    
    if len(peaks) == 0:
        return 0.0
    
    # Find the peak closest to our main peak
    peak_distances = np.abs(peaks - peak_idx)
    main_peak_in_peaks = np.argmin(peak_distances)
    
    prominences = peak_prominences(photocurrent, peaks)[0]
    
    return prominences[main_peak_in_peaks]

def _default_result():
    """Return safe defaults for failed extractions."""
    return SimulationResult(
        max_energy=0.0,
        max_photocurrent=0.0,
        prominence=0.0,
        quality_factor=0.0
    )
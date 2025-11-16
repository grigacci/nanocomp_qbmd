import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from models import SimulationResult
import warnings
import os

# Physical constants
H_C = 1240.0  # eV·nm (Planck constant × speed of light)
PEAK_MARGIN = 0.05  # Fractional margin to consider peak "near edge"


# In extract.py
def getSimulationResult(photocurrent_file, peak_detection_threshold=0.05):
    """Extract metrics with zero-data detection."""
    
    if not os.path.exists(photocurrent_file):
        raise FileNotFoundError(f"File missing: {photocurrent_file}")
    
    if os.path.getsize(photocurrent_file) == 0:
        raise ValueError(f"File empty: {photocurrent_file}")
    
    try:
        data = np.loadtxt(photocurrent_file)
    except:
        raise ValueError(f"Invalid file format: {photocurrent_file}")
    
    if data.size == 0 or len(data.shape) != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid data shape: {data.shape}")
    
    wavelength_nm = data[:, 0]
    energy_ev = H_C / wavelength_nm
    
    photocurrent_raw = np.abs(data[:, 1])
    
    # Check if all values are zero
    if np.max(np.abs(photocurrent_raw)) < 1e-15:
        raise ValueError(f"All photocurrent values are zero")
    
    # Normalize safely
    max_pc = np.max(photocurrent_raw)
    if max_pc == 0:
        raise ValueError("Max photocurrent is zero")
    
    photocurrent = photocurrent_raw / max_pc
    
    # Find peak
    peak_idx = np.argmax(photocurrent)
    
    
    # Calculate metrics
    prominence = calculate_prominence(energy_ev, photocurrent, peak_idx)
    q_factor = calculate_q_factor(energy_ev, photocurrent, peak_idx)
    
    return SimulationResult(
        max_energy=energy_ev[peak_idx],
        max_photocurrent=np.max(photocurrent_raw),
        prominence=prominence,
        quality_factor=q_factor
    )


# In extract.py
def calculate_q_factor(energy, photocurrent, peak_idx):
    """
    Calculate Q-factor using scipy's robust peak_widths.
    Returns 0.0 for any invalid or suspicious case.
    """
    
    n_points = len(photocurrent)
    
    # --- STRICT: Peak must be in middle 80% ---
    # Increase margin to 15% to be safer
    if peak_idx < n_points * PEAK_MARGIN or peak_idx > n_points * (1 - PEAK_MARGIN):
        print(f"  ❌ Peak at index {peak_idx}/{n_points} is forbidden (edge region)")
        return 0.0
    
    # --- Use scipy's robust interpolation ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            # rel_height=0.5 for half-maximum
            results = peak_widths(
                photocurrent, 
                [peak_idx], 
                rel_height=0.5
            )
            
            # Unpack results
            width_index = results[0][0]      # Width in index units
            height = results[1][0]           # Height at half-max
            left_ip = results[2][0]          # Left interpolation point (sub-sample)
            right_ip = results[3][0]         # Right interpolation point (sub-sample)
            
            # --- VALIDATE INTERPOLATION RESULTS ---
            
            # Check for zero/invalid width
            if width_index <= 0.5:
                print(f"  ❌ peak_widths found zero width ({width_index:.3f} samples)")
                return 0.0
            
            # Check interpolation points are in valid range
            if left_ip < 0 or right_ip > n_points - 1:
                print(f"  ❌ Interpolation point out of bounds: L={left_ip:.1f}, R={right_ip:.1f}")
                return 0.0
            
            # --- Convert to energy units ---
            # Use np.interp for accurate sub-sample interpolation
            left_energy = np.interp(left_ip, np.arange(n_points), energy)
            right_energy = np.interp(right_ip, np.arange(n_points), energy)
            
            fwhm = right_energy - left_energy
            
            # --- FINAL VALIDATION ---
            if fwhm <= 0:
                print(f"  ❌ NEGATIVE FWHM: {fwhm:.6f} eV")
                print(f"      Left={left_energy:.3f} eV, Right={right_energy:.3f} eV")
                print(f"      This is a numerical artifact - returning 0.0")
                return 0.0
            
            # Sanity: FWHM should be less than 30% of spectrum range
            total_range = energy[-1] - energy[0]
            if fwhm > total_range * 0.3:
                print(f"  ❌ FWHM ({fwhm:.3f} eV) > 30% of spectrum range")
                return 0.0
            
            # Calculate Q-factor
            q_factor = energy[peak_idx] / fwhm
            
            # Cap to reasonable range
            if q_factor > 5000:
                print(f"  ⚠️  Capping Q-factor from {q_factor:.0f} to 500")
                q_factor = 500.0
            
            print(f"  ✅ Q-factor: {q_factor:.2f} (E={energy[peak_idx]:.3f} eV, FWHM={fwhm:.4f} eV)")
            return q_factor
            
        except Exception as e:
            print(f"  ❌ peak_widths failed: {e}")
            return 0.0

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


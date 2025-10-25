import numpy as np


def getQEMetric(photocurrent_file):
    data = np.loadtxt(photocurrent_file)
    energy = data[:, 0]  # eV
    photocurrent = data[:, 1]  # normalized units
    
    # Find peak photocurrent value
    QE_peak = np.max(photocurrent)
    
    return QE_peak


def getAdptiveIntegratedQE(photocurrent_file, threshold_fraction=0.05):
    """
    Detailed example showing how multiple points exceed threshold
    """
    data = np.loadtxt(photocurrent_file)
    energy = data[:, 0]
    photocurrent = data[:, 1]
    
    # Step 1: Find the MAXIMUM VALUE (not index)
    peak_value = np.max(photocurrent)  
    # Example: peak_value = 0.92
    
    # Step 2: Calculate threshold as percentage of peak
    threshold = peak_value * threshold_fraction  
    # Example: threshold = 0.92 * 0.05 = 0.046
    
    # Step 3: Create boolean mask for ALL points above threshold
    above_threshold = photocurrent > threshold
    # This compares EVERY element in photocurrent array
    # Example result: [F, F, F, T, T, T, T, T, T, T, F, F, F]
    #                          ^^^^^^^^^^^^^^^^^ 
    #                          Peak region captured
    
    # Step 4: Count how many points are above threshold
    num_points_above = np.sum(above_threshold)
    print(f"Peak value: {peak_value:.4f}")
    print(f"Threshold (5%): {threshold:.4f}")
    print(f"Number of points above threshold: {num_points_above}")
    
    # Step 5: Extract energies and photocurrents above threshold
    energy_above = energy[above_threshold]
    photocurrent_above = photocurrent[above_threshold]
    
    # Step 6: Integrate over the peak region
    if num_points_above > 0:
        QE_integrated = np.trapz(photocurrent_above, energy_above)
    else:
        QE_integrated = 0
    
    return QE_integrated, (energy_above[0], energy_above[-1])



def getDarkCurrentMetric(photocurrent_file, energy_threshold=100):
    data = np.loadtxt(photocurrent_file)
    energy = data[:, 0]  # eV
    photocurrent = data[:, 1]
    
    # Average photocurrent below threshold energy (no absorption)
    mask = energy < energy_threshold
    I_dark = np.mean(np.abs(photocurrent[mask]))
    
    return I_dark

def getSelectivityMetric(photocurrent_file, threshold=0.5):
    data = np.loadtxt(photocurrent_file)
    energy = data[:, 0]
    photocurrent = data[:, 1]
    
    peak_value = np.max(photocurrent)
    peak_energy = energy[np.argmax(photocurrent)]
    
    # Calculate FWHM (Full Width at Half Maximum)
    half_max = peak_value * threshold
    above_half = photocurrent > half_max
    bandwidth = energy[above_half][-1] - energy[above_half][0] if np.any(above_half) else 0
    
    selectivity = peak_value / bandwidth if bandwidth > 0 else 0
    
    return selectivity


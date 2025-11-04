import numpy as np
from models import SimulationResult

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

    peak_energy_integral = np.trapezoid(photocurrent[above_threshold], energy[above_threshold])

    total_energy_integral = np.trapezoid(np.abs(photocurrent), energy)

    # Calculate average photocurrent below threshold
    prominence = peak_energy_integral / total_energy_integral if total_energy_integral != 0 else 0
    
    return prominence



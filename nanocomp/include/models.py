class SimulationResult:
    max_energy: float
    max_photocurrent: float
    prominence: float
    quality_factor: float
    
def __init__(self, max_energy: float, max_photocurrent: float, prominence: float, quality_factor: float):
        self.max_energy = max_energy
        self.max_photocurrent = max_photocurrent
        self.prominence = prominence
        self.quality_factor = quality_factor
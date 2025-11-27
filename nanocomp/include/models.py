class SimulationResult:
    """Data class for storing simulation results."""
    max_energy: float
    max_photocurrent: float
    prominence: float
    quality_factor: float
    
    def __init__(self, max_energy: float, max_photocurrent: float, prominence: float, quality_factor: float):
        self.max_energy = max_energy
        self.max_photocurrent = max_photocurrent
        self.prominence = prominence
        self.quality_factor = quality_factor
    
    def __repr__(self):
        return (f"SimulationResult(max_energy={self.max_energy}, "
                f"max_photocurrent={self.max_photocurrent}, "
                f"prominence={self.prominence}, "
                f"quality_factor={self.quality_factor})")
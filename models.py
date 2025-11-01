from dataclasses import dataclass


@dataclass
class SimulationResult:
    max_energy: float
    max_photocurrent: float
    prominence: float
    quality_factor: float

# diagnostic.py
import numpy as np
import matplotlib.pyplot as plt
from extract import calculate_q_factor

# Load ONE of your failing results
data = np.loadtxt("./runs/sim_00002/08x9.4d0_10.9d0_1.1d0_09x12.7d0_4.0d0/Photocurrent_SL.txt")  # ADJUST PATH


wavelength_nm = data[:, 0]
energy_ev = 1240.0 / wavelength_nm
photocurrent_raw = data[:, 1]

# Take absolute value if needed
if np.all(photocurrent_raw < 0):
    photocurrent_raw = np.abs(photocurrent_raw)

# Normalize like your code does
photocurrent = photocurrent_raw / np.max(photocurrent_raw) if np.max(photocurrent_raw) > 0 else photocurrent_raw

# Find peak
peak_idx = np.argmax(photocurrent)
peak_energy = energy_ev[peak_idx]
peak_value = photocurrent[peak_idx]

print("\n" + "="*60)
print("DIAGNOSTIC REPORT")
print("="*60)
print(f"Peak index: {peak_idx} (out of {len(photocurrent)} points)")
print(f"Peak location: {peak_energy:.3f} eV (λ={wavelength_nm[peak_idx]:.1f} nm)")
print(f"Peak value: {peak_value:.6f}")
print(f"Data range: {energy_ev[0]:.3f} to {energy_ev[-1]:.3f} eV")
print(f"Peak distance from left edge: {peak_idx} points")
print(f"Peak distance from right edge: {len(photocurrent) - peak_idx - 1} points")
print("="*60)

# Try to calculate Q-factor
q = calculate_q_factor(energy_ev, photocurrent, peak_idx)
print(f"Q-factor calculated: {q:.3f}")

# Plot
plt.figure(figsize=(12, 5))

# Top: Photocurrent vs Energy
plt.subplot(1, 2, 1)
plt.plot(energy_ev, photocurrent, 'b-', label='Photocurrent')
plt.axvline(peak_energy, color='r', linestyle='--', label=f'Peak @ {peak_energy:.3f} eV')
plt.axhline(peak_value * 0.5, color='g', linestyle=':', label='Half-maximum')
plt.xlabel("Energy (eV)")
plt.ylabel("Normalized Photocurrent")
plt.title("Peak Detection")
plt.legend()
plt.grid(True, alpha=0.3)

# Bottom: Zoom to peak region
plt.subplot(1, 2, 2)
window = 50  # Show 50 points around peak
left = max(0, peak_idx - window)
right = min(len(energy_ev), peak_idx + window)
plt.plot(energy_ev[left:right], photocurrent[left:right], 'b-')
plt.axvline(peak_energy, color='r', linestyle='--')
plt.axhline(peak_value * 0.5, color='g', linestyle=':')
plt.xlabel("Energy (eV)")
plt.ylabel("Normalized Photocurrent")
plt.title("Zoomed View")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
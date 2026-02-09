"""Generate test CSV files for the HIPE Experimental Pipeline.

Run once:  python generate_test_csvs.py
Creates three CSV files in test_data/ that can be uploaded to the app.
"""

import numpy as np
import csv
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(OUT_DIR, exist_ok=True)


# --- 1. NMR DOSY data ---
# Simulates a HIPE emulsion with D ~ 8.5e-11 m²/s (large droplets)
np.random.seed(42)
n_points = 16
gradients = np.linspace(0.02, 0.5, n_points)

gamma = 2.675e8
delta = 2e-3
big_delta = 50e-3
b_values = (gamma * gradients * delta) ** 2 * (big_delta - delta / 3.0)

D_true = 8.5e-11
I0 = 1000.0
intensities = I0 * np.exp(-D_true * b_values) + np.random.normal(0, 15.0, n_points)
intensities = np.clip(intensities, 0, None)

with open(os.path.join(OUT_DIR, "nmr_dosy_hipe.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["gradient_strength_T_per_m", "signal_intensity_au"])
    for g, i in zip(gradients, intensities):
        writer.writerow([f"{g:.4f}", f"{i:.2f}"])

print(f"Created: nmr_dosy_hipe.csv  ({n_points} data points, D_true={D_true:.1e} m²/s)")


# --- 2. Circular Dichroism data ---
# Simulates an IgG mAb (beta-sheet dominant): negative band ~218 nm, positive ~197 nm
np.random.seed(42)
wavelengths = np.linspace(190, 260, 150)
ellipticity = (
    -8.0 * np.exp(-0.5 * ((wavelengths - 218) / 8) ** 2)
    + 12.0 * np.exp(-0.5 * ((wavelengths - 197) / 6) ** 2)
    - 2.0 * np.exp(-0.5 * ((wavelengths - 235) / 15) ** 2)
)
ellipticity += np.random.normal(0, 0.3, len(wavelengths))

with open(os.path.join(OUT_DIR, "cd_spectrum_igg.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["wavelength_nm", "molar_ellipticity_x1000"])
    for w, e in zip(wavelengths, ellipticity):
        writer.writerow([f"{w:.1f}", f"{e:.3f}"])

print(f"Created: cd_spectrum_igg.csv  ({len(wavelengths)} data points, beta-sheet pattern)")


# --- 3. Stressed / unfolded CD data (for comparison) ---
np.random.seed(99)
ellipticity_unfolded = (
    -20.0 * np.exp(-0.5 * ((wavelengths - 198) / 6) ** 2)
    + 2.0 * np.exp(-0.5 * ((wavelengths - 218) / 10) ** 2)
)
ellipticity_unfolded += np.random.normal(0, 0.4, len(wavelengths))

with open(os.path.join(OUT_DIR, "cd_spectrum_unfolded.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["wavelength_nm", "molar_ellipticity_x1000"])
    for w, e in zip(wavelengths, ellipticity_unfolded):
        writer.writerow([f"{w:.1f}", f"{e:.3f}"])

print(f"Created: cd_spectrum_unfolded.csv  ({len(wavelengths)} data points, random coil)")


# --- 4. Aggregated protein DOSY data (larger particles, slower diffusion) ---
np.random.seed(77)
D_agg = 2.0e-11  # much slower diffusion → larger aggregates
intensities_agg = I0 * np.exp(-D_agg * b_values) + np.random.normal(0, 20.0, n_points)
intensities_agg = np.clip(intensities_agg, 0, None)

with open(os.path.join(OUT_DIR, "nmr_dosy_aggregated.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["gradient_strength_T_per_m", "signal_intensity_au"])
    for g, i in zip(gradients, intensities_agg):
        writer.writerow([f"{g:.4f}", f"{i:.2f}"])

print(f"Created: nmr_dosy_aggregated.csv  ({n_points} data points, D_true={D_agg:.1e} m²/s — aggregated)")


# --- 5. Bi-exponential DOSY data (high-concentration HIPE, restricted diffusion) ---
np.random.seed(55)
Df_true, Ds_true, xi_true = 6e-10, 5e-11, 0.55
intensities_biexp = I0 * (
    xi_true * np.exp(-Df_true * b_values) + (1 - xi_true) * np.exp(-Ds_true * b_values)
) + np.random.normal(0, 10.0, n_points)
intensities_biexp = np.clip(intensities_biexp, 0, None)

with open(os.path.join(OUT_DIR, "nmr_dosy_biexp_hipe.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["gradient_strength_T_per_m", "signal_intensity_au"])
    for g, i in zip(gradients, intensities_biexp):
        writer.writerow([f"{g:.4f}", f"{i:.2f}"])

print(f"Created: nmr_dosy_biexp_hipe.csv  ({n_points} data points, Df={Df_true:.1e}, Ds={Ds_true:.1e} m²/s — bi-exponential)")


print("\nAll test files saved to:", OUT_DIR)

"""
HIPE Analysis Module
--------------------
Data processing, curve fitting, and example data generation for
NMR DOSY diffusion, Circular Dichroism, and Microscopy analysis.

All example data is clearly synthetic and for demonstration purposes only.
"""

import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import io
import csv


# =============================================================
# NMR DOSY Diffusion Analysis
# =============================================================

def mono_exponential_decay(b, I0, D):
    """Mono-exponential decay for DOSY diffusion.

    I = I0 * exp(-D * b)
    where b = (gamma * g * delta)^2 * (Delta - delta/3)
    """
    return I0 * np.exp(-D * b)


def calculate_b_values(gradients, gamma=2.675e8, delta=2e-3, big_delta=50e-3):
    """Calculate b-values from gradient strengths.

    Args:
        gradients: gradient strengths in T/m
        gamma: gyromagnetic ratio (1H default)
        delta: gradient pulse duration (s)
        big_delta: diffusion delay (s)
    """
    return (gamma * gradients * delta) ** 2 * (big_delta - delta / 3.0)


def fit_dosy_decay(gradients, intensities, gamma=2.675e8, delta=2e-3, big_delta=50e-3):
    """Mono-exponential fit: S = I0 * exp(-D * b).

    Appropriate for water, low-concentration HIPE, or free diffusion regimes.

    Returns:
        dict with D, I0, r_squared, b_values, fitted_intensities, model
    """
    b_values = calculate_b_values(gradients, gamma, delta, big_delta)

    popt, _ = curve_fit(
        mono_exponential_decay, b_values, intensities,
        p0=[intensities[0], 1e-10],
        bounds=([0, 0], [np.inf, 1e-6]),
    )
    I0_fit, D_fit = popt

    fitted = mono_exponential_decay(b_values, I0_fit, D_fit)
    ss_res = np.sum((intensities - fitted) ** 2)
    ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "model": "mono-exponential",
        "D": D_fit,
        "I0": I0_fit,
        "r_squared": r_squared,
        "b_values": b_values,
        "fitted_intensities": fitted,
    }


# --- Bi-exponential model (Scigliani, Grant & Mohammadigoushki, 2023) ---

def biexponential_decay(b, I0, xi, Df, Ds):
    """Bi-exponential signal attenuation for restricted diffusion.

    S(B) = I0 * [xi * exp(-B * Df) + (1 - xi) * exp(-B * Ds)]

    Args:
        b: diffusion weighting (b-values)
        I0: initial signal intensity
        xi: pool fraction for fast diffusion component (0 to 1)
        Df: fast apparent diffusion coefficient (m^2/s)
        Ds: slow apparent diffusion coefficient (m^2/s)
    """
    return I0 * (xi * np.exp(-Df * b) + (1.0 - xi) * np.exp(-Ds * b))


def fit_dosy_biexp(gradients, intensities, gamma=2.675e8, delta=2e-3, big_delta=50e-3):
    """Bi-exponential fit: S = I0 * [xi*exp(-Df*b) + (1-xi)*exp(-Ds*b)].

    Appropriate for higher-concentration HIPE or restricted diffusion regimes
    where two distinct diffusion pools exist.

    Returns:
        dict with Df, Ds, xi, I0, r_squared, b_values, fitted_intensities, model
    """
    b_values = calculate_b_values(gradients, gamma, delta, big_delta)

    # Initial guesses: xi=0.5, Df ~ 1e-9, Ds ~ 1e-10
    p0 = [intensities[0], 0.5, 1e-9, 1e-10]
    bounds = (
        [0, 0.01, 0, 0],            # lower bounds
        [np.inf, 0.99, 1e-6, 1e-6],  # upper bounds
    )

    popt, _ = curve_fit(
        biexponential_decay, b_values, intensities,
        p0=p0, bounds=bounds, maxfev=50000,
    )
    I0_fit, xi_fit, Df_fit, Ds_fit = popt

    # Ensure Df >= Ds by convention
    if Ds_fit > Df_fit:
        Df_fit, Ds_fit = Ds_fit, Df_fit
        xi_fit = 1.0 - xi_fit

    fitted = biexponential_decay(b_values, I0_fit, xi_fit, Df_fit, Ds_fit)
    ss_res = np.sum((intensities - fitted) ** 2)
    ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "model": "bi-exponential",
        "Df": Df_fit,
        "Ds": Ds_fit,
        "xi": xi_fit,
        "I0": I0_fit,
        "r_squared": r_squared,
        "b_values": b_values,
        "fitted_intensities": fitted,
    }


def _aic(n, rss, k):
    """Akaike Information Criterion: AIC = n*ln(RSS/n) + 2k."""
    return n * np.log(rss / n) + 2 * k


def auto_fit_dosy(gradients, intensities, gamma=2.675e8, delta=2e-3, big_delta=50e-3):
    """Automatically select mono- or bi-exponential fit using AIC.

    Fits both models, then picks the one with the lower AIC score.
    Bi-exponential is only preferred if it provides a statistically
    meaningful improvement (AIC penalty for 4 params vs 2).

    Falls back to mono-exponential if bi-exponential fitting fails
    or if fewer than 6 data points are available (not enough to
    reliably fit 4 parameters).

    Returns:
        dict with fit results (same format as fit_dosy_decay or fit_dosy_biexp)
        plus 'selection_reason' explaining why the model was chosen.
    """
    n = len(gradients)

    # Always fit mono-exponential first
    try:
        mono = fit_dosy_decay(gradients, intensities, gamma, delta, big_delta)
    except Exception as exc:
        raise RuntimeError(
            "Could not fit the data. Check that column 1 is gradient strength (T/m) "
            f"and column 2 is signal intensity. ({type(exc).__name__}: {exc})"
        ) from None
    mono_rss = np.sum((intensities - mono["fitted_intensities"]) ** 2)
    mono_aic = _aic(n, mono_rss, k=2)

    # Need at least 6 points to reliably fit 4 parameters
    if n < 6:
        mono["selection_reason"] = (
            f"Mono-exponential selected (only {n} data points — "
            "too few for bi-exponential fit)"
        )
        return mono

    # Try bi-exponential
    try:
        biexp = fit_dosy_biexp(gradients, intensities, gamma, delta, big_delta)
        biexp_rss = np.sum((intensities - biexp["fitted_intensities"]) ** 2)
        biexp_aic = _aic(n, biexp_rss, k=4)

        # Check if Df and Ds actually separated (ratio > 2x means distinct pools)
        ratio = biexp["Df"] / biexp["Ds"] if biexp["Ds"] > 0 else 1.0
        pools_separated = ratio > 2.0

        if biexp_aic < mono_aic and pools_separated:
            biexp["selection_reason"] = (
                f"Bi-exponential selected (AIC: {biexp_aic:.1f} vs {mono_aic:.1f}, "
                f"Df/Ds ratio: {ratio:.1f}x — two distinct diffusion pools detected)"
            )
            return biexp
        else:
            reason = []
            if biexp_aic >= mono_aic:
                reason.append(f"AIC favors mono ({mono_aic:.1f} vs {biexp_aic:.1f})")
            if not pools_separated:
                reason.append(f"Df/Ds ratio only {ratio:.1f}x (pools not distinct)")
            mono["selection_reason"] = (
                f"Mono-exponential selected ({'; '.join(reason)})"
            )
            return mono

    except Exception:
        mono["selection_reason"] = (
            "Mono-exponential selected (bi-exponential fit failed to converge)"
        )
        return mono


def generate_example_dosy_data(regime="low_concentration"):
    """Generate synthetic DOSY decay data for demonstration.

    Regimes:
        - low_concentration: free diffusion, mono-exponential (D ~ 8.5e-11)
        - high_concentration: restricted diffusion, bi-exponential (Df ~ 6e-10, Ds ~ 5e-11)
    """
    np.random.seed(42)
    n_points = 16
    I0 = 1000.0
    gradients = np.linspace(0.02, 0.5, n_points)
    b_values = calculate_b_values(gradients)

    if regime == "high_concentration":
        # Restricted diffusion: two pools (fast + slow)
        Df_true, Ds_true, xi_true = 6e-10, 5e-11, 0.55
        intensities_clean = biexponential_decay(b_values, I0, xi_true, Df_true, Ds_true)
        noise = np.random.normal(0, 10.0, n_points)
    else:
        # Free diffusion: single exponential
        D_true = 8.5e-11
        intensities_clean = mono_exponential_decay(b_values, I0, D_true)
        noise = np.random.normal(0, 15.0, n_points)

    intensities = np.clip(intensities_clean + noise, 0, None)
    return gradients, intensities


def plot_dosy_results(gradients, intensities, fit_result):
    """Create a Plotly figure for DOSY decay with fit (mono or bi-exponential)."""
    b_values = fit_result["b_values"]
    b_smooth = np.linspace(b_values.min(), b_values.max(), 200)

    if fit_result["model"] == "bi-exponential":
        fit_smooth = biexponential_decay(
            b_smooth, fit_result["I0"], fit_result["xi"],
            fit_result["Df"], fit_result["Ds"],
        )
        fit_label = "Bi-exponential Fit"
    else:
        fit_smooth = mono_exponential_decay(b_smooth, fit_result["I0"], fit_result["D"])
        fit_label = "Mono-exponential Fit"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=b_values, y=intensities,
        mode="markers", name="Experimental Data",
        marker=dict(size=10, color="#4da6ff"),
    ))
    fig.add_trace(go.Scatter(
        x=b_smooth, y=fit_smooth,
        mode="lines", name=fit_label,
        line=dict(color="#ff4b4b", width=2),
    ))
    fig.update_layout(
        title="NMR DOSY Diffusion Decay",
        xaxis_title="b-value (s/m\u00b2)",
        yaxis_title="Signal Intensity (a.u.)",
        template="plotly_dark",
        height=450,
        legend=dict(x=0.6, y=0.95),
    )
    return fig


# =============================================================
# Circular Dichroism Analysis
# =============================================================

def generate_example_cd_data(structure="beta_sheet", noise_level=0.3):
    """Generate synthetic CD spectrum for demonstration.

    Structures:
        - beta_sheet: typical mAb IgG (predominantly beta-sheet)
        - alpha_helix: alpha-helical protein for comparison
        - unfolded: denatured/random coil
    """
    np.random.seed(42)
    wavelengths = np.linspace(190, 260, 150)

    if structure == "beta_sheet":
        # Typical IgG mAb: negative band ~218 nm, positive ~195-200 nm
        ellipticity = (
            -8.0 * np.exp(-0.5 * ((wavelengths - 218) / 8) ** 2)
            + 12.0 * np.exp(-0.5 * ((wavelengths - 197) / 6) ** 2)
            - 2.0 * np.exp(-0.5 * ((wavelengths - 235) / 15) ** 2)
        )
    elif structure == "alpha_helix":
        # Alpha-helix: negative bands at 208 and 222 nm, positive at 193 nm
        ellipticity = (
            -15.0 * np.exp(-0.5 * ((wavelengths - 208) / 5) ** 2)
            - 15.0 * np.exp(-0.5 * ((wavelengths - 222) / 5) ** 2)
            + 25.0 * np.exp(-0.5 * ((wavelengths - 193) / 4) ** 2)
        )
    else:
        # Random coil / unfolded: strong negative at ~198 nm, weak elsewhere
        ellipticity = (
            -20.0 * np.exp(-0.5 * ((wavelengths - 198) / 6) ** 2)
            + 2.0 * np.exp(-0.5 * ((wavelengths - 218) / 10) ** 2)
        )

    noise = np.random.normal(0, noise_level, len(wavelengths))
    ellipticity += noise

    return wavelengths, ellipticity


def analyze_cd_spectrum(wavelengths, ellipticity):
    """Basic CD spectrum analysis: find peaks, troughs, and estimate structure."""
    min_idx = np.argmin(ellipticity)
    max_idx = np.argmax(ellipticity)

    min_wl = wavelengths[min_idx]
    min_val = ellipticity[min_idx]
    max_wl = wavelengths[max_idx]
    max_val = ellipticity[max_idx]

    # Simple secondary structure estimation from spectral features
    if 215 <= min_wl <= 222 and max_wl < 200:
        structure = "Predominantly beta-sheet (typical IgG mAb pattern)"
    elif 205 <= min_wl <= 212:
        structure = "Predominantly alpha-helical"
    elif min_wl < 200:
        structure = "Predominantly random coil / unfolded"
    else:
        structure = "Mixed secondary structure"

    return {
        "min_wavelength": min_wl,
        "min_ellipticity": min_val,
        "max_wavelength": max_wl,
        "max_ellipticity": max_val,
        "estimated_structure": structure,
    }


def plot_cd_spectrum(wavelengths, ellipticity, analysis):
    """Create a Plotly figure for CD spectrum."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelengths, y=ellipticity,
        mode="lines", name="CD Spectrum",
        line=dict(color="#4da6ff", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_trace(go.Scatter(
        x=[analysis["min_wavelength"]],
        y=[analysis["min_ellipticity"]],
        mode="markers", name=f"Min ({analysis['min_wavelength']:.0f} nm)",
        marker=dict(size=12, color="#ff4b4b", symbol="triangle-down"),
    ))
    fig.add_trace(go.Scatter(
        x=[analysis["max_wavelength"]],
        y=[analysis["max_ellipticity"]],
        mode="markers", name=f"Max ({analysis['max_wavelength']:.0f} nm)",
        marker=dict(size=12, color="#00cc66", symbol="triangle-up"),
    ))
    fig.update_layout(
        title="Circular Dichroism Spectrum",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Molar Ellipticity (\u00d710\u00b3 deg\u00b7cm\u00b2/dmol)",
        template="plotly_dark",
        height=450,
    )
    return fig


# =============================================================
# Microscopy Timeline Analysis
# =============================================================

def generate_example_microscopy_timeline():
    """Generate synthetic microscopy stability data (droplet size over time).

    Returns day numbers and average droplet diameters in micrometers.
    """
    np.random.seed(42)
    days = np.array([0, 1, 2, 3, 5, 7, 10, 14])

    # Stable HIPE: slight increase in droplet size due to minor Ostwald ripening
    base_size = 5.2  # micrometers
    sizes = base_size + 0.08 * days + np.random.normal(0, 0.15, len(days))
    sizes = np.clip(sizes, 4.0, None)

    # Standard deviation of droplet sizes (polydispersity)
    std_devs = 1.2 + 0.05 * days + np.random.normal(0, 0.1, len(days))
    std_devs = np.clip(std_devs, 0.5, None)

    return days, sizes, std_devs


def plot_microscopy_timeline(days, sizes, std_devs):
    """Create a Plotly figure for droplet size over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=sizes,
        mode="lines+markers",
        name="Mean Droplet Diameter",
        line=dict(color="#4da6ff", width=2),
        marker=dict(size=10),
        error_y=dict(type="data", array=std_devs, visible=True, color="#4da6ff"),
    ))
    fig.update_layout(
        title="HIPE Droplet Size Stability Over Time",
        xaxis_title="Time (days)",
        yaxis_title="Mean Droplet Diameter (\u00b5m)",
        template="plotly_dark",
        height=450,
    )
    return fig


# =============================================================
# CSV Parsing Utilities
# =============================================================

def parse_csv_two_columns(uploaded_file):
    """Parse a two-column CSV file (x, y) and return cleaned numpy arrays.

    Handles:
        - UTF-8 BOM markers
        - Tab and semicolon delimiters (auto-detected)
        - Header rows and non-numeric lines (skipped)
        - NaN / Inf values (removed)
        - Duplicate x-values (removed, keeps first)
        - Result sorted by x ascending
    """
    raw = uploaded_file.getvalue()
    # Handle BOM; fall back to latin-1 for non-UTF8 files
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")

    # Auto-detect delimiter: try tab, semicolon, then comma
    first_line = content.split("\n", 1)[0]
    if "\t" in first_line:
        delimiter = "\t"
    elif ";" in first_line:
        delimiter = ";"
    else:
        delimiter = ","

    reader = csv.reader(io.StringIO(content), delimiter=delimiter)

    x_vals, y_vals = [], []
    for row in reader:
        if len(row) < 2:
            continue
        try:
            x = float(row[0].strip())
            y = float(row[1].strip())
        except ValueError:
            continue  # skip header or non-numeric rows
        # Skip NaN / Inf
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x_vals.append(x)
        y_vals.append(y)

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)

    if len(x_arr) == 0:
        return x_arr, y_arr

    # Remove duplicate x-values (keep first occurrence)
    _, unique_idx = np.unique(x_arr, return_index=True)
    x_arr = x_arr[unique_idx]
    y_arr = y_arr[unique_idx]

    # Sort by x ascending
    sort_idx = np.argsort(x_arr)
    x_arr = x_arr[sort_idx]
    y_arr = y_arr[sort_idx]

    return x_arr, y_arr

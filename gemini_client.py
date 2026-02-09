import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
try:
    from google.api_core.exceptions import DeadlineExceeded, InternalServerError
    _RETRYABLE = (ResourceExhausted, ServiceUnavailable, DeadlineExceeded, InternalServerError)
except ImportError:
    _RETRYABLE = (ResourceExhausted, ServiceUnavailable)
import streamlit as st
from prompts import (
    SYSTEM_INSTRUCTION,
    STABILITY_ANALYSIS_PROMPT,
    MECHANISM_EXPLANATION_PROMPT,
    FORMULATION_COMPARISON_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    NMR_DOSY_INTERPRETATION_PROMPT,
    CD_INTERPRETATION_PROMPT,
    MICROSCOPY_INTERPRETATION_PROMPT,
)

MAX_RETRIES = 3
BASE_DELAY = 2  # seconds


def _stream_with_retry(model, prompt):
    """Generate streaming content with exponential backoff on rate limits.

    Retries on 429/500/503/504, but NEVER after content has been yielded
    (retrying mid-stream would duplicate output and corrupt the response).
    """
    chunks_yielded = False
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    chunks_yielded = True
                    yield chunk.text
            return  # success — exit
        except _RETRYABLE as e:
            # Never retry after partial output — would duplicate content
            if chunks_yielded or attempt == MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt)
            st.toast(f"Rate limited — retrying in {delay}s (attempt {attempt + 2}/{MAX_RETRIES})")
            time.sleep(delay)
        except ValueError as e:
            # google.generativeai raises ValueError on blocked/filtered content
            msg = str(e).lower()
            if "block" in msg or "safety" in msg or "finish_reason" in msg:
                raise RuntimeError(
                    "Response blocked by content safety filter. "
                    "Try adjusting parameters or switching models."
                ) from None
            raise


def initialize_client():
    """Configure the Gemini API with the stored key.

    Checks Streamlit secrets first, then environment variable.
    Returns True if configured, False otherwise.
    """
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return False

    genai.configure(api_key=api_key)
    return True


def get_model(model_name="gemini-3-pro-preview"):
    """Create a GenerativeModel with the mAb expert system instruction."""
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.95,
        ),
    )


def stream_stability(
    model_name, concentration, ph, buffer, protein,
    protein_concentration, temperature, interface_type
):
    """Run stability analysis and yield streaming chunks."""
    model = get_model(model_name)
    prompt = STABILITY_ANALYSIS_PROMPT.format(
        concentration=concentration,
        ph=ph,
        buffer=buffer,
        protein=protein,
        protein_concentration=protein_concentration,
        temperature=temperature,
        interface_type=interface_type,
    )
    yield from _stream_with_retry(model, prompt)


def stream_mechanism(model_name, interface_type):
    """Run mechanism explanation and yield streaming chunks."""
    model = get_model(model_name)
    prompt = MECHANISM_EXPLANATION_PROMPT.format(
        interface_type=interface_type,
    )
    yield from _stream_with_retry(model, prompt)


def stream_comparison(
    model_name,
    conc_a, ph_a, buffer_a, protein_a, protein_conc_a, temp_a,
    conc_b, ph_b, buffer_b, protein_b, protein_conc_b, temp_b,
):
    """Run formulation comparison and yield streaming chunks."""
    model = get_model(model_name)
    prompt = FORMULATION_COMPARISON_PROMPT.format(
        conc_a=conc_a, ph_a=ph_a, buffer_a=buffer_a,
        protein_a=protein_a, protein_conc_a=protein_conc_a, temp_a=temp_a,
        conc_b=conc_b, ph_b=ph_b, buffer_b=buffer_b,
        protein_b=protein_b, protein_conc_b=protein_conc_b, temp_b=temp_b,
    )
    yield from _stream_with_retry(model, prompt)


def stream_literature(model_name, topic):
    """Run literature review and yield streaming chunks."""
    model = get_model(model_name)
    prompt = LITERATURE_REVIEW_PROMPT.format(topic=topic)
    yield from _stream_with_retry(model, prompt)


def stream_nmr_interpretation(model_name, fit_result, n_gradients):
    """Run NMR DOSY interpretation and yield streaming chunks.

    Args:
        model_name: Gemini model to use
        fit_result: dict from fit_dosy_decay or fit_dosy_biexp
        n_gradients: number of gradient steps
    """
    model = get_model(model_name)

    if fit_result["model"] == "bi-exponential":
        fit_model = "Bi-exponential: S(b) = I0 [xi exp(-Df b) + (1-xi) exp(-Ds b)]"
        fit_details = (
            f"- **Fast diffusion coefficient (Df):** {fit_result['Df']:.2e} m^2/s\n"
            f"- **Slow diffusion coefficient (Ds):** {fit_result['Ds']:.2e} m^2/s\n"
            f"- **Fast pool fraction (xi):** {fit_result['xi']:.3f}\n"
            f"- **Df/Ds ratio:** {fit_result['Df']/max(fit_result['Ds'], 1e-30):.1f}x"
        )
        interp_guidance = (
            "Explain what the fast (Df) and slow (Ds) diffusion coefficients indicate "
            "about the sample. What does the pool fraction (xi) reveal about the relative "
            "abundance of freely vs. restricted diffusing species? What does the Df/Ds ratio "
            "tell us about the degree of diffusion restriction?"
        )
    else:
        fit_model = "Mono-exponential: S(b) = I0 exp(-D b)"
        fit_details = (
            f"- **Diffusion coefficient (D):** {fit_result['D']:.2e} m^2/s"
        )
        interp_guidance = (
            "What does this single diffusion coefficient indicate about the sample's "
            "hydrodynamic radius and molecular/particle size? Is a mono-exponential model "
            "appropriate, or should a bi-exponential model be considered?"
        )

    prompt = NMR_DOSY_INTERPRETATION_PROMPT.format(
        fit_model=fit_model,
        fit_details=fit_details,
        r_squared=f"{fit_result['r_squared']:.4f}",
        n_gradients=n_gradients,
        interp_guidance=interp_guidance,
    )
    yield from _stream_with_retry(model, prompt)


def stream_cd_interpretation(
    model_name, wavelength_range, dominant_features, sample_type,
    min_wavelength, max_wavelength
):
    """Run CD interpretation and yield streaming chunks."""
    model = get_model(model_name)
    prompt = CD_INTERPRETATION_PROMPT.format(
        wavelength_range=wavelength_range,
        dominant_features=dominant_features,
        sample_type=sample_type,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
    )
    yield from _stream_with_retry(model, prompt)


def stream_microscopy_interpretation(
    model_name, sample_type, observation_period, observations, magnification
):
    """Run microscopy interpretation and yield streaming chunks."""
    model = get_model(model_name)
    prompt = MICROSCOPY_INTERPRETATION_PROMPT.format(
        sample_type=sample_type,
        observation_period=observation_period,
        observations=observations,
        magnification=magnification,
    )
    yield from _stream_with_retry(model, prompt)

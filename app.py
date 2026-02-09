import re
import hashlib
import streamlit as st
import plotly.graph_objects as go
from gemini_client import (
    initialize_client,
    stream_stability,
    stream_mechanism,
    stream_comparison,
    stream_literature,
    stream_nmr_interpretation,
    stream_cd_interpretation,
    stream_microscopy_interpretation,
)
from hipe_analysis import (
    generate_example_dosy_data,
    auto_fit_dosy,
    plot_dosy_results,
    generate_example_cd_data,
    analyze_cd_spectrum,
    plot_cd_spectrum,
    generate_example_microscopy_timeline,
    plot_microscopy_timeline,
    parse_csv_two_columns,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="mAb StabilityAI",
    page_icon="\U0001f9ea",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .risk-high {
        background-color: #ff4b4b; color: white;
        padding: 0.5rem 1.5rem; border-radius: 0.5rem;
        font-size: 1.5rem; font-weight: bold;
        display: inline-block; margin: 0.5rem 0;
    }
    .risk-medium {
        background-color: #ffa600; color: white;
        padding: 0.5rem 1.5rem; border-radius: 0.5rem;
        font-size: 1.5rem; font-weight: bold;
        display: inline-block; margin: 0.5rem 0;
    }
    .risk-low {
        background-color: #00cc66; color: white;
        padding: 0.5rem 1.5rem; border-radius: 0.5rem;
        font-size: 1.5rem; font-weight: bold;
        display: inline-block; margin: 0.5rem 0;
    }
    .verdict-box {
        background-color: #1e2130; border-left: 4px solid #4da6ff;
        padding: 1rem; border-radius: 0 0.5rem 0.5rem 0;
        font-size: 1.2rem; margin: 1rem 0; color: white;
    }
    .metric-card {
        background-color: #1e2130; border-radius: 0.5rem;
        padding: 1rem; text-align: center; margin: 0.5rem 0;
    }
    .footer {
        text-align: center; padding: 2rem 0 1rem 0;
        color: #888; font-size: 0.85rem;
        border-top: 1px solid #333; margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
def parse_risk_level(text):
    match = re.search(r"RISK_LEVEL:\s*(High|Medium|Low)", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    for level in ["High", "Medium", "Low"]:
        if f"**{level}**" in text:
            return level
    return None


def parse_verdict(text):
    match = re.search(r"VERDICT:\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    return None


def _cache_key(*args):
    """Create a hash key from arbitrary arguments for response caching."""
    return hashlib.md5(str(args).encode()).hexdigest()


def plot_risk_gauge(risk_level):
    """Create a semicircular gauge chart for the stability risk level."""
    value_map = {"Low": 15, "Medium": 50, "High": 85}
    color_map = {"Low": "#00cc66", "Medium": "#ffa600", "High": "#ff4b4b"}
    val = value_map.get(risk_level, 50)
    clr = color_map.get(risk_level, "#ffa600")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val,
        number={"suffix": "", "font": {"size": 1, "color": "rgba(0,0,0,0)"}},
        title={"text": f"<b>{risk_level} Risk</b>", "font": {"size": 22, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)",
                     "tickvals": [], "showticklabels": False},
            "bar": {"color": clr, "thickness": 0.3},
            "bgcolor": "rgba(30,33,48,0.5)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "rgba(0,204,102,0.15)"},
                {"range": [33, 66], "color": "rgba(255,166,0,0.15)"},
                {"range": [66, 100], "color": "rgba(255,75,75,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": val,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


# --- Formulation scoring heuristics ---
def _score_ph(ph):
    """Score pH optimality for mAb stability (optimal 5.5-6.5)."""
    optimum = 6.0
    distance = abs(ph - optimum)
    if distance <= 0.5:
        return 10
    elif distance <= 1.0:
        return 8
    elif distance <= 1.5:
        return 6
    elif distance <= 2.0:
        return 4
    return 2


def _score_buffer(buffer_name):
    """Score buffer suitability for mAb stability."""
    b = buffer_name.lower()
    if "histidine" in b:
        return 10
    elif "citrate" in b:
        return 8
    elif "acetate" in b:
        return 7
    elif "succinate" in b:
        return 7
    elif "phosphate" in b:
        return 6
    elif "tris" in b:
        return 4
    return 5


def _score_protein(protein_name):
    """Score protein agent effectiveness as HIPE stabilizer."""
    p = protein_name.lower()
    if "bsa" in p or "bovine serum" in p:
        return 10
    elif "igg" in p or "immunoglobulin" in p:
        return 9
    elif "whey" in p or "wpi" in p:
        return 8
    elif "lactoglobulin" in p:
        return 8
    elif "lactalbumin" in p:
        return 7
    elif "casein" in p:
        return 7
    elif "lysozyme" in p:
        return 6
    elif "ovalbumin" in p:
        return 6
    elif "gelatin" in p:
        return 5
    elif "soy" in p:
        return 5
    return 5


def _score_temperature(temp):
    """Score thermal stability (lower storage temp = better)."""
    if temp <= 5:
        return 10
    elif temp <= 15:
        return 8
    elif temp <= 25:
        return 6
    elif temp <= 37:
        return 4
    return 2


def _score_concentration(conc):
    """Score mAb concentration optimality (moderate = best)."""
    if 10 <= conc <= 50:
        return 10
    elif 50 < conc <= 100:
        return 7
    elif 100 < conc <= 150:
        return 5
    elif conc > 150:
        return 3
    return 6  # very low


def score_formulation(ph, buffer, protein, temperature, concentration):
    """Return dict of scores (0-10) for radar chart axes."""
    return {
        "pH Optimality": _score_ph(ph),
        "Buffer Suitability": _score_buffer(buffer),
        "Protein Agent": _score_protein(protein),
        "Thermal Stability": _score_temperature(temperature),
        "Concentration": _score_concentration(concentration),
    }


def plot_formulation_radar(scores_a, scores_b):
    """Create a radar/spider chart comparing two formulations."""
    categories = list(scores_a.keys())
    vals_a = [scores_a[c] for c in categories]
    vals_b = [scores_b[c] for c in categories]
    # Close the polygon
    categories += [categories[0]]
    vals_a += [vals_a[0]]
    vals_b += [vals_b[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a, theta=categories, fill="toself",
        name="Formulation A",
        line=dict(color="#4da6ff", width=2),
        fillcolor="rgba(77,166,255,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b, theta=categories, fill="toself",
        name="Formulation B",
        line=dict(color="#ff6b6b", width=2),
        fillcolor="rgba(255,107,107,0.2)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 10],
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="rgba(255,255,255,0.5)", size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.15)",
                tickfont=dict(color="white", size=12),
            ),
        ),
        showlegend=True,
        legend=dict(font=dict(color="white", size=12), x=0.85, y=1.15),
        height=400,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def stream_to_container(generator):
    """Stream text from a generator into a Streamlit container, return full text.

    If the stream breaks mid-response (network error, safety filter, etc.),
    shows what was received and raises the error so callers can display it.
    """
    container = st.empty()
    full_text = ""
    try:
        for chunk in generator:
            full_text += chunk
            container.markdown(full_text + "\u2588")
    except Exception as e:
        # Show partial content if we got anything before the error
        if full_text:
            container.markdown(full_text + "\n\n---\n*Response interrupted.*")
        raise e
    # Clear the streaming container — the persistent display section below
    # will render the final result (avoids showing it twice).
    container.empty()
    if not full_text:
        st.warning("Model returned an empty response. Try again or switch models.")
    return full_text


# --- Shared Options ---
BUFFER_OPTIONS = [
    "Histidine (20 mM)", "Phosphate (10 mM)", "Citrate (10 mM)",
    "Acetate (10 mM)", "Tris (25 mM)", "Succinate (10 mM)",
]

PROTEIN_OPTIONS = [
    "BSA (Bovine Serum Albumin)",
    "Lysozyme",
    "\u03b2-Lactoglobulin",
    "Ovalbumin",
    "Whey Protein Isolate (WPI)",
    "\u03b1-Lactalbumin",
    "Sodium Caseinate",
    "Gelatin",
    "Soy Protein Isolate",
    "IgG (Immunoglobulin G)",
]

INTERFACE_OPTIONS = [
    "Air-Liquid Interface", "Silicone Oil-Water Interface",
    "Glass Container Surface", "Stainless Steel Surface",
    "Ice-Liquid Interface (Freeze-Thaw)",
    "Agitation / Mechanical Stress", "Pumping / Shear Stress",
]

MECHANISM_INTERFACE_OPTIONS = [
    "Air-Liquid Interface", "Silicone Oil-Water Interface",
    "Glass Container Surface", "Stainless Steel Surface",
    "Ice-Liquid Interface (Freeze-Thaw)",
    "Rubber Stopper Surface", "Plastic (COP/COC) Container Surface",
]

# --- Presets ---
STABILITY_PRESETS = {
    "Custom (enter manually)": None,
    "BSA-stabilized HIPE (standard)": {
        "concentration": 50.0, "ph": 7.0, "buffer": 1, "protein": 0,
        "protein_concentration": "2.0% (w/v)", "temperature": 25.0,
        "interface": 0,
    },
    "Lysozyme-stabilized HIPE": {
        "concentration": 100.0, "ph": 6.0, "buffer": 0, "protein": 1,
        "protein_concentration": "1.0% (w/v)", "temperature": 25.0,
        "interface": 1,
    },
    "High-concentration stress test": {
        "concentration": 200.0, "ph": 7.4, "buffer": 1, "protein": 4,
        "protein_concentration": "5.0% (w/v)", "temperature": 40.0,
        "interface": 4,
    },
}

COMPARISON_PRESETS = {
    "Custom (enter manually)": None,
    "BSA vs. Lysozyme HIPE": {
        "conc_a": 50.0, "ph_a": 7.0, "buf_a": 1, "prot_a": 0,
        "pconc_a": "2.0% (w/v)", "temp_a": 25.0,
        "conc_b": 50.0, "ph_b": 6.0, "buf_b": 0, "prot_b": 1,
        "pconc_b": "1.0% (w/v)", "temp_b": 25.0,
    },
}

LITERATURE_TOPICS = [
    "Protein-stabilized HIPE (High Internal Phase Emulsions) as model systems for mAb stability",
    "NMR DOSY diffusion analysis for characterizing protein aggregation",
    "Circular dichroism for monitoring mAb secondary structure changes",
    "Protein adsorption and conformational changes at oil-water interfaces",
    "BSA and lysozyme as model proteins for emulsion stability studies",
    "Freeze-thaw stress effects on monoclonal antibody structure",
    "Whey protein and \u03b2-lactoglobulin stabilized emulsions for biopharmaceutical modeling",
]


# --- API Initialization ---
if not initialize_client():
    st.error(
        "Google API Key not found. Add `GOOGLE_API_KEY` to "
        "`.streamlit/secrets.toml` or set it as an environment variable."
    )
    st.stop()

# Session state for result persistence and response cache
for key in [
    "stability_result", "mechanism_result", "comparison_result",
    "literature_result", "nmr_result", "cd_result", "microscopy_result",
]:
    if key not in st.session_state:
        st.session_state[key] = None

if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}


# --- Sidebar ---
with st.sidebar:
    st.title("\U0001f9ea mAb StabilityAI")
    st.caption("Monoclonal Antibody Stability Prediction")

    st.divider()

    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-3-pro-preview", "gemini-3-flash-preview",
         "gemini-2.5-pro", "gemini-2.5-flash"],
        help="Gemini 3 Pro: deepest reasoning. Flash: faster responses.",
    )

    st.divider()

    st.markdown("### About")
    st.markdown(
        "mAb StabilityAI combines Google Gemini 3 AI with experimental "
        "analysis pipelines (NMR DOSY, Circular Dichroism, Microscopy) to "
        "predict and validate monoclonal antibody structural stability. "
        "Protein-stabilized HIPEs serve as cost-effective model systems for mAb interface studies."
    )

    st.divider()

    st.markdown("### Quick Reference")
    st.markdown("""
**Common Buffers:** Histidine, Phosphate, Citrate, Acetate, Tris

**Common Protein Agents:** BSA, Lysozyme, \u03b2-Lactoglobulin, WPI, Casein

**Typical Ranges:**
- pH: 5.0 - 7.4
- Protein concentration: 0.5 - 5.0% (w/v)
- Temperature: 5\u00b0C to 40\u00b0C
    """)


# --- Main Content ---
st.title("mAb StabilityAI")
st.markdown(
    "*AI-Powered Monoclonal Antibody Stability Prediction & "
    "Experimental Validation \u2014 Powered by Google Gemini 3*"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "\U0001f50d Stability Analysis",
    "\U0001f9ec Mechanism Explorer",
    "\u2696\ufe0f Formulation Comparison",
    "\U0001f4da Literature & Research",
    "\U0001f52c HIPE Experimental Pipeline",
])


# ============================================================
# Tab 1: Stability Analysis
# ============================================================
with tab1:
    st.header("Stability Risk Analysis")
    st.markdown("Enter formulation parameters to assess structural stability risk.")

    preset_choice = st.selectbox(
        "Load Example Preset", list(STABILITY_PRESETS.keys()),
        key="stability_preset",
    )
    preset = STABILITY_PRESETS[preset_choice]

    col1, col2 = st.columns(2)
    with col1:
        concentration = st.number_input(
            "mAb Concentration (mg/mL)", min_value=0.1, max_value=300.0,
            value=preset["concentration"] if preset else 50.0, step=5.0,
        )
        ph = st.number_input(
            "pH", min_value=3.0, max_value=9.0,
            value=preset["ph"] if preset else 6.0, step=0.1,
        )
        buffer = st.selectbox(
            "Buffer System", BUFFER_OPTIONS,
            index=preset["buffer"] if preset else 0,
        )
    with col2:
        protein = st.selectbox(
            "Protein Agent", PROTEIN_OPTIONS,
            index=preset["protein"] if preset else 0,
        )
        protein_concentration = st.text_input(
            "Protein Concentration",
            value=preset["protein_concentration"] if preset else "2.0% (w/v)",
        )
        temperature = st.number_input(
            "Storage Temperature (\u00b0C)", min_value=-80.0, max_value=60.0,
            value=preset["temperature"] if preset else 25.0, step=5.0,
        )

    interface_type = st.selectbox(
        "Interface / Stress Type", INTERFACE_OPTIONS,
        index=preset["interface"] if preset else 0,
    )

    if st.button("Analyze Stability", type="primary", use_container_width=True):
        if not protein_concentration.strip():
            st.warning("Please enter a protein concentration (e.g., 2.0% w/v).")
        else:
            cache_key = _cache_key(
                "stability", model_name, concentration, ph, buffer, protein,
                protein_concentration.strip(), temperature, interface_type,
            )
            cached = st.session_state.response_cache.get(cache_key)
            if cached:
                st.toast("Loaded from cache (same parameters).")
                st.session_state.stability_result = cached
            else:
                try:
                    result = stream_to_container(stream_stability(
                        model_name, concentration, ph, buffer, protein,
                        protein_concentration.strip(), temperature, interface_type,
                    ))
                    st.session_state.stability_result = result
                    if result:
                        st.session_state.response_cache[cache_key] = result
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.info("Try a Flash model if you're hitting rate limits.")

    if st.session_state.stability_result:
        result = st.session_state.stability_result
        risk_level = parse_risk_level(result)
        if risk_level:
            gauge_col, badge_col = st.columns([1, 1])
            with gauge_col:
                st.plotly_chart(plot_risk_gauge(risk_level), use_container_width=True)
            with badge_col:
                st.markdown(
                    f'<div class="risk-{risk_level.lower()}" '
                    f'style="margin-top:2rem;text-align:center;">'
                    f'Risk Level: {risk_level}</div>',
                    unsafe_allow_html=True,
                )
                scores = score_formulation(ph, buffer, protein, temperature, concentration)
                for axis, score in scores.items():
                    bar_pct = score * 10
                    bar_color = "#00cc66" if score >= 7 else "#ffa600" if score >= 4 else "#ff4b4b"
                    st.markdown(
                        f'<div style="margin:4px 0;font-size:0.85rem;">'
                        f'<span style="color:#ccc;">{axis}</span> '
                        f'<span style="color:white;font-weight:bold;">{score}/10</span>'
                        f'<div style="background:#1e2130;border-radius:4px;height:8px;margin-top:2px;">'
                        f'<div style="background:{bar_color};width:{bar_pct}%;height:8px;'
                        f'border-radius:4px;"></div></div></div>',
                        unsafe_allow_html=True,
                    )
        st.markdown("---")
        st.markdown(result)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Analysis", data=result,
                file_name="mab_stability_analysis.md", mime="text/markdown",
            )
        with c2:
            if st.button("Clear Result", key="clear_stability"):
                st.session_state.stability_result = None
                st.rerun()


# ============================================================
# Tab 2: Mechanism Explorer
# ============================================================
with tab2:
    st.header("Mechanism Explorer")
    st.markdown("Learn how different interfaces cause structural instability in mAbs.")

    mechanism_interface = st.selectbox(
        "Select Interface Type to Explore",
        MECHANISM_INTERFACE_OPTIONS, key="mechanism_interface",
    )

    if st.button("Explain Mechanism", type="primary", use_container_width=True):
        cache_key = _cache_key("mechanism", model_name, mechanism_interface)
        cached = st.session_state.response_cache.get(cache_key)
        if cached:
            st.toast("Loaded from cache.")
            st.session_state.mechanism_result = cached
        else:
            try:
                result = stream_to_container(stream_mechanism(model_name, mechanism_interface))
                st.session_state.mechanism_result = result
                if result:
                    st.session_state.response_cache[cache_key] = result
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.session_state.mechanism_result:
        st.markdown("---")
        st.markdown(st.session_state.mechanism_result)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Explanation", data=st.session_state.mechanism_result,
                file_name="mab_mechanism_explanation.md", mime="text/markdown",
            )
        with c2:
            if st.button("Clear Result", key="clear_mechanism"):
                st.session_state.mechanism_result = None
                st.rerun()


# ============================================================
# Tab 3: Formulation Comparison
# ============================================================
with tab3:
    st.header("Formulation Comparison")
    st.markdown("Compare two formulations side-by-side.")

    comp_preset_choice = st.selectbox(
        "Load Example Preset", list(COMPARISON_PRESETS.keys()),
        key="comparison_preset",
    )
    cp = COMPARISON_PRESETS[comp_preset_choice]

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Formulation A")
        conc_a = st.number_input("Concentration (mg/mL)", value=cp["conc_a"] if cp else 50.0,
                                 step=5.0, key="conc_a", min_value=0.1, max_value=300.0)
        ph_a = st.number_input("pH", value=cp["ph_a"] if cp else 7.0,
                               step=0.1, key="ph_a", min_value=3.0, max_value=9.0)
        buffer_a = st.selectbox("Buffer", BUFFER_OPTIONS, key="buf_a",
                                index=cp["buf_a"] if cp else 0)
        protein_a = st.selectbox("Protein Agent", PROTEIN_OPTIONS, key="prot_a",
                                 index=cp["prot_a"] if cp else 0)
        protein_conc_a = st.text_input("Protein Conc.",
                                       value=cp["pconc_a"] if cp else "2.0% (w/v)", key="pconc_a")
        temp_a = st.number_input("Temperature (\u00b0C)", value=cp["temp_a"] if cp else 25.0,
                                 step=5.0, key="temp_a", min_value=-80.0, max_value=60.0)
    with col_b:
        st.subheader("Formulation B")
        conc_b = st.number_input("Concentration (mg/mL)", value=cp["conc_b"] if cp else 100.0,
                                 step=5.0, key="conc_b", min_value=0.1, max_value=300.0)
        ph_b = st.number_input("pH", value=cp["ph_b"] if cp else 7.0,
                               step=0.1, key="ph_b", min_value=3.0, max_value=9.0)
        buffer_b = st.selectbox("Buffer", BUFFER_OPTIONS, key="buf_b",
                                index=cp["buf_b"] if cp else 1)
        protein_b = st.selectbox("Protein Agent", PROTEIN_OPTIONS, key="prot_b",
                                 index=cp["prot_b"] if cp else 1)
        protein_conc_b = st.text_input("Protein Conc.",
                                       value=cp["pconc_b"] if cp else "1.0% (w/v)", key="pconc_b")
        temp_b = st.number_input("Temperature (\u00b0C)", value=cp["temp_b"] if cp else 25.0,
                                 step=5.0, key="temp_b", min_value=-80.0, max_value=60.0)

    if st.button("Compare Formulations", type="primary", use_container_width=True):
        if not protein_conc_a.strip() or not protein_conc_b.strip():
            st.warning("Please enter protein concentrations for both formulations.")
        else:
            cache_key = _cache_key(
                "comparison", model_name,
                conc_a, ph_a, buffer_a, protein_a, protein_conc_a.strip(), temp_a,
                conc_b, ph_b, buffer_b, protein_b, protein_conc_b.strip(), temp_b,
            )
            cached = st.session_state.response_cache.get(cache_key)
            if cached:
                st.toast("Loaded from cache.")
                st.session_state.comparison_result = cached
            else:
                try:
                    result = stream_to_container(stream_comparison(
                        model_name,
                        conc_a, ph_a, buffer_a, protein_a, protein_conc_a.strip(), temp_a,
                        conc_b, ph_b, buffer_b, protein_b, protein_conc_b.strip(), temp_b,
                    ))
                    st.session_state.comparison_result = result
                    if result:
                        st.session_state.response_cache[cache_key] = result
                except Exception as e:
                    st.error(f"Comparison failed: {e}")

    if st.session_state.comparison_result:
        result = st.session_state.comparison_result
        verdict = parse_verdict(result)

        # --- Visual comparison: radar chart + verdict ---
        scores_a = score_formulation(ph_a, buffer_a, protein_a, temp_a, conc_a)
        scores_b = score_formulation(ph_b, buffer_b, protein_b, temp_b, conc_b)
        total_a = sum(scores_a.values())
        total_b = sum(scores_b.values())

        radar_col, verdict_col = st.columns([3, 2])
        with radar_col:
            st.plotly_chart(plot_formulation_radar(scores_a, scores_b),
                           use_container_width=True)
        with verdict_col:
            if verdict:
                st.markdown(
                    f'<div class="verdict-box">{verdict}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:0.9rem;color:#aaa;">Composite Score</div>'
                f'<div style="font-size:1.8rem;font-weight:bold;color:#4da6ff;">'
                f'A: {total_a}/50</div>'
                f'<div style="font-size:1.8rem;font-weight:bold;color:#ff6b6b;">'
                f'B: {total_b}/50</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(result)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Comparison", data=result,
                file_name="mab_formulation_comparison.md", mime="text/markdown",
            )
        with c2:
            if st.button("Clear Result", key="clear_comparison"):
                st.session_state.comparison_result = None
                st.rerun()


# ============================================================
# Tab 4: Literature & Research
# ============================================================
with tab4:
    st.header("Literature & Research")
    st.markdown(
        "Explore published research on mAb stability, HIPE model systems, "
        "and analytical techniques. Powered by Gemini 3's scientific knowledge."
    )

    topic_choice = st.selectbox(
        "Select a research topic", LITERATURE_TOPICS, key="lit_topic",
    )
    custom_topic = st.text_input(
        "Or enter a custom topic",
        placeholder="e.g., Effect of pH on IgG1 aggregation at air-water interfaces",
        key="lit_custom",
    )
    topic = custom_topic if custom_topic else topic_choice

    if st.button("Search Literature", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Please select or enter a research topic.")
        else:
            cache_key = _cache_key("literature", model_name, topic.strip())
            cached = st.session_state.response_cache.get(cache_key)
            if cached:
                st.toast("Loaded from cache.")
                st.session_state.literature_result = cached
            else:
                try:
                    result = stream_to_container(stream_literature(model_name, topic.strip()))
                    st.session_state.literature_result = result
                    if result:
                        st.session_state.response_cache[cache_key] = result
                except Exception as e:
                    st.error(f"Literature search failed: {e}")

    if st.session_state.literature_result:
        st.markdown("---")
        st.markdown(st.session_state.literature_result)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Review", data=st.session_state.literature_result,
                file_name="mab_literature_review.md", mime="text/markdown",
            )
        with c2:
            if st.button("Clear Result", key="clear_literature"):
                st.session_state.literature_result = None
                st.rerun()


# ============================================================
# Tab 5: HIPE Experimental Pipeline
# ============================================================
with tab5:
    st.header("HIPE Experimental Analysis Pipeline")
    st.markdown(
        "Analyze experimental data from protein-stabilized HIPE (High Internal Phase Emulsion) studies. "
        "Upload your own data or use example datasets for demonstration. "
        "Gemini 3 interprets results and translates findings to mAb stability predictions."
    )

    st.info(
        "**Why HIPEs?** Monoclonal antibodies are expensive to work with directly. "
        "Protein-stabilized HIPEs provide a cost-effective model system for studying "
        "structural stability at interfaces. Proteins act as the emulsifier/stabilizing agent, "
        "mimicking mAb adsorption behavior. Results can then be correlated to predict mAb behavior.",
        icon="\U0001f4a1",
    )

    hipe_tab1, hipe_tab2, hipe_tab3 = st.tabs([
        "\U0001f9f2 NMR DOSY Analysis",
        "\U0001f4c9 Circular Dichroism",
        "\U0001f52c Microscopy Timeline",
    ])

    # ----------------------------------------------------------
    # HIPE Sub-tab 1: NMR DOSY
    # ----------------------------------------------------------
    with hipe_tab1:
        st.subheader("NMR DOSY Diffusion Analysis")
        st.markdown(
            "Upload or load DOSY data and the best-fit model is selected automatically. "
            "Mono-exponential (free diffusion) or bi-exponential (restricted diffusion) "
            "is chosen using AIC model comparison."
        )

        nmr_data_source = st.radio(
            "Data Source", ["Load Example Data", "Upload CSV"],
            key="nmr_source", horizontal=True,
        )

        gradients = intensities = None

        if nmr_data_source == "Load Example Data":
            nmr_regime = st.selectbox(
                "Example Data Regime",
                ["low_concentration", "high_concentration"],
                format_func=lambda x: {
                    "low_concentration": "Low concentration (free diffusion, D \u2248 8.5\u00d710\u207b\u00b9\u00b9 m\u00b2/s)",
                    "high_concentration": "High concentration (restricted diffusion, Df \u2248 6\u00d710\u207b\u00b9\u2070, Ds \u2248 5\u00d710\u207b\u00b9\u00b9)",
                }[x],
                key="nmr_regime",
            )
            if st.button("Load Example NMR Data", key="load_nmr_example"):
                gradients, intensities = generate_example_dosy_data(regime=nmr_regime)
                st.session_state["nmr_gradients"] = gradients
                st.session_state["nmr_intensities"] = intensities

            if "nmr_gradients" in st.session_state:
                gradients = st.session_state["nmr_gradients"]
                intensities = st.session_state["nmr_intensities"]
        else:
            uploaded_nmr = st.file_uploader(
                "Upload CSV (column 1: gradient strength T/m, column 2: intensity)",
                type=["csv"], key="nmr_upload",
            )
            if uploaded_nmr:
                gradients, intensities = parse_csv_two_columns(uploaded_nmr)
                if len(gradients) == 0:
                    st.error("No valid numeric data found in the CSV. Check the file format.")
                    gradients = intensities = None
                elif len(gradients) < 4:
                    st.warning(f"Only {len(gradients)} data points found — at least 4 are needed for a reliable fit.")

        if gradients is not None and len(gradients) > 3:
            try:
                # --- Automatic model selection via AIC ---
                fit = auto_fit_dosy(gradients, intensities)

                fig = plot_dosy_results(gradients, intensities, fit)
                st.plotly_chart(fig, use_container_width=True)

                # Show why this model was selected
                st.info(fit.get("selection_reason", ""), icon="\U0001f9e0")

                # --- Display metrics based on model type ---
                if fit["model"] == "bi-exponential":
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Df (fast)", f"{fit['Df']:.2e} m\u00b2/s")
                    mc2.metric("Ds (slow)", f"{fit['Ds']:.2e} m\u00b2/s")
                    mc3.metric("\u03be (fast fraction)", f"{fit['xi']:.3f}")
                    mc4.metric("R\u00b2 (Fit Quality)", f"{fit['r_squared']:.4f}")

                    ratio = fit['Df'] / fit['Ds'] if fit['Ds'] > 0 else float('inf')
                    st.caption(
                        f"**Bi-exponential fit:** Df/Ds ratio = {ratio:.1f}x | "
                        f"Data points: {len(gradients)}"
                    )
                else:
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Diffusion Coefficient", f"{fit['D']:.2e} m\u00b2/s")
                    mc2.metric("R\u00b2 (Fit Quality)", f"{fit['r_squared']:.4f}")
                    mc3.metric("Data Points", f"{len(gradients)}")

                if st.button("Interpret with Gemini 3", key="interpret_nmr",
                             type="primary", use_container_width=True):
                    try:
                        result = stream_to_container(stream_nmr_interpretation(
                            model_name, fit, len(gradients),
                        ))
                        st.session_state.nmr_result = result
                    except Exception as e:
                        st.error(f"Interpretation failed: {e}")

                if st.session_state.nmr_result:
                    st.markdown("---")
                    st.markdown(st.session_state.nmr_result)
                    st.download_button(
                        "Download NMR Analysis", data=st.session_state.nmr_result,
                        file_name="nmr_dosy_analysis.md", mime="text/markdown",
                        key="dl_nmr",
                    )
            except Exception as e:
                st.error(f"Fitting failed: {e}. Check your data format.")

    # ----------------------------------------------------------
    # HIPE Sub-tab 2: Circular Dichroism
    # ----------------------------------------------------------
    with hipe_tab2:
        st.subheader("Circular Dichroism Spectroscopy")
        st.markdown(
            "Analyze secondary structure from CD spectra. Characteristic bands indicate "
            "\u03b1-helix, \u03b2-sheet, or random coil content."
        )

        cd_data_source = st.radio(
            "Data Source", ["Load Example Data", "Upload CSV"],
            key="cd_source", horizontal=True,
        )

        wavelengths = ellipticity = None

        if cd_data_source == "Load Example Data":
            cd_structure = st.selectbox(
                "Example Structure Type",
                ["beta_sheet", "alpha_helix", "unfolded"],
                format_func=lambda x: {
                    "beta_sheet": "Beta-sheet (typical mAb IgG)",
                    "alpha_helix": "Alpha-helical protein",
                    "unfolded": "Unfolded / denatured",
                }[x],
                key="cd_example_type",
            )
            if st.button("Load Example CD Data", key="load_cd_example"):
                wavelengths, ellipticity = generate_example_cd_data(cd_structure)
                st.session_state["cd_wavelengths"] = wavelengths
                st.session_state["cd_ellipticity"] = ellipticity

            if "cd_wavelengths" in st.session_state:
                wavelengths = st.session_state["cd_wavelengths"]
                ellipticity = st.session_state["cd_ellipticity"]
        else:
            uploaded_cd = st.file_uploader(
                "Upload CSV (column 1: wavelength nm, column 2: ellipticity)",
                type=["csv"], key="cd_upload",
            )
            if uploaded_cd:
                wavelengths, ellipticity = parse_csv_two_columns(uploaded_cd)
                if len(wavelengths) == 0:
                    st.error("No valid numeric data found in the CSV. Check the file format.")
                    wavelengths = ellipticity = None
                elif len(wavelengths) < 6:
                    st.warning(f"Only {len(wavelengths)} data points found — at least 6 are recommended for CD analysis.")

        if wavelengths is not None and len(wavelengths) > 5:
            analysis = analyze_cd_spectrum(wavelengths, ellipticity)
            fig = plot_cd_spectrum(wavelengths, ellipticity, analysis)
            st.plotly_chart(fig, use_container_width=True)

            mc1, mc2 = st.columns(2)
            mc1.metric("Min Band", f"{analysis['min_wavelength']:.0f} nm")
            mc2.metric("Max Band", f"{analysis['max_wavelength']:.0f} nm")
            st.info(f"**Estimated structure:** {analysis['estimated_structure']}")

            if st.button("Interpret with Gemini 3", key="interpret_cd",
                         type="primary", use_container_width=True):
                try:
                    wl_range = f"{wavelengths.min():.0f}-{wavelengths.max():.0f}"
                    result = stream_to_container(stream_cd_interpretation(
                        model_name, wl_range, analysis["estimated_structure"],
                        "HIPE emulsion / protein system",
                        f"{analysis['min_wavelength']:.0f}",
                        f"{analysis['max_wavelength']:.0f}",
                    ))
                    st.session_state.cd_result = result
                except Exception as e:
                    st.error(f"Interpretation failed: {e}")

            if st.session_state.cd_result:
                st.markdown("---")
                st.markdown(st.session_state.cd_result)
                st.download_button(
                    "Download CD Analysis", data=st.session_state.cd_result,
                    file_name="cd_spectrum_analysis.md", mime="text/markdown",
                    key="dl_cd",
                )

    # ----------------------------------------------------------
    # HIPE Sub-tab 3: Microscopy Timeline
    # ----------------------------------------------------------
    with hipe_tab3:
        st.subheader("Microscopy Stability Timeline")
        st.markdown(
            "Track HIPE droplet size and morphology over time to assess physical stability. "
            "Changes in droplet size indicate coalescence, Ostwald ripening, or degradation."
        )

        micro_data_source = st.radio(
            "Data Source", ["Load Example Data", "Enter Observations Manually"],
            key="micro_source", horizontal=True,
        )

        if micro_data_source == "Load Example Data":
            st.caption("Example: 14-day HIPE stability study with droplet size tracking")
            if st.button("Load Example Microscopy Data", key="load_micro_example"):
                days, sizes, std_devs = generate_example_microscopy_timeline()
                st.session_state["micro_days"] = days
                st.session_state["micro_sizes"] = sizes
                st.session_state["micro_stds"] = std_devs

            if "micro_days" in st.session_state:
                days = st.session_state["micro_days"]
                sizes = st.session_state["micro_sizes"]
                std_devs = st.session_state["micro_stds"]

                fig = plot_microscopy_timeline(days, sizes, std_devs)
                st.plotly_chart(fig, use_container_width=True)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Initial Size", f"{sizes[0]:.1f} \u00b5m")
                mc2.metric("Final Size (Day 14)", f"{sizes[-1]:.1f} \u00b5m")
                change = ((sizes[-1] - sizes[0]) / sizes[0]) * 100
                mc3.metric("Size Change", f"{change:+.1f}%")

                observations = (
                    f"Initial mean droplet diameter: {sizes[0]:.1f} \u00b5m. "
                    f"Final mean droplet diameter at day 14: {sizes[-1]:.1f} \u00b5m. "
                    f"Size change: {change:+.1f}%. "
                    f"Standard deviation increased from {std_devs[0]:.1f} to {std_devs[-1]:.1f} \u00b5m. "
                    "Minor Ostwald ripening observed. No significant coalescence or phase separation."
                )

                if st.button("Interpret with Gemini 3", key="interpret_micro",
                             type="primary", use_container_width=True):
                    try:
                        result = stream_to_container(stream_microscopy_interpretation(
                            model_name, "HIPE (oil-in-water high internal phase emulsion)",
                            "14 days", observations, "40x optical",
                        ))
                        st.session_state.microscopy_result = result
                    except Exception as e:
                        st.error(f"Interpretation failed: {e}")

                if st.session_state.microscopy_result:
                    st.markdown("---")
                    st.markdown(st.session_state.microscopy_result)
                    st.download_button(
                        "Download Microscopy Analysis",
                        data=st.session_state.microscopy_result,
                        file_name="microscopy_analysis.md", mime="text/markdown",
                        key="dl_micro",
                    )
        else:
            st.markdown("Enter your microscopy observations manually:")
            manual_sample = st.text_input("Sample Type", value="HIPE emulsion", key="micro_sample")
            manual_period = st.text_input("Observation Period", value="14 days", key="micro_period")
            manual_mag = st.text_input("Magnification", value="40x optical", key="micro_mag")
            manual_obs = st.text_area(
                "Key Observations",
                value="Describe droplet size changes, coalescence, phase separation, etc.",
                height=120, key="micro_obs",
            )
            if st.button("Interpret Observations", key="interpret_micro_manual",
                         type="primary", use_container_width=True):
                try:
                    result = stream_to_container(stream_microscopy_interpretation(
                        model_name, manual_sample, manual_period,
                        manual_obs, manual_mag,
                    ))
                    st.session_state.microscopy_result = result
                except Exception as e:
                    st.error(f"Interpretation failed: {e}")

            if st.session_state.microscopy_result:
                st.markdown("---")
                st.markdown(st.session_state.microscopy_result)


# --- Footer ---
st.markdown(
    '<div class="footer">'
    "mAb StabilityAI &mdash; Built for the Gemini 3 Hackathon 2026<br>"
    "Powered by Google Gemini 3 &bull; Built with Streamlit &bull; "
    "HIPE Experimental Validation Pipeline"
    "</div>",
    unsafe_allow_html=True,
)

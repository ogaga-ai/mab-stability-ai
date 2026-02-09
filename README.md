# mAb StabilityAI

**AI-Powered Monoclonal Antibody Stability Prediction & Experimental Validation Platform**

Built for the [Gemini 3 Hackathon 2026](https://gemini3.devpost.com) | Powered by Google Gemini 3 Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mab-stability-ai-gqqkgltxnv34zcwwwd5rapp.streamlit.app/)

---

## Inspiration

In 2020, COVID-19 wiped out approximately 7.1 million people worldwide. Pharmaceutical companies raced to develop vaccines, but for patients with weaker immune systems, vaccines alone weren't enough — their bodies couldn't generate antibodies fast enough. Researchers turned to **monoclonal antibodies (mAbs)** — laboratory-designed antibodies that could immediately protect individuals upon administration. Eli Lilly's bamlanivimab and Genentech's Actemra were among the first mAb-based treatments deployed in the US.

This experience inspired me to pursue a PhD in Chemical Engineering, focused on understanding **why mAbs lose their shape during development and delivery** — a problem called conformational instability that reduces drug efficacy. During my research, I spent long hours manually analyzing formulation parameters, interpreting NMR diffusion data, and trying to predict stability outcomes. It was an iterative process that could have been solved using AI, but I didn't have a way to do it efficiently — until now.

mAb StabilityAI combines my PhD research with Google Gemini 3 Pro to automate and accelerate this entire workflow.

## What It Does

mAb StabilityAI is a five-module scientific platform:

1. **Stability Risk Analysis** — Input formulation parameters (concentration, pH, buffer, protein stabilizer, temperature, interface type) and receive an AI-generated risk assessment with a visual gauge chart, formulation score breakdown, and actionable recommendations.

2. **Mechanism Explorer** — Select any interface type and receive a detailed explanation of how that interface causes mAb instability — from molecular adsorption to aggregation pathways.

3. **Formulation Comparison** — Compare two formulations side-by-side with a radar chart overlay, composite scores, and an AI-powered verdict.

4. **Literature & Research** — Query Gemini 3 Pro's scientific knowledge for structured literature reviews on mAb stability topics.

5. **HIPE Experimental Pipeline** — The core engine. Upload NMR DOSY diffusion data, Circular Dichroism spectra, or microscopy observations. The app automatically fits mathematical models (mono-exponential or bi-exponential, selected via AIC), generates interactive visualizations, and uses Gemini 3 Pro to interpret results and translate HIPE findings into mAb stability predictions.

## The Science: Why HIPEs?

Monoclonal antibodies are expensive to work with directly. During my PhD, I discovered that **protein-stabilized High Internal Phase Emulsions (HIPEs)** — where proteins like BSA and lysozyme act as the emulsifier — behave similarly to mAbs at interfaces. This makes HIPEs a cost-effective model system for studying protein structural stability. If we can ensure HIPEs are stable under different experimental conditions, we can predict the same for mAbs.

This app is the first tool to codify that HIPE-to-mAb correlation into a working AI platform.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| AI Engine | Google Gemini 3 Pro (`google-generativeai` SDK) |
| Visualization | Plotly (gauge charts, radar plots, decay curves, CD spectra) |
| Computation | NumPy, SciPy (curve fitting, AIC model selection) |
| Deployment | Streamlit Community Cloud |

## Project Structure

```
├── app.py                 # Main Streamlit application (5 tabs)
├── gemini_client.py       # Gemini API abstraction with streaming & retry logic
├── prompts.py             # 8 structured prompt templates
├── hipe_analysis.py       # Scientific computation engine (fitting, analysis)
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Dark theme configuration
└── .gitignore
```

## Running Locally

```bash
# Clone the repository
git clone https://github.com/ogaga-ai/mab-stability-ai.git
cd mab-stability-ai

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
mkdir -p .streamlit
echo 'GOOGLE_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml

# Run the app
streamlit run app.py
```

## Key Technical Features

- **Automatic model selection via AIC** — Fits both mono-exponential and bi-exponential decay models and selects the best fit using the Akaike Information Criterion
- **Crash-proof streaming** — Exponential backoff retry with a `chunks_yielded` guard that prevents content duplication on mid-stream failures
- **Formulation scoring engine** — Heuristic scoring across 5 stability dimensions (pH, buffer, protein agent, temperature, concentration) grounded in biopharmaceutical best practices
- **Robust data handling** — Supports CSV uploads with BOM markers, multiple delimiters, latin-1 encoding, and automatic header detection

## References

- Scigliani, Grant & Mohammadigoushki (2023) — Bi-exponential diffusion model for HIPE systems

## Author

**Ogaga Scigliani** — M.S. Chemical Engineering

---

*Built for the Gemini 3 Hackathon 2026*

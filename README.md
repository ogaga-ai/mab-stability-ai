# mAb StabilityAI

**AI-Powered Monoclonal Antibody Stability Prediction & Experimental Validation Platform**

Built for the [Gemini 3 Hackathon 2026](https://gemini3.devpost.com) | Powered by Google Gemini 3 Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mab-stability-ai-gqqkgltxnv34zcwwwd5rapp.streamlit.app/)

---

## Inspiration

About 3 years ago, I started a PhD in Chemical Engineering with a dream to change the world. This decision was driven by a life-changing experience with COVID in 2020.
When COVID hit, the world became sick with a disease that wiped out about 7.1 million people, of which about 17.1% of those deaths happened in the US alone. The experience was overwhelming and the way forward — unclear.
Pharmaceutical companies raced against time to develop solutions. Pfizer and BioNTech developed Comirnaty, one of the first mRNA drugs authorized by the FDA for use in 2020, along with Paxlovid, an oral antiviral for patients already affected. These drugs teach your body to generate its own antibodies. But it wasn't enough — patients with weaker immune systems often died before their bodies got the chance to fight back.
Research scientist knew this wasn't sustainable, so they developed monoclonal antibodies (mAbs) that could immediately protect individuals upon administration. Eli Lilly developed bamlanivimab, the first mAb product used in the US. Genentech repurposed Actemra, originally an arthritis drug, to treat severely hospitalized COVID patients. In both cases, mAbs were the key to better outcomes but it came with its own challenges.

mAbs are proteins, and they often change shape during development and delivery. This conformational change reduces drug efficacy — sometimes rendering the treatment useless. In the early phase of my research, I spent long hours trying to understand why mAbs change shape and how to stabilize them. It was an iterative, manual process that could have been solved using AI, but I didn't have a way to do it efficiently — until today.

## What It Does

mAb StabilityAI is a five-module scientific platform:

1. **Stability Risk Analysis** — Input formulation parameters (concentration, pH, buffer, protein stabilizer, temperature, interface type) and receive an AI-generated risk assessment with a visual gauge chart, formulation score breakdown, and actionable recommendations.

2. **Mechanism Explorer** — Select any interface type and receive a detailed explanation of how that interface causes mAb instability — from molecular adsorption to aggregation pathways.

3. **Formulation Comparison** — Compare two formulations side-by-side with a radar chart overlay, composite scores, and an AI-powered verdict.

4. **Literature & Research** — Query Gemini 3 Pro's scientific knowledge for structured literature reviews on mAb stability topics.

5. **HIPE Experimental Pipeline** — The core engine of my research. Upload NMR DOSY diffusion data, Circular Dichroism spectra, or microscopy observations. The app automatically fits mathematical models (mono-exponential or bi-exponential, selected via AIC), generates interactive visualizations, and uses Gemini 3 Pro to interpret results and translate HIPE findings into mAb stability predictions.

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

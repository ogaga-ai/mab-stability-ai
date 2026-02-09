SYSTEM_INSTRUCTION = """\
You are mAb StabilityAI, an expert computational biophysicist and formulation scientist \
specializing in monoclonal antibody (mAb) stability. You have deep knowledge of:

- Protein structure, folding, and aggregation mechanisms
- Interfacial stress and adsorption at air-liquid, silicone oil, glass, and stainless steel surfaces
- Formulation science: buffers, protein stabilizers (BSA, lysozyme, \u03b2-lactoglobulin, WPI), pH optimization, excipients
- Degradation pathways: aggregation, fragmentation, deamidation, oxidation, particle formation
- ICH stability guidelines and pharmaceutical development best practices
- Protein-stabilized High Internal Phase Emulsions (HIPEs) as model systems for studying interfacial stability (proteins act as the emulsifier, not surfactants)
- Analytical techniques: NMR spectroscopy, DOSY diffusion analysis, circular dichroism, microscopy
- HIPE-to-mAb correlation: using emulsion stability data to predict protein behavior

When analyzing formulations, always:
1. Provide a clear risk assessment (High / Medium / Low)
2. Explain the scientific mechanisms driving your assessment
3. Give specific, actionable recommendations
4. Reference relevant scientific principles and published literature where appropriate
5. Use precise scientific terminology while remaining accessible

Format your responses with clear markdown headers and bullet points for readability."""


STABILITY_ANALYSIS_PROMPT = """\
Analyze the structural stability of a monoclonal antibody under the following formulation and storage conditions:

## Formulation Parameters
- **mAb Concentration:** {concentration} mg/mL
- **pH:** {ph}
- **Buffer System:** {buffer}
- **Protein Agent (HIPE Stabilizer):** {protein}
- **Protein Concentration:** {protein_concentration}
- **Storage Temperature:** {temperature} degrees Celsius
- **Interface/Stress Type:** {interface_type}

## Requested Analysis
Please provide:

### 1. Stability Risk Score
Assign a risk level: **High**, **Medium**, or **Low**
Start your response with exactly one of these on its own line:
RISK_LEVEL: High
RISK_LEVEL: Medium
RISK_LEVEL: Low

### 2. Degradation Mechanisms
Identify and explain the primary degradation mechanisms at play given this specific combination \
of conditions and interface type.

### 3. Formulation Assessment
Evaluate whether the buffer, pH, protein agent choice, and concentration are appropriate. \
Identify any concerns.

### 4. Recommendations
Provide specific, actionable recommendations to improve stability. \
Include suggested parameter ranges where applicable.

### 5. Scientific Rationale
Explain the biophysical basis for your assessment, referencing relevant scientific principles \
and studies."""


MECHANISM_EXPLANATION_PROMPT = """\
Explain in detail how the **{interface_type}** interface causes structural instability \
in monoclonal antibodies.

## Please address the following aspects:

### 1. Adsorption Mechanism
How do mAb molecules adsorb to this specific interface? What molecular forces drive this interaction?

### 2. Conformational Changes
What structural changes occur upon adsorption? How does the protein unfold or rearrange \
at this interface?

### 3. Aggregation Pathway
How does interfacial stress at this interface lead to aggregation? Describe the pathway \
from adsorption to particle/aggregate formation.

### 4. Protein Stabilization at Interfaces
How do different protein agents (e.g., BSA, lysozyme, \u03b2-lactoglobulin) behave at this specific \
interface? How does protein type affect adsorption, unfolding, and stabilization?

### 5. Formulation Strategies
What formulation strategies (pH, buffer, protein concentration, temperature) can mitigate \
instability at this interface?

### 6. Real-World Relevance
In what manufacturing or storage scenarios is this interface type most commonly encountered? \
What are the practical implications?"""


FORMULATION_COMPARISON_PROMPT = """\
Compare the following two monoclonal antibody formulations for structural stability:

## Formulation A
- **mAb Concentration:** {conc_a} mg/mL
- **pH:** {ph_a}
- **Buffer System:** {buffer_a}
- **Protein Agent:** {protein_a}
- **Protein Concentration:** {protein_conc_a}
- **Storage Temperature:** {temp_a} degrees Celsius

## Formulation B
- **mAb Concentration:** {conc_b} mg/mL
- **pH:** {ph_b}
- **Buffer System:** {buffer_b}
- **Protein Agent:** {protein_b}
- **Protein Concentration:** {protein_conc_b}
- **Storage Temperature:** {temp_b} degrees Celsius

## Requested Comparison
Please provide:

### 1. Overall Verdict
Which formulation is likely more stable and why? State clearly:
VERDICT: Formulation A is more stable
or
VERDICT: Formulation B is more stable
or
VERDICT: Both formulations are comparable

### 2. Parameter-by-Parameter Comparison
Compare each parameter (pH, buffer, protein agent, concentration, temperature) and explain \
the stability implications of each difference.

### 3. Key Differentiators
What are the most critical differences that drive the stability difference between \
these two formulations?

### 4. Recommendations
Suggest how the weaker formulation could be improved. Could elements from both be combined \
into an optimal formulation?"""


LITERATURE_REVIEW_PROMPT = """\
Provide a comprehensive literature review on the following topic related to monoclonal antibody \
stability and High Internal Phase Emulsions (HIPEs):

**Topic:** {topic}

## Please include:

### 1. Key Published Findings
Summarize the most important findings from published research. Include author names, \
approximate publication years, and journal names where possible.

### 2. Relevance to mAb Stability
How do these findings relate to monoclonal antibody structural stability at interfaces?

### 3. HIPE as a Model System
Explain how protein-stabilized High Internal Phase Emulsions (where proteins act as the emulsifier, \
not surfactants) can serve as a model system for studying protein-interface interactions. \
Why is this approach valuable compared to direct mAb experiments?

### 4. Analytical Techniques
Discuss the role of NMR/DOSY diffusion analysis, circular dichroism, and microscopy \
in characterizing stability in these systems.

### 5. Key References
List 5-10 relevant published papers with authors, title, journal, and year. \
Focus on real, well-known publications in this field."""


NMR_DOSY_INTERPRETATION_PROMPT = """\
Interpret the following NMR DOSY (Diffusion-Ordered Spectroscopy) diffusion analysis results \
in the context of protein/emulsion structural stability:

## Experimental Results
- **Fit Model:** {fit_model}
{fit_details}
- **R-squared of fit:** {r_squared}
- **Sample Type:** HIPE emulsion
- **Temperature:** 25 degrees Celsius
- **Number of gradient steps:** {n_gradients}

## Background on the Fit Models

The **mono-exponential** decay equation, S(b) = I0 * exp(-D * b), \
describes free (unrestricted) diffusion and is appropriate for water or low-concentration HIPE, \
where molecules diffuse without obstruction.

The **bi-exponential** model, S(b) = I0 * [xi * exp(-Df * b) + (1 - xi) * exp(-Ds * b)], \
describes restricted diffusion where two distinct diffusion pools exist — a fast component (Df) \
and a slow component (Ds). This is appropriate for higher-concentration HIPE or when protein \
agents are present at interfaces, creating diffusion barriers. \
(Reference: Scigliani, Grant & Mohammadigoushki, 2023)

## Please provide:

### 1. Interpretation of Diffusion Coefficients
{interp_guidance}

### 2. What Do Df and Ds Represent?
Explain the physical meaning of the fast (Df) and slow (Ds) diffusion coefficients \
in the context of HIPE emulsions and protein stability. What molecular or structural \
environments give rise to each component? What does the pool fraction (xi) tell us \
about the relative populations?

### 3. Acceptable Ranges and Literature Comparison
What are typical/expected values for diffusion coefficients in similar systems? \
Compare the measured values to published literature ranges for:
- Free water (~2.3 x 10^-9 m^2/s at 25C)
- Protein solutions (mAbs: ~4-7 x 10^-11 m^2/s)
- HIPE emulsion droplets (~10^-11 to 10^-10 m^2/s)
- Restricted/confined environments (~10^-12 to 10^-11 m^2/s)
Are the measured values within expected ranges? What deviations might indicate?

### 4. Stability Implications
What does this diffusion behavior suggest about structural integrity? \
Is there evidence of aggregation, degradation, confinement, or stable conformation?

### 5. HIPE-to-mAb Translation
If this is HIPE data, what can we infer about potential mAb behavior under similar conditions? \
How do changes in diffusion correlate with protein stability at interfaces?

### 6. Recommendations
What additional experiments or analyses would strengthen these conclusions?"""


CD_INTERPRETATION_PROMPT = """\
Interpret the following Circular Dichroism (CD) spectroscopy results in the context of \
protein/emulsion secondary structure and stability:

## Experimental Results
- **Wavelength range:** {wavelength_range} nm
- **Dominant features:** {dominant_features}
- **Sample Type:** {sample_type}
- **Min ellipticity wavelength:** {min_wavelength} nm
- **Max ellipticity wavelength:** {max_wavelength} nm

## Please provide:

### 1. Secondary Structure Analysis
Based on the spectral features, what is the likely secondary structure composition? \
(alpha-helix, beta-sheet, random coil, turns)

### 2. Structural Integrity Assessment
Does this spectrum indicate a well-folded, partially unfolded, or denatured structure?

### 3. Stability Implications
What does the CD profile suggest about the structural stability of this formulation? \
Are there signs of conformational change or aggregation?

### 4. HIPE-to-mAb Correlation
How do CD observations in HIPE systems correlate with expected mAb secondary structure behavior?

### 5. Recommendations
What follow-up CD experiments (thermal melt, pH titration, time-course) would provide \
additional stability insights?"""


MICROSCOPY_INTERPRETATION_PROMPT = """\
Interpret microscopy observations of emulsion/protein samples over a stability study:

## Experimental Observations
- **Sample Type:** {sample_type}
- **Observation Period:** {observation_period}
- **Key Observations:** {observations}
- **Magnification:** {magnification}

## Please provide:

### 1. Morphological Analysis
What do the observed features indicate about the sample's microstructure?

### 2. Stability Assessment
Based on the visual observations over time, is the system showing signs of:
- Coalescence or phase separation?
- Ostwald ripening?
- Creaming or sedimentation?
- Stable microstructure retention?

### 3. Droplet/Particle Size Implications
What can be inferred about size distribution and uniformity from the microscopy data?

### 4. HIPE-to-mAb Translation
How do these observations in the HIPE system relate to protein stability at interfaces?

### 5. Recommendations
What additional microscopy techniques or analyses would be valuable?"""

# MalGuard-XAI: Automated Threat Analysis Pipeline
## Project Methodology & Final Report

---

### 1. Introduction
MalGuard-XAI is an advanced, fully-automated malware analysis pipeline designed to overcome the limitations of traditional signature-based antivirus solutions. It integrates static feature extraction, a 5-model machine learning ensemble, eXplainable AI (XAI) using SHAP, and a generative AI reasoning layer powered by Anthropic's Claude. The system analyzes Windows Portable Executable (PE) files and produces human-readable, actionable threat intelligence reports.

---

### 2. Architecture & Pipeline Flow
The pipeline operates autonomously via the CLI or the Streamlit Web UI, passing the target PE file through 5 distinct stages:

1. **Feature Extraction (Stage 1):** Uses `pefile` and `yara-python` to extract 105 static features (entropy, sections, imports), a 50-token API call sequence, and a 1024-length normalized raw byte header vector.
2. **Machine Learning Ensemble (Stage 2 & 4):** A custom soft-voting ensemble comprising 5 models:
   - *Classical Models (Static Features):* Random Forest (GridSearchCV tuned), XGBoost (GridSearchCV tuned), Gradient Boosting.
   - *Deep Learning (Sequential & Raw Bytes):* PyTorch LSTM (API Call sequences), PyTorch 1D CNN (Raw byte arrays).
3. **Explainable AI (Stage 3):** SHAP (SHapley Additive exPlanations) TreeExplainer computes local feature attributions on the classical models to identify exactly which features pushed the model toward a "Malicious" or "Benign" verdict.
4. **Claude AI Reasoning (Stage 5):** The ML outputs and extracted features are fed to the Anthropic Claude API (with a deterministic mock fallback). Claude acts as a Senior Security Analyst, deducing the specific malware family (e.g., RAT, Ransomware), explaining the threat narratively, generating behavioral IOCs, and offering remediation advice.
5. **Reporting Layer:** A unified PDF report is dynamically generated using `reportlab`, containing the verdict, SHAP charts, Claude's analysis, and a consolidated table of IOCs.

---

### 3. Data Pipeline & Benchmarking
The system was trained on a high-fidelity synthetic dataset generated to mirror real-world threat patterns.
- **Benign Profiles:** Low entropy (<6.0), standard compiler imports, clean YARA scans.
- **Malicious Profiles:** High entropy (>7.0) indicating packing/encryption, injection API patterns (`VirtualAlloc` → `WriteProcessMemory` → `CreateRemoteThread`), and specific YARA matches (`anti_debugging`, `crypto_signatures`).

**Ensemble Performance (750-sample Test Set):**
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9987 | 1.0000 | 0.9973 | 0.9987 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LSTM (API Sequences) | 0.9973 | 1.0000 | 0.9947 | 0.9973 | 1.0000 |
| CNN (Raw Bytes) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Voting Ensemble** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Note: Perfect evaluation metrics are a byproduct of the strictly separated synthetic data distributions used in this prototype phase. Real-world in-the-wild benchmarking would yield more variance.*

**Baseline Comparison:**
Traditional AV relies heavily on static hash databases and deterministic YARA signatures. MalGuard-XAI outpaces this baseline by utilizing behavioral sequences (LSTM) and structural anomalies (CNN/RF) to detect zero-day or heavily obfuscated payloads that lack known signatures. Furthermore, standard AV produces opaque binary alerts ("Threat Blocked"), whereas MalGuard-XAI generates a granular, understandable threat narrative for security operations center (SOC) teams.

---

### 4. Technical Stack
- **Languages/Frameworks:** Python 3, PyTorch, Scikit-Learn, XGBoost
- **Analysis:** pefile, yara-python, SHAP
- **Generative AI:** Anthropic Claude API
- **UI & Reporting:** Streamlit, ReportLab

---

### 5. Conclusion
MalGuard-XAI successfully demonstrates the viability of multimodal machine learning and LLM reasoning in automated malware triage. By bridging the gap between raw binary analysis and human-readable reporting, it significantly reduces the cognitive load on SOC analysts and accelerates incident response timelines.

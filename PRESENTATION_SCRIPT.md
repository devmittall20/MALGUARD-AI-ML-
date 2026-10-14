# MalGuard-XAI: Final Presentation Script

## 1. Introduction (2 mins)
**Speaker:** 
"Hello everyone. Today, we are presenting **MalGuard-XAI**, a next-generation automated threat analysis pipeline. 
Traditional antivirus relies heavily on static signatures—if a piece of malware is packed or encrypted, the signature changes, and traditional AV fails. Even when it detects something, the alert is opaque: it just says 'Threat Blocked'.

MalGuard-XAI solves this by combining a multimodal machine learning ensemble with generative AI reasoning. We not only detect highly obfuscated zero-day threats, but we explain exactly *why* it's a threat using SHAP and Anthropic's Claude API."

---

## 2. Architecture Overview (3 mins)
**Speaker:** 
"Let’s look under the hood. Our pipeline operates in 5 distinct stages:
1. **Extraction:** We extract 105 static features, a 50-token API call sequence, and a 1024-byte header vector directly from the binary.
2. **Machine Learning:** We feed this into a 5-model soft-voting ensemble. This includes Random Forest, XGBoost, and Gradient Boosting for static features, plus a PyTorch LSTM for API sequences and a 1D CNN for raw bytes.
3. **SHAP XAI:** We calculate SHAP values to identify the local features driving the classification.
4. **Claude AI:** We pass the ML output and features to Anthropic's Claude, which acts as a virtual SOC analyst, deducing the malware family and generating behavioral Indicators of Compromise (IOCs).
5. **Reporting:** Everything is packaged into a downloadable PDF."

---

## 3. Live Demo - Web App (5 mins)
*(Open the Streamlit App: `streamlit run app.py`)*

**Speaker:**
"Instead of just talking about it, let's see it in action via our Streamlit Web Demo. 
I'm going to upload a standard Windows executable, `notepad.exe`."

*(Upload `notepad.exe` to the Streamlit UI and click "Run Analysis")*

**Speaker:**
"Behind the scenes, the PE file is being stripped, analyzed by our 5 PyTorch and Scikit-Learn models, and evaluated by Claude.

As you can see on the dashboard:
- **Verdict:** Benign. The ensemble correctly identified this as safe with high confidence.
- **AI Threat Reasoning:** Claude confirms that the static characteristics are consistent with standard compilation profiles.
- **SHAP Chart:** On the right, the SHAP plot shows that features like `imports_func_count` pushed the model toward a benign verdict.
- **PDF Report:** Finally, I can click this button to download the automated PDF threat intelligence report, completely ready for a SOC compliance audit."

---

## 4. Conclusion & Q&A (2 mins)
**Speaker:**
"By bridging the gap between raw binary analysis and human-readable reporting, MalGuard-XAI significantly reduces the cognitive load on SOC analysts and accelerates incident response timelines. 

Thank you for your time. I'll now open the floor for any questions."

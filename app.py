import os
import sys
import tempfile
import streamlit as st
import pandas as pd
from PIL import Image

# Modify path to allow importing src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.analysis.analyze import main as analyze_main

class DummyArgs:
    def __init__(self, file_path):
        self.file = file_path
        self.models_dir = "models"
        self.reports_dir = "reports"
        self.data_dir = "data/processed"
        self.yara_rules = None

st.set_page_config(
    page_title="MalGuard-XAI | AI Malware Analysis",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ MalGuard-XAI: Automated Threat Analysis")
st.markdown("Upload a Windows Portable Executable (PE) file to analyze it using our 5-model multimodal ensemble (RF, XGBoost, Gradient Boosting, LSTM, CNN) and Claude AI reasoning layer.")

uploaded_file = st.file_uploader("Upload PE File (.exe, .dll, etc.)", type=["exe", "dll", "sys", "bin"])

if uploaded_file is not None:
    st.info(f"File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
    
    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing binary..."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            try:
                # Prepare arguments for the analyzer
                args = DummyArgs(tmp_path)
                
                # Run the pipeline (this might take a few seconds)
                results = analyze_main(custom_args=args)
                
                # --- RESULTS DISPLAY ---
                st.success("Analysis Complete!")
                
                # Top Metrics Row
                col1, col2, col3, col4 = st.columns(4)
                
                verdict = results["verdict"]
                verdict_color = "🔴" if verdict == "Malicious" else "🟢"
                col1.metric("Verdict", f"{verdict_color} {verdict}")
                
                confidence = results["confidence"]
                col2.metric("Confidence", f"{confidence * 100:.1f}%")
                
                risk_level = results["risk_level"]
                col3.metric("Risk Level", risk_level)
                
                family = results.get("claude_analysis", {}).get("malware_family", "None")
                col4.metric("Malware Family", family)
                
                st.markdown("---")
                
                # Detailed Layout
                left_col, right_col = st.columns([1, 1])
                
                with left_col:
                    st.subheader("🤖 AI Threat Reasoning")
                    claude_analysis = results.get("claude_analysis", {})
                    
                    st.markdown(f"**Summary:** {claude_analysis.get('summary', '')}")
                    st.markdown(f"**Detailed Analysis:**\n{claude_analysis.get('detailed_analysis', '')}")
                    
                    st.markdown("### 🔍 Indicators of Compromise (IOCs)")
                    ioc_data = results.get("ioc_data", {})
                    st.code(f"MD5: {ioc_data.get('md5')}\nSHA-256: {ioc_data.get('sha256')}")
                    
                    if ioc_data.get("suspicious_apis"):
                        st.markdown("**Suspicious APIs:**")
                        st.write(", ".join(ioc_data["suspicious_apis"][:10]))
                        
                    if ioc_data.get("network_iocs"):
                        st.markdown("**Network IOCs:**")
                        for ioc in ioc_data["network_iocs"]:
                            st.write(f"- `{ioc}`")
                            
                    if ioc_data.get("registry_iocs"):
                        st.markdown("**Registry IOCs:**")
                        for ioc in ioc_data["registry_iocs"]:
                            st.write(f"- `{ioc}`")
                            
                    st.markdown("### 🛡️ Remediation")
                    st.info(claude_analysis.get("remediation", "No action required."))
                    
                with right_col:
                    st.subheader("📊 SHAP Feature Explanations")
                    shap_path = results.get("shap_chart")
                    if shap_path and os.path.exists(shap_path):
                        img = Image.open(shap_path)
                        st.image(img, caption="Local Feature Attributions driving the ML prediction.", use_container_width=True)
                    else:
                        st.warning("SHAP chart not available.")
                        
                st.markdown("---")
                
                # PDF Download
                pdf_path = results.get("pdf_report")
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        
                    st.download_button(
                        label="📄 Download Full Threat Intelligence Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"MalGuard_Report_{uploaded_file.name}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                
            except SystemExit as e:
                st.error(f"Analysis failed. Analyzer exited with code {e.code}.")
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

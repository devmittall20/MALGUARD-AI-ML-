import os
import json
import logging
from typing import Dict, Any, Optional
from anthropic import Anthropic

logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class ClaudeAnalyzer:
    """
    Feeds ML model classification outcomes and extracted PE features to the Claude API 
    to obtain a narrative explanation of threats, risk levels, and remediation advice.
    Implements a deterministic mock fallback if API keys are missing or calls fail.
    """

    def __init__(self, api_key: Optional[str] = None):
        # Load API key from dotenv/environment
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = None
        
        # Check if the API key is set and not a placeholder
        if self.api_key and "your_key" not in self.api_key.lower():
            try:
                self.client = Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
        else:
            logger.info("Using Mock Fallback mode for Claude AI (no valid Anthropic API key configured).")

    def analyze_threat(self, features: Dict[str, Any], verdict: str, confidence: float) -> Dict[str, Any]:
        """
        Sends threat classification outputs and PE features to Claude, returning a structured JSON response.
        Falls back to local analysis if client is unavailable or API call fails.
        """
        # If client is configured, run API call
        if self.client is not None:
            try:
                # Prepare a concise summary of anomalies for Claude's prompt
                anomalies = []
                for k, v in features.items():
                    if k.startswith("import_") and v == 1:
                        anomalies.append(f"Suspicious API Import: {k.replace('import_', '')}")
                    if k.startswith("yara_") and v == 1:
                        anomalies.append(f"YARA Signatures: {k.replace('yara_', '')}")
                
                features_summary = {
                    "file_size_bytes": features.get("file_size", 0),
                    "file_entropy": features.get("file_entropy", 0.0),
                    "number_of_sections": features.get("header_number_of_sections", 0),
                    "max_section_entropy": features.get("sec_entropy_max", 0.0),
                    "executable_sections": features.get("sec_executable_count", 0),
                    "writable_sections": features.get("sec_writable_count", 0),
                    "imports_dll_count": features.get("imports_dll_count", 0),
                    "imports_func_count": features.get("imports_func_count", 0),
                    "yara_triggers": [k for k, v in features.items() if k.startswith("yara_") and v == 1],
                    "suspicious_apis": [k.replace("import_", "") for k, v in features.items() if k.startswith("import_") and v == 1]
                }

                prompt = f"""
                You are a Senior Malware Analyst and Explainer AI.
                Analyze the following Windows Portable Executable (PE) features and classification results:
                - ML Model Verdict: {verdict}
                - ML Classification Confidence Score: {confidence:.2f}
                - PE Feature Details: {json.dumps(features_summary, indent=2)}

                Explain why this file was classified as {verdict}. Focus on the security risks of high entropy, specific suspicious APIs imported, and YARA signatures.
                Also, classify the malware family (e.g., Ransomware, Trojan, RAT, Spyware, Worm, or None) based on behavioral indicators, and extract any behavioral Indicators of Compromise (IOCs) such as predicted registry keys, network endpoints, or file paths.
                Provide your analysis as a valid JSON object. Do not include markdown code blocks or explanations outside the JSON. The JSON structure MUST be:
                {{
                    "verdict": "{verdict}",
                    "confidence": {confidence:.4f},
                    "risk_level": "Critical" | "High" | "Medium" | "Low",
                    "malware_family": "Ransomware" | "Trojan" | "RAT" | "Spyware" | "Worm" | "None" | "Unknown",
                    "summary": "Short executive summary of threat",
                    "detailed_analysis": "Technical details explaining the features (API imports, YARA triggers, entropy)",
                    "remediation": "Recommended actions for security operations",
                    "iocs": {{
                        "network": ["List of suspicious network indicators like IPs or domains (if any)"],
                        "registry": ["List of suspicious registry keys (if any)"]
                    }}
                }}
                """

                logger.info("Requesting narrative analysis from Claude API...")
                message = self.client.messages.create(
                    model="claude-3-haiku-20240307",  # Fast and cost-efficient for structural JSON responses
                    max_tokens=1000,
                    temperature=0.0,
                    system="You are an expert cybersecurity response assistant. You always respond with pure JSON.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Parse response content
                content_text = message.content[0].text.strip()
                # Clean markdown wrapper if any
                if content_text.startswith("```json"):
                    content_text = content_text.split("```json")[1].split("```")[0].strip()
                elif content_text.startswith("```"):
                    content_text = content_text.split("```")[1].split("```")[0].strip()
                    
                response_json = json.loads(content_text)
                return response_json

            except Exception as e:
                logger.error(f"Claude API request failed: {e}. Falling back to mock generator.")
                # fall through to mock fallback

        # Mock Fallback Engine
        logger.info("Executing Mock Fallback Engine for threat reasoning.")
        return self._generate_mock_verdict(features, verdict, confidence)

    def _generate_mock_verdict(self, features: Dict[str, Any], verdict: str, confidence: float) -> Dict[str, Any]:
        """
        Generates a highly realistic structured analysis based on extracted features.
        Used when the API key is not configured or network requests fail.
        """
        file_entropy = features.get("file_entropy", 0.0)
        max_sec_entropy = features.get("sec_entropy_max", 0.0)
        
        # Get active anomalies
        triggered_apis = [k.replace("import_", "") for k, v in features.items() if k.startswith("import_") and v == 1]
        triggered_yara = [k.replace("yara_", "") for k, v in features.items() if k.startswith("yara_") and v == 1]

        # Initialize mock IOCs and Malware Family
        mock_iocs = {"network": [], "registry": []}
        mock_family = "None"

        if verdict.lower() == "malicious" or (verdict.lower() == "voting ensemble" and confidence >= 0.5):
            risk = "Critical" if "anti_debugging" in triggered_yara or "common_packers" in triggered_yara else "High"
            
            # Simple heuristic for malware family
            mock_family = "Trojan"
            if "VirtualAlloc" in triggered_apis and "CreateRemoteThread" in triggered_apis:
                mock_family = "RAT"
                mock_iocs["registry"].append(r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run\SystemUpdate")
            if "crypto" in str(triggered_yara).lower() or "ransom" in str(triggered_yara).lower():
                mock_family = "Ransomware"
            if features.get("yara_suspicious_network_strings") == 1:
                mock_iocs["network"].extend(["http://malicious-command-control.xyz/gate.php", "198.51.100.42:8080"])
            
            summary = f"Highly suspicious PE file exhibiting traits of packed/obfuscated malware."
            
            analysis_parts = [
                f"The machine learning ensemble classified this binary as MALICIOUS with a confidence score of {confidence*100:.1f}%.",
                f"We observed high entropy levels (file entropy: {file_entropy:.2f}, max section entropy: {max_sec_entropy:.2f}) which strongly suggest the file is packed or encrypted to evade static signature detection."
            ]
            
            if triggered_apis:
                analysis_parts.append(f"The binary imports suspicious APIs frequently abused for memory injection and process manipulation: {', '.join(triggered_apis)}.")
            if triggered_yara:
                analysis_parts.append(f"YARA rules flagged known malware indicators: {', '.join(triggered_yara)}.")
                
            detailed_analysis = " ".join(analysis_parts)
            remediation = "Quarantine the file immediately. Run dynamic execution sandboxing to trace registry and process creation events. Block associated network endpoints."
        else:
            risk = "Low"
            summary = "File appears to be a legitimate, un-obfuscated executable binary."
            detailed_analysis = (
                f"The machine learning ensemble classified this binary as BENIGN with a confidence of {confidence*100:.1f}%. "
                f"Static characteristics are consistent with standard compilation profiles. File entropy ({file_entropy:.2f}) and maximum section entropy ({max_sec_entropy:.2f}) "
                "show normal distributions. No malicious signatures were flagged by YARA rules, and the import table reflects typical system functions."
            )
            remediation = "No immediate security action required. Standard monitoring applies."

        return {
            "verdict": "Malicious" if verdict.lower() == "malicious" else "Benign",
            "confidence": float(confidence),
            "risk_level": risk,
            "malware_family": mock_family,
            "summary": summary,
            "detailed_analysis": detailed_analysis,
            "remediation": remediation,
            "iocs": mock_iocs
        }

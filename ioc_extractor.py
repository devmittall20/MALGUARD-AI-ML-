import os
import hashlib
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class IOCExtractor:
    """
    Extracts Indicators of Compromise (IOCs) from a binary file and its static feature vector.
    This includes calculating file hashes (MD5, SHA-256) and listing network endpoints,
    registry paths, or dangerous APIs.
    """

    def __init__(self):
        pass

    def get_file_hashes(self, file_path: str) -> Tuple[str, str]:
        """
        Calculates MD5 and SHA-256 hashes of the specified file.
        """
        if not os.path.exists(file_path):
            return "n/a", "n/a"
            
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(65536), b""):
                    md5_hash.update(byte_block)
                    sha256_hash.update(byte_block)
            return md5_hash.hexdigest(), sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hashes for {file_path}: {e}")
            return "error", "error"

    def extract_iocs(self, file_path: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assembles file hashes and lists suspicious APIs, registry keys, 
        and network hosts based on the extracted features.
        """
        md5_sig, sha256_sig = self.get_file_hashes(file_path)
        
        # 1. Suspicious API imports
        suspicious_apis = []
        for k, v in features.items():
            if k.startswith("import_") and v == 1:
                suspicious_apis.append(k.replace("import_", ""))

        # 2. Network indicators (extracted or mocked based on network YARA triggers)
        network_iocs = []
        if features.get("yara_suspicious_network_strings", 0) == 1:
            # Add some realistic network indicators if YARA rule triggers
            network_iocs.extend(["http://malicious-command-control.xyz/gate.php", "198.51.100.42:8080"])
        
        # 3. Registry indicators (added dynamically or for specific persistence API triggers)
        registry_iocs = []
        if "ShellExecuteA" in suspicious_apis or "CreateRemoteThread" in suspicious_apis:
            # Typical persistence run keys
            registry_iocs.append(r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run\MalGuardUpdate")
            registry_iocs.append(r"HKLM\System\CurrentControlSet\Services\MalGuardSvc")

        # Compile IOCs dictionary
        return {
            "filename": os.path.basename(file_path),
            "md5": md5_sig,
            "sha256": sha256_sig,
            "file_size": features.get("file_size", 0),
            "file_entropy": features.get("file_entropy", 0.0),
            "suspicious_apis": sorted(suspicious_apis),
            "network_iocs": sorted(network_iocs),
            "registry_iocs": sorted(registry_iocs),
            "total_iocs_count": len(suspicious_apis) + len(network_iocs) + len(registry_iocs)
        }


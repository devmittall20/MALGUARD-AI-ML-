import os
import random
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ..extraction.extractor import PEFeatureExtractor

logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class DatasetManager:
    """
    Manages collection, generation, splitting, and saving of features for MalGuard-XAI.
    """

    def __init__(self, data_root: str = "data", yara_rules_path: Optional[str] = None):
        self.data_root = data_root
        self.raw_benign = os.path.join(data_root, "raw", "benign")
        self.raw_malicious = os.path.join(data_root, "raw", "malicious")
        self.processed_dir = os.path.join(data_root, "processed")
        self.extractor = PEFeatureExtractor(yara_rules_path=yara_rules_path)
        
        # Ensure directories exist
        os.makedirs(self.raw_benign, exist_ok=True)
        os.makedirs(self.raw_malicious, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Re-export feature keys schema to guarantee consistency
        self.suspicious_imports = self.extractor.suspicious_imports
        self.vocab_mnemonics = self.extractor.vocab_mnemonics
        self.vocab_2grams = self.extractor.vocab_2grams
        self.vocab_3grams = self.extractor.vocab_3grams

    def get_feature_schema(self) -> List[str]:
        """Returns the sorted list of all expected feature names (excluding label)."""
        schema = [
            "file_size", "file_entropy", "header_machine", "header_number_of_sections",
            "header_time_date_stamp", "header_characteristics", "opt_magic", "opt_major_linker_version",
            "opt_minor_linker_version", "opt_size_of_code", "opt_size_of_init_data", "opt_size_of_uninit_data",
            "opt_address_of_entry_point", "opt_base_of_code", "opt_image_base", "opt_section_alignment",
            "opt_file_alignment", "opt_major_os_version", "opt_minor_os_version", "opt_major_image_version",
            "opt_minor_image_version", "opt_major_subsystem_version", "opt_minor_subsystem_version",
            "opt_size_of_image", "opt_size_of_headers", "opt_check_sum", "opt_subsystem",
            "opt_dll_characteristics", "opt_size_of_stack_reserve", "opt_size_of_stack_commit",
            "opt_size_of_heap_reserve", "opt_size_of_heap_commit", "opt_loader_flags", "opt_num_rva_and_sizes",
            "sec_entropy_min", "sec_entropy_max", "sec_entropy_mean", "sec_entropy_var",
            "sec_raw_size_min", "sec_raw_size_max", "sec_raw_size_mean", "sec_raw_size_var",
            "sec_virtual_size_min", "sec_virtual_size_max", "sec_virtual_size_mean", "sec_virtual_size_var",
            "sec_executable_count", "sec_writable_count", "imports_dll_count", "imports_func_count",
            "exports_count", "opcode_total_count", "yara_suspicious_apis", "yara_anti_debugging",
            "yara_common_packers", "yara_suspicious_network_strings"
        ]
        
        # Add dynamic lists
        for imp in self.suspicious_imports:
            schema.append(f"import_{imp}")
        for op in self.vocab_mnemonics:
            schema.append(f"opcode_freq_{op}")
        for gram in self.vocab_2grams:
            schema.append(f"opcode_2gram_{gram}")
        for gram in self.vocab_3grams:
            schema.append(f"opcode_3gram_{gram}")
            
        # Add API sequence and raw byte features
        for i in range(50):
            schema.append(f"api_seq_{i}")
        for i in range(1024):
            schema.append(f"byte_{i}")
            
        return sorted(schema)

    def scan_and_extract(self) -> pd.DataFrame:
        """
        Scans raw directories, extracts features from PE files, and compiles a DataFrame.
        """
        records = []
        schema_set = set(self.get_feature_schema())

        # Scan benign files
        for f_name in os.listdir(self.raw_benign):
            f_path = os.path.join(self.raw_benign, f_name)
            if os.path.isfile(f_path):
                logger.info(f"Extracting features from benign: {f_name}")
                feats = self.extractor.extract_features(f_path)
                if "error" not in feats:
                    # Clean features to match schema exactly
                    clean_feats = {k: v for k, v in feats.items() if k in schema_set}
                    clean_feats["label"] = 0
                    records.append(clean_feats)
                else:
                    logger.warning(f"Failed to parse {f_name}: {feats['error']}")

        # Scan malicious files
        for f_name in os.listdir(self.raw_malicious):
            f_path = os.path.join(self.raw_malicious, f_name)
            if os.path.isfile(f_path):
                logger.info(f"Extracting features from malicious: {f_name}")
                feats = self.extractor.extract_features(f_path)
                if "error" not in feats:
                    # Clean features to match schema exactly
                    clean_feats = {k: v for k, v in feats.items() if k in schema_set}
                    clean_feats["label"] = 1
                    records.append(clean_feats)
                else:
                    logger.warning(f"Failed to parse {f_name}: {feats['error']}")

        if not records:
            return pd.DataFrame(columns=self.get_feature_schema() + ["label"])

        df = pd.DataFrame(records)
        # Ensure all columns are present and filled
        for col in self.get_feature_schema():
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.get_feature_schema() + ["label"]]
        return df

    def generate_synthetic_dataset(self, n_samples: int = 1000, benign_ratio: float = 0.5) -> pd.DataFrame:
        """
        Generates a high-fidelity synthetic PE feature dataset based on empirical distributions 
        of benign and malicious Windows executables. Useful for testing pipelines and training.
        """
        logger.info(f"Generating synthetic dataset of size {n_samples} with benign_ratio={benign_ratio}")
        n_benign = int(n_samples * benign_ratio)
        n_malicious = n_samples - n_benign
        
        schema = self.get_feature_schema()
        records = []

        # Helper to generate bounded values
        def normal_bounded(mean: float, std: float, lower: float, upper: float) -> float:
            val = np.random.normal(mean, std)
            return float(np.clip(val, lower, upper))

        # 1. Generate Benign Samples
        for _ in range(n_benign):
            rec = {}
            # File info
            rec["file_size"] = float(np.random.lognormal(13.5, 0.8)) # ~500KB - 2MB
            rec["file_entropy"] = normal_bounded(5.4, 0.7, 1.0, 7.9)
            rec["header_machine"] = int(np.random.choice([0x014c, 0x8664], p=[0.2, 0.8]))
            rec["header_number_of_sections"] = int(normal_bounded(4.2, 1.1, 2, 8))
            rec["header_time_date_stamp"] = int(1700000000 + np.random.randint(-1e8, 1e8))
            rec["header_characteristics"] = int(np.random.choice([0x0102, 0x2102]))
            
            # Optional Header
            rec["opt_magic"] = int(np.random.choice([0x10b, 0x20b], p=[0.2, 0.8]))
            rec["opt_major_linker_version"] = int(normal_bounded(14.0, 2.0, 6, 20))
            rec["opt_minor_linker_version"] = int(np.random.randint(0, 30))
            rec["opt_size_of_code"] = int(rec["file_size"] * 0.4)
            rec["opt_size_of_init_data"] = int(rec["file_size"] * 0.4)
            rec["opt_size_of_uninit_data"] = int(np.random.choice([0, 1024, 4096], p=[0.9, 0.05, 0.05]))
            rec["opt_address_of_entry_point"] = int(np.random.randint(0x1000, 0x50000))
            rec["opt_base_of_code"] = 0x1000
            rec["opt_image_base"] = int(np.random.choice([0x400000, 0x140000000]))
            rec["opt_section_alignment"] = 4096
            rec["opt_file_alignment"] = 512
            rec["opt_major_os_version"] = int(np.random.choice([6, 10], p=[0.3, 0.7]))
            rec["opt_minor_os_version"] = 0
            rec["opt_major_image_version"] = 0
            rec["opt_minor_image_version"] = 0
            rec["opt_major_subsystem_version"] = int(rec["opt_major_os_version"])
            rec["opt_minor_subsystem_version"] = 0
            rec["opt_size_of_image"] = int(rec["file_size"] * 1.5)
            rec["opt_size_of_headers"] = 1024
            rec["opt_check_sum"] = random.randint(0, 0xffffffff)
            rec["opt_subsystem"] = int(np.random.choice([2, 3], p=[0.8, 0.2])) # GUI vs Console
            rec["opt_dll_characteristics"] = int(np.random.choice([0x8140, 0xc140]))
            rec["opt_size_of_stack_reserve"] = 0x100000
            rec["opt_size_of_stack_commit"] = 0x1000
            rec["opt_size_of_heap_reserve"] = 0x100000
            rec["opt_size_of_heap_commit"] = 0x1000
            rec["opt_loader_flags"] = 0
            rec["opt_num_rva_and_sizes"] = 16

            # Sections
            rec["sec_entropy_min"] = normal_bounded(1.2, 0.8, 0.0, 4.0)
            rec["sec_entropy_max"] = normal_bounded(5.6, 0.6, 3.0, 6.8) # Moderate entropy
            rec["sec_entropy_mean"] = (rec["sec_entropy_min"] + rec["sec_entropy_max"]) / 2
            rec["sec_entropy_var"] = normal_bounded(1.5, 0.5, 0.1, 4.0)
            
            rec["sec_raw_size_min"] = float(np.random.choice([512, 1024, 4096]))
            rec["sec_raw_size_max"] = rec["file_size"] * 0.6
            rec["sec_raw_size_mean"] = rec["file_size"] * 0.3
            rec["sec_raw_size_var"] = (rec["sec_raw_size_max"] - rec["sec_raw_size_min"]) ** 2 / 12
            
            rec["sec_virtual_size_min"] = rec["sec_raw_size_min"] * 1.1
            rec["sec_virtual_size_max"] = rec["sec_raw_size_max"] * 1.2
            rec["sec_virtual_size_mean"] = rec["sec_raw_size_mean"] * 1.15
            rec["sec_virtual_size_var"] = (rec["sec_virtual_size_max"] - rec["sec_virtual_size_min"]) ** 2 / 12

            rec["sec_executable_count"] = 1
            rec["sec_writable_count"] = int(normal_bounded(1.8, 0.6, 1, 4))

            # Imports & Exports
            rec["imports_dll_count"] = int(normal_bounded(4.5, 1.8, 1, 10))
            rec["imports_func_count"] = int(normal_bounded(180, 70, 10, 500))
            rec["exports_count"] = int(np.random.choice([0, np.random.randint(1, 20)], p=[0.8, 0.2]))

            # Suspicious imports (very low probability for benign)
            for imp in self.suspicious_imports:
                p = 0.05
                if imp in ["GetProcAddress", "LoadLibraryA", "LoadLibraryW"]:
                    p = 0.35 # Standard utilities
                rec[f"import_{imp}"] = int(np.random.random() < p)

            # Opcode Disassembly
            rec["opcode_total_count"] = int(normal_bounded(800, 300, 100, 2000))
            
            # Opcode frequencies (standard code profile)
            freqs = {
                "mov": normal_bounded(0.33, 0.04, 0.20, 0.45),
                "push": normal_bounded(0.22, 0.03, 0.10, 0.35),
                "pop": normal_bounded(0.08, 0.02, 0.01, 0.15),
                "call": normal_bounded(0.08, 0.02, 0.02, 0.15),
                "jmp": normal_bounded(0.04, 0.01, 0.00, 0.08),
                "xor": normal_bounded(0.02, 0.01, 0.00, 0.06), # XOR is rare in standard compiler code
                "add": normal_bounded(0.04, 0.01, 0.00, 0.08),
                "sub": normal_bounded(0.04, 0.01, 0.00, 0.08),
                "cmp": normal_bounded(0.04, 0.01, 0.00, 0.08),
                "test": normal_bounded(0.03, 0.01, 0.00, 0.06),
                "je": normal_bounded(0.03, 0.01, 0.00, 0.06),
                "jne": normal_bounded(0.03, 0.01, 0.00, 0.06),
                "ret": normal_bounded(0.02, 0.01, 0.00, 0.06),
                "lea": normal_bounded(0.04, 0.01, 0.00, 0.08)
            }
            # Normalize single mnemonic frequencies
            tot_sum = sum(freqs.values())
            for op in self.vocab_mnemonics:
                rec[f"opcode_freq_{op}"] = freqs[op] / tot_sum

            # N-grams (rare for benign)
            for gram in self.vocab_2grams:
                p = 0.05
                if gram in ["push_mov", "mov_mov"]:
                    p = 0.25
                rec[f"opcode_2gram_{gram}"] = normal_bounded(p, p/2, 0.0, 0.4)
            for gram in self.vocab_3grams:
                p = 0.02
                if gram in ["push_mov_sub", "mov_mov_mov"]:
                    p = 0.1
                rec[f"opcode_3gram_{gram}"] = normal_bounded(p, p/2, 0.0, 0.2)

            # YARA signatures (very low)
            rec["yara_suspicious_apis"] = int(np.random.random() < 0.08)
            rec["yara_anti_debugging"] = int(np.random.random() < 0.03)
            rec["yara_common_packers"] = int(np.random.random() < 0.01)
            rec["yara_suspicious_network_strings"] = int(np.random.random() < 0.12)

            # Generate synthetic API sequence for benign
            n_imports = random.randint(5, 30)
            benign_api_pool = list(range(18, 32))
            api_seq = [float(random.choice(benign_api_pool)) for _ in range(n_imports)]
            for idx in range(len(api_seq)):
                if random.random() < 0.15:
                    api_seq[idx] = 1.0 # UNK
            api_seq = api_seq + [0.0] * (50 - len(api_seq))
            for i, val in enumerate(api_seq):
                rec[f"api_seq_{i}"] = val

            # Generate synthetic raw bytes for benign
            byte_seq = [0.0] * 1024
            byte_seq[0] = 77.0 / 255.0 # M
            byte_seq[1] = 90.0 / 255.0 # Z
            for i in range(2, 64):
                byte_seq[i] = float(random.randint(0, 100)) / 255.0
            for i in range(64, 256):
                if random.random() < 0.4:
                    byte_seq[i] = float(random.randint(0, 120)) / 255.0
            for i in range(256, 1024):
                if random.random() < 0.05:
                    byte_seq[i] = float(random.randint(0, 50)) / 255.0
            for i, val in enumerate(byte_seq):
                rec[f"byte_{i}"] = val

            rec["label"] = 0
            records.append(rec)

        # 2. Generate Malicious Samples
        for _ in range(n_malicious):
            rec = {}
            # File info
            rec["file_size"] = float(np.random.lognormal(12.0, 0.9)) # ~100KB - 500KB (often smaller/packed)
            # High entropy (due to encryption/packing)
            rec["file_entropy"] = normal_bounded(7.4, 0.4, 5.0, 7.99)
            rec["header_machine"] = int(np.random.choice([0x014c, 0x8664], p=[0.3, 0.7]))
            rec["header_number_of_sections"] = int(normal_bounded(5.1, 1.4, 2, 9))
            rec["header_time_date_stamp"] = int(1700000000 + np.random.randint(-1e8, 1e8))
            rec["header_characteristics"] = int(np.random.choice([0x0102, 0x2102]))

            # Optional Header
            rec["opt_magic"] = int(np.random.choice([0x10b, 0x20b], p=[0.3, 0.7]))
            rec["opt_major_linker_version"] = int(normal_bounded(10.0, 4.0, 2, 20))
            rec["opt_minor_linker_version"] = int(np.random.randint(0, 30))
            rec["opt_size_of_code"] = int(rec["file_size"] * 0.3)
            rec["opt_size_of_init_data"] = int(rec["file_size"] * 0.5)
            rec["opt_size_of_uninit_data"] = int(np.random.choice([0, 4096, 16384], p=[0.7, 0.2, 0.1]))
            rec["opt_address_of_entry_point"] = int(np.random.randint(0x1000, 0x80000))
            rec["opt_base_of_code"] = 0x1000
            rec["opt_image_base"] = int(np.random.choice([0x400000, 0x140000000]))
            rec["opt_section_alignment"] = 4096
            rec["opt_file_alignment"] = 512
            rec["opt_major_os_version"] = int(np.random.choice([4, 5, 6, 10], p=[0.1, 0.1, 0.3, 0.5]))
            rec["opt_minor_os_version"] = 0
            rec["opt_major_image_version"] = 0
            rec["opt_minor_image_version"] = 0
            rec["opt_major_subsystem_version"] = int(rec["opt_major_os_version"])
            rec["opt_minor_subsystem_version"] = 0
            rec["opt_size_of_image"] = int(rec["file_size"] * 2.2) # Packed files map larger virtual size
            rec["opt_size_of_headers"] = 1024
            rec["opt_check_sum"] = random.randint(0, 0xffffffff)
            rec["opt_subsystem"] = int(np.random.choice([2, 3], p=[0.9, 0.1]))
            rec["opt_dll_characteristics"] = int(np.random.choice([0x8140, 0xc140, 0x0000], p=[0.4, 0.4, 0.2]))
            rec["opt_size_of_stack_reserve"] = 0x100000
            rec["opt_size_of_stack_commit"] = 0x1000
            rec["opt_size_of_heap_reserve"] = 0x100000
            rec["opt_size_of_heap_commit"] = 0x1000
            rec["opt_loader_flags"] = 0
            rec["opt_num_rva_and_sizes"] = 16

            # Sections
            rec["sec_entropy_min"] = normal_bounded(2.5, 1.2, 0.0, 6.0)
            rec["sec_entropy_max"] = normal_bounded(7.7, 0.2, 6.5, 7.99) # High maximum section entropy (packed)
            rec["sec_entropy_mean"] = (rec["sec_entropy_min"] + rec["sec_entropy_max"]) / 2
            rec["sec_entropy_var"] = normal_bounded(2.8, 0.8, 0.5, 5.0)

            rec["sec_raw_size_min"] = float(np.random.choice([0, 512, 1024])) # Packed sections can have 0 raw size
            rec["sec_raw_size_max"] = rec["file_size"] * 0.8
            rec["sec_raw_size_mean"] = rec["file_size"] * 0.25
            rec["sec_raw_size_var"] = (rec["sec_raw_size_max"] - rec["sec_raw_size_min"]) ** 2 / 12

            # Virtual size is often much larger than raw size in malware
            rec["sec_virtual_size_min"] = rec["sec_raw_size_min"] * 1.5
            rec["sec_virtual_size_max"] = rec["sec_raw_size_max"] * 3.5
            rec["sec_virtual_size_mean"] = rec["sec_raw_size_mean"] * 2.5
            rec["sec_virtual_size_var"] = (rec["sec_virtual_size_max"] - rec["sec_virtual_size_min"]) ** 2 / 12

            rec["sec_executable_count"] = int(np.random.choice([1, 2], p=[0.8, 0.2]))
            rec["sec_writable_count"] = int(normal_bounded(2.5, 0.8, 1, 5)) # More writable sections

            # Imports & Exports (malware/packed files typically import very few DLLs/funcs)
            rec["imports_dll_count"] = int(normal_bounded(1.8, 0.9, 1, 4))
            rec["imports_func_count"] = int(normal_bounded(24, 15, 2, 70))
            rec["exports_count"] = int(np.random.choice([0, np.random.randint(1, 5)], p=[0.95, 0.05]))

            # Suspicious imports (high probability)
            for imp in self.suspicious_imports:
                p = 0.55
                if imp in ["GetProcAddress", "LoadLibraryA", "LoadLibraryW"]:
                    p = 0.88 # Very high utility in malware to resolve APIs
                elif imp in ["VirtualAlloc", "VirtualProtect"]:
                    p = 0.70 # Memory injection
                elif imp in ["WriteProcessMemory", "CreateRemoteThread"]:
                    p = 0.45 # Process hollowing / injection
                rec[f"import_{imp}"] = int(np.random.random() < p)

            # Opcode Disassembly (often smaller code block disassembles due to encryption)
            rec["opcode_total_count"] = int(normal_bounded(150, 80, 10, 500))

            # Opcode frequencies (lots of XOR, decryption loops, call, jump obfuscation)
            freqs = {
                "mov": normal_bounded(0.24, 0.04, 0.10, 0.35),
                "push": normal_bounded(0.12, 0.03, 0.05, 0.25),
                "pop": normal_bounded(0.06, 0.02, 0.01, 0.12),
                "call": normal_bounded(0.07, 0.02, 0.01, 0.15),
                "jmp": normal_bounded(0.08, 0.02, 0.01, 0.15), # High jumps
                "xor": normal_bounded(0.12, 0.04, 0.02, 0.25), # XOR is heavily present for decryption
                "add": normal_bounded(0.06, 0.02, 0.00, 0.12),
                "sub": normal_bounded(0.06, 0.02, 0.00, 0.12),
                "cmp": normal_bounded(0.05, 0.02, 0.00, 0.12),
                "test": normal_bounded(0.04, 0.02, 0.00, 0.10),
                "je": normal_bounded(0.04, 0.02, 0.00, 0.10),
                "jne": normal_bounded(0.04, 0.02, 0.00, 0.10),
                "ret": normal_bounded(0.01, 0.01, 0.00, 0.05),
                "lea": normal_bounded(0.01, 0.01, 0.00, 0.05)
            }
            # Normalize
            tot_sum = sum(freqs.values())
            for op in self.vocab_mnemonics:
                rec[f"opcode_freq_{op}"] = freqs[op] / tot_sum

            # N-grams (high for signature transitions)
            for gram in self.vocab_2grams:
                p = 0.08
                if gram in ["xor_xor", "call_jmp"]:
                    p = 0.22
                rec[f"opcode_2gram_{gram}"] = normal_bounded(p, p/2, 0.0, 0.5)
            for gram in self.vocab_3grams:
                p = 0.04
                if gram in ["mov_xor_xor", "test_je_jmp"]:
                    p = 0.14
                rec[f"opcode_3gram_{gram}"] = normal_bounded(p, p/2, 0.0, 0.3)

            # YARA signatures (high)
            rec["yara_suspicious_apis"] = int(np.random.random() < 0.75)
            rec["yara_anti_debugging"] = int(np.random.random() < 0.55)
            rec["yara_common_packers"] = int(np.random.random() < 0.40)
            rec["yara_suspicious_network_strings"] = int(np.random.random() < 0.48)

            # Generate synthetic API sequence for malicious
            n_imports = random.randint(3, 20)
            malicious_api_pool = list(range(2, 18))
            api_seq = [float(random.choice(malicious_api_pool)) for _ in range(n_imports)]
            if len(api_seq) >= 3 and random.random() < 0.7:
                idx = random.randint(0, len(api_seq) - 3)
                api_seq[idx] = 2.0
                api_seq[idx+1] = 4.0
                api_seq[idx+2] = 5.0
            api_seq = api_seq + [0.0] * (50 - len(api_seq))
            for i, val in enumerate(api_seq):
                rec[f"api_seq_{i}"] = val

            # Generate synthetic raw bytes for malicious
            byte_seq = [0.0] * 1024
            byte_seq[0] = 77.0 / 255.0 # M
            byte_seq[1] = 90.0 / 255.0 # Z
            for i in range(2, 64):
                byte_seq[i] = float(random.randint(0, 150)) / 255.0
            for i in range(64, 1024):
                byte_seq[i] = float(random.randint(0, 255)) / 255.0
            for i, val in enumerate(byte_seq):
                rec[f"byte_{i}"] = val

            rec["label"] = 1
            records.append(rec)

        # Mix and return
        random.shuffle(records)
        df = pd.DataFrame(records)
        df = df[schema + ["label"]]
        return df

    def split_and_save(self, df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[str, str, str]:
        """
        Splits dataset into Train/Val/Test (70/15/15) stratified by label, 
        saves to data/processed as CSV and Parquet formats.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        logger.info(f"Splitting dataset: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
        
        # Split features and target
        X = df.drop(columns=["label"])
        y = df["label"]
        
        # Stratified splits
        # First split train off
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=42
        )
        
        # Split temp into val and test
        val_relative_ratio = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - val_relative_ratio), stratify=y_temp, random_state=42
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        # File paths
        train_csv = os.path.join(self.processed_dir, "train.csv")
        val_csv = os.path.join(self.processed_dir, "val.csv")
        test_csv = os.path.join(self.processed_dir, "test.csv")

        train_pq = os.path.join(self.processed_dir, "train.parquet")
        val_pq = os.path.join(self.processed_dir, "val.parquet")
        test_pq = os.path.join(self.processed_dir, "test.parquet")

        # Save to CSV
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        # Save to Parquet
        train_df.to_parquet(train_pq, index=False)
        val_df.to_parquet(val_pq, index=False)
        test_df.to_parquet(test_pq, index=False)

        logger.info(f"Dataset split complete. Train shape: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        return self.processed_dir, str(len(train_df)), str(len(val_df) + len(test_df))

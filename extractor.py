import os
import math
import logging
from typing import Dict, Any, List, Optional
import pefile
import yara
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64, CsError

logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class PEFeatureExtractor:
    """
    Statically extracts security-relevant features from Windows Portable Executable (PE) files.
    Features include PE Headers, Section Entropy, Imports/Exports, Opcode N-grams (via Capstone), 
    and signature matches (via YARA).
    """

    def __init__(self, yara_rules_path: Optional[str] = None):
        # Load and compile YARA rules
        self.yara_rules = None
        if yara_rules_path and os.path.exists(yara_rules_path):
            try:
                self.yara_rules = yara.compile(filepath=yara_rules_path)
                logger.info(f"YARA rules compiled successfully from {yara_rules_path}")
            except Exception as e:
                logger.error(f"Failed to compile YARA rules from {yara_rules_path}: {e}")
        else:
            # Fallback to local rules.yar if it exists in the same directory
            local_rules = os.path.join(os.path.dirname(__file__), "rules.yar")
            if os.path.exists(local_rules):
                try:
                    self.yara_rules = yara.compile(filepath=local_rules)
                    logger.info(f"YARA rules compiled successfully from default {local_rules}")
                except Exception as e:
                    logger.error(f"Failed to compile default YARA rules: {e}")

        # Define vocabulary for Opcode N-grams
        self.vocab_mnemonics = [
            "mov", "push", "pop", "call", "jmp", "xor", "add", "sub", "cmp", "test", "je", "jne", "ret", "lea"
        ]
        self.vocab_2grams = [
            "push_mov", "mov_mov", "xor_xor", "mov_call", "call_jmp", "cmp_je", "cmp_jne", 
            "test_je", "test_jne", "push_push", "pop_ret", "add_ret"
        ]
        self.vocab_3grams = [
            "push_mov_sub", "mov_xor_xor", "cmp_je_jmp", "push_push_call", "mov_mov_mov", 
            "test_je_jmp", "mov_call_pop"
        ]

        # Define critical suspicious imports to check explicitly
        self.suspicious_imports = [
            "VirtualAlloc", "VirtualProtect", "WriteProcessMemory", "CreateRemoteThread",
            "QueueUserAPC", "SetThreadContext", "NtCreateThreadEx", "LdrLoadDll",
            "GetProcAddress", "LoadLibraryA", "LoadLibraryW", "InternetOpenA", 
            "InternetOpenW", "HttpOpenRequestA", "HttpOpenRequestW", "ShellExecuteA"
        ]

        # Fixed vocabulary for API sequence encoding (vocab size = 32)
        self.api_vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
        }
        for idx, imp in enumerate(self.suspicious_imports, start=2):
            self.api_vocab[imp.lower()] = idx
            
        common_apis = [
            "ExitProcess", "Sleep", "GetCommandLineA", "GetCommandLineW", 
            "GetVersion", "GetVersionExA", "GetVersionExW", "RegOpenKeyExA", 
            "RegOpenKeyExW", "RegSetValueExA", "RegSetValueExW", "RegCloseKey",
            "GetCurrentProcess", "CloseHandle"
        ]
        for idx, imp in enumerate(common_apis, start=18):
            self.api_vocab[imp.lower()] = idx

    def calculate_entropy(self, data: bytes) -> float:
        """Calculates the Shannon entropy of a byte sequence."""
        if not data:
            return 0.0
        entropy = 0.0
        counts = [0] * 256
        for byte in data:
            counts[byte] += 1
        for count in counts:
            if count > 0:
                p = count / len(data)
                entropy -= p * math.log2(p)
        return entropy

    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a PE file and extracts a comprehensive feature dictionary.
        If parsing fails, returns a dictionary with 'error' set.
        """
        features: Dict[str, Any] = {}
        
        if not os.path.exists(file_path):
            return {"error": f"File does not exist: {file_path}"}
            
        try:
            # Initialize API sequence and raw byte features
            for i in range(50):
                features[f"api_seq_{i}"] = 0.0
            for i in range(1024):
                features[f"byte_{i}"] = 0.0

            # File metadata
            file_size = os.path.getsize(file_path)
            features["file_size"] = file_size
            
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            features["file_entropy"] = self.calculate_entropy(file_bytes)
            
            # Load PE file
            pe = pefile.PE(file_path)
        except pefile.PEFormatError as e:
            return {"error": f"Not a valid PE file: {e}"}
        except Exception as e:
            return {"error": f"Failed to read/parse PE file: {e}"}

        try:
            # 1. PE Header Features
            features["header_machine"] = pe.FILE_HEADER.Machine
            features["header_number_of_sections"] = pe.FILE_HEADER.NumberOfSections
            features["header_time_date_stamp"] = pe.FILE_HEADER.TimeDateStamp
            features["header_characteristics"] = pe.FILE_HEADER.Characteristics
            
            # Optional Header Features
            if hasattr(pe, "OPTIONAL_HEADER"):
                features["opt_magic"] = pe.OPTIONAL_HEADER.Magic
                features["opt_major_linker_version"] = pe.OPTIONAL_HEADER.MajorLinkerVersion
                features["opt_minor_linker_version"] = pe.OPTIONAL_HEADER.MinorLinkerVersion
                features["opt_size_of_code"] = pe.OPTIONAL_HEADER.SizeOfCode
                features["opt_size_of_init_data"] = pe.OPTIONAL_HEADER.SizeOfInitializedData
                features["opt_size_of_uninit_data"] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
                features["opt_address_of_entry_point"] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
                features["opt_base_of_code"] = pe.OPTIONAL_HEADER.BaseOfCode
                features["opt_image_base"] = pe.OPTIONAL_HEADER.ImageBase
                features["opt_section_alignment"] = pe.OPTIONAL_HEADER.SectionAlignment
                features["opt_file_alignment"] = pe.OPTIONAL_HEADER.FileAlignment
                features["opt_major_os_version"] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
                features["opt_minor_os_version"] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
                features["opt_major_image_version"] = pe.OPTIONAL_HEADER.MajorImageVersion
                features["opt_minor_image_version"] = pe.OPTIONAL_HEADER.MinorImageVersion
                features["opt_major_subsystem_version"] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
                features["opt_minor_subsystem_version"] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
                features["opt_size_of_image"] = pe.OPTIONAL_HEADER.SizeOfImage
                features["opt_size_of_headers"] = pe.OPTIONAL_HEADER.SizeOfHeaders
                features["opt_check_sum"] = pe.OPTIONAL_HEADER.CheckSum
                features["opt_subsystem"] = pe.OPTIONAL_HEADER.Subsystem
                features["opt_dll_characteristics"] = pe.OPTIONAL_HEADER.DllCharacteristics
                features["opt_size_of_stack_reserve"] = pe.OPTIONAL_HEADER.SizeOfStackReserve
                features["opt_size_of_stack_commit"] = pe.OPTIONAL_HEADER.SizeOfStackCommit
                features["opt_size_of_heap_reserve"] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
                features["opt_size_of_heap_commit"] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
                features["opt_loader_flags"] = pe.OPTIONAL_HEADER.LoaderFlags
                features["opt_num_rva_and_sizes"] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes
            else:
                # Fill missing optional headers with 0
                for opt_field in ["opt_magic", "opt_major_linker_version", "opt_minor_linker_version",
                                  "opt_size_of_code", "opt_size_of_init_data", "opt_size_of_uninit_data",
                                  "opt_address_of_entry_point", "opt_base_of_code", "opt_image_base",
                                  "opt_section_alignment", "opt_file_alignment", "opt_major_os_version",
                                  "opt_minor_os_version", "opt_major_image_version", "opt_minor_image_version",
                                  "opt_major_subsystem_version", "opt_minor_subsystem_version",
                                  "opt_size_of_image", "opt_size_of_headers", "opt_check_sum", "opt_subsystem",
                                  "opt_dll_characteristics", "opt_size_of_stack_reserve", "opt_size_of_stack_commit",
                                  "opt_size_of_heap_reserve", "opt_size_of_heap_commit", "opt_loader_flags",
                                  "opt_num_rva_and_sizes"]:
                    features[opt_field] = 0

            # 2. Section Details & Entropy
            entropies = []
            raw_sizes = []
            virtual_sizes = []
            exec_sections_count = 0
            write_sections_count = 0
            
            for section in pe.sections:
                sec_data = section.get_data()
                sec_entropy = self.calculate_entropy(sec_data)
                entropies.append(sec_entropy)
                raw_sizes.append(section.SizeOfRawData)
                virtual_sizes.append(section.Misc_VirtualSize)
                
                # Check section characteristics
                # IMAGE_SCN_MEM_EXECUTE = 0x20000000
                # IMAGE_SCN_MEM_WRITE   = 0x80000000
                if section.Characteristics & 0x20000000:
                    exec_sections_count += 1
                if section.Characteristics & 0x80000000:
                    write_sections_count += 1

            if entropies:
                features["sec_entropy_min"] = min(entropies)
                features["sec_entropy_max"] = max(entropies)
                features["sec_entropy_mean"] = sum(entropies) / len(entropies)
                features["sec_entropy_var"] = sum((x - features["sec_entropy_mean"])**2 for x in entropies) / len(entropies)
            else:
                features["sec_entropy_min"] = features["sec_entropy_max"] = features["sec_entropy_mean"] = features["sec_entropy_var"] = 0.0

            if raw_sizes:
                features["sec_raw_size_min"] = min(raw_sizes)
                features["sec_raw_size_max"] = max(raw_sizes)
                features["sec_raw_size_mean"] = sum(raw_sizes) / len(raw_sizes)
                features["sec_raw_size_var"] = sum((x - features["sec_raw_size_mean"])**2 for x in raw_sizes) / len(raw_sizes)
            else:
                features["sec_raw_size_min"] = features["sec_raw_size_max"] = features["sec_raw_size_mean"] = features["sec_raw_size_var"] = 0.0

            if virtual_sizes:
                features["sec_virtual_size_min"] = min(virtual_sizes)
                features["sec_virtual_size_max"] = max(virtual_sizes)
                features["sec_virtual_size_mean"] = sum(virtual_sizes) / len(virtual_sizes)
                features["sec_virtual_size_var"] = sum((x - features["sec_virtual_size_mean"])**2 for x in virtual_sizes) / len(virtual_sizes)
            else:
                features["sec_virtual_size_min"] = features["sec_virtual_size_max"] = features["sec_virtual_size_mean"] = features["sec_virtual_size_var"] = 0.0

            features["sec_executable_count"] = exec_sections_count
            features["sec_writable_count"] = write_sections_count

            # 3. Imports & Exports
            imported_dlls_count = 0
            imported_funcs_count = 0
            
            # Initialize suspicious import features
            for imp_name in self.suspicious_imports:
                features[f"import_{imp_name}"] = 0

            if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                imported_dlls_count = len(pe.DIRECTORY_ENTRY_IMPORT)
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    for imp in entry.imports:
                        imported_funcs_count += 1
                        if imp.name:
                            imp_name_decoded = imp.name.decode("utf-8", errors="ignore")
                            for target_imp in self.suspicious_imports:
                                if imp_name_decoded.lower() == target_imp.lower():
                                    features[f"import_{target_imp}"] = 1

            features["imports_dll_count"] = imported_dlls_count
            features["imports_func_count"] = imported_funcs_count

            # Exports count
            exports_count = 0
            if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
                exports_count = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
            features["exports_count"] = exports_count

            # 4. Disassembly and Opcode N-grams (Capstone)
            # Find the section containing the Entry Point, or fall back to the first executable section
            code_bytes = b""
            ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint if hasattr(pe, "OPTIONAL_HEADER") else 0
            
            # Find section for EP
            ep_section = None
            for section in pe.sections:
                if section.VirtualAddress <= ep < section.VirtualAddress + section.Misc_VirtualSize:
                    ep_section = section
                    break
            
            if ep_section:
                offset_in_sec = ep - ep_section.VirtualAddress
                sec_data = ep_section.get_data()
                # Extract up to 4096 bytes from entry point
                code_bytes = sec_data[offset_in_sec : offset_in_sec + 4096]
            
            # If entry point section couldn't be parsed or was empty, search for any executable section
            if not code_bytes:
                for section in pe.sections:
                    if section.Characteristics & 0x20000000: # Executable
                        code_bytes = section.get_data()[:4096]
                        break
            
            # Initialize all opcode features to 0.0
            for op in self.vocab_mnemonics:
                features[f"opcode_freq_{op}"] = 0.0
            for gram in self.vocab_2grams:
                features[f"opcode_2gram_{gram}"] = 0.0
            for gram in self.vocab_3grams:
                features[f"opcode_3gram_{gram}"] = 0.0
            features["opcode_total_count"] = 0

            # Perform disassembly
            if code_bytes:
                # Check machine mode (32-bit vs 64-bit)
                # IMAGE_FILE_MACHINE_I386 = 0x014c (32-bit x86)
                # IMAGE_FILE_MACHINE_AMD64 = 0x8664 (64-bit x64)
                mode = CS_MODE_64 if features["header_machine"] == 0x8664 else CS_MODE_32
                try:
                    md = Cs(CS_ARCH_X86, mode)
                    instructions = list(md.disasm(code_bytes, ep))
                    
                    mnemonics = [insn.mnemonic for insn in instructions]
                    features["opcode_total_count"] = len(mnemonics)
                    
                    if mnemonics:
                        # 1-gram counts
                        for op in self.vocab_mnemonics:
                            cnt = mnemonics.count(op)
                            features[f"opcode_freq_{op}"] = cnt / len(mnemonics)
                            
                        # 2-grams
                        twograms = [f"{mnemonics[i]}_{mnemonics[i+1]}" for i in range(len(mnemonics) - 1)]
                        for gram in self.vocab_2grams:
                            cnt = twograms.count(gram)
                            features[f"opcode_2gram_{gram}"] = cnt / len(twograms) if twograms else 0.0
                            
                        # 3-grams
                        threegrams = [f"{mnemonics[i]}_{mnemonics[i+1]}_{mnemonics[i+2]}" for i in range(len(mnemonics) - 2)]
                        for gram in self.vocab_3grams:
                            cnt = threegrams.count(gram)
                            features[f"opcode_3gram_{gram}"] = cnt / len(threegrams) if threegrams else 0.0
                except CsError as e:
                    logger.warning(f"Capstone disassembly failed for {file_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error during disassembly of {file_path}: {e}")

            # 5. YARA signatures
            features["yara_suspicious_apis"] = 0
            features["yara_anti_debugging"] = 0
            features["yara_common_packers"] = 0
            features["yara_suspicious_network_strings"] = 0
            
            if self.yara_rules:
                try:
                    matches = self.yara_rules.match(filepath=file_path)
                    for match in matches:
                        rule_name = match.rule
                        if rule_name == "suspicious_apis":
                            features["yara_suspicious_apis"] = 1
                        elif rule_name == "anti_debugging":
                            features["yara_anti_debugging"] = 1
                        elif rule_name == "common_packers":
                            features["yara_common_packers"] = 1
                        elif rule_name == "suspicious_network_strings":
                            features["yara_suspicious_network_strings"] = 1
                except Exception as e:
                    logger.warning(f"YARA matching failed for {file_path}: {e}")

            # 6. Extract raw bytes (first 1024 bytes)
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read(1024)
                file_bytes_padded = file_bytes + b"\x00" * (1024 - len(file_bytes))
                for i in range(1024):
                    features[f"byte_{i}"] = float(file_bytes_padded[i]) / 255.0
            except Exception as e:
                logger.warning(f"Failed to extract raw bytes from {file_path}: {e}")

            # 7. Extract API sequence (up to 50 API imports)
            api_seq = []
            if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    for imp in entry.imports:
                        if imp.name:
                            try:
                                imp_name = imp.name.decode("utf-8", errors="ignore").lower()
                                if imp_name in self.api_vocab:
                                    api_seq.append(float(self.api_vocab[imp_name]))
                                else:
                                    api_seq.append(1.0)
                            except Exception:
                                api_seq.append(1.0)

            # Pad/truncate API sequence to length 50
            if len(api_seq) > 50:
                api_seq = api_seq[:50]
            else:
                api_seq = api_seq + [0.0] * (50 - len(api_seq))
            for i, val in enumerate(api_seq):
                features[f"api_seq_{i}"] = val

            pe.close()
            
        except Exception as e:
            # Handle general internal parsing issues
            logger.error(f"General failure extracting features from {file_path}: {e}")
            return {"error": f"General extraction failure: {e}"}

        return features

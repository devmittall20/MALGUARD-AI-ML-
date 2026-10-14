import os
import tempfile
import pytest
from src.extraction.extractor import PEFeatureExtractor

def test_entropy_calculation():
    extractor = PEFeatureExtractor()
    # Uniform distribution (all bytes 0..255 once)
    all_bytes = bytes(range(256))
    # Shannon entropy of 256 unique symbols should be 8.0 bits
    assert abs(extractor.calculate_entropy(all_bytes) - 8.0) < 1e-5
    
    # Single value sequence entropy should be 0.0 bits
    single_val_bytes = b"\x00" * 100
    assert extractor.calculate_entropy(single_val_bytes) == 0.0

def test_non_existent_file():
    extractor = PEFeatureExtractor()
    res = extractor.extract_features("non_existent_file_xyz.exe")
    assert "error" in res
    assert "does not exist" in res["error"]

def test_invalid_pe_file():
    extractor = PEFeatureExtractor()
    # Create a dummy text file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Hello world, this is a plain text file not a PE format.")
        tmp_name = f.name
        
    try:
        res = extractor.extract_features(tmp_name)
        assert "error" in res
        assert "Not a valid PE file" in res["error"]
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_real_pe_file_extraction():
    extractor = PEFeatureExtractor()
    # Locate a common Windows system file that is guaranteed to be a valid PE file
    possible_paths = [
        r"C:\Windows\System32\cmd.exe",
        r"C:\Windows\System32\notepad.exe",
        r"C:\Windows\System32\kernel32.dll"
    ]
    
    target_path = None
    for path in possible_paths:
        if os.path.exists(path):
            target_path = path
            break
            
    if target_path is None:
        pytest.skip("No common Windows PE files found in C:\\Windows\\System32; skipping real PE test.")
        
    res = extractor.extract_features(target_path)
    
    # Assert features were parsed successfully
    assert "error" not in res
    assert "file_size" in res
    assert res["file_size"] > 0
    assert "file_entropy" in res
    assert 0.0 <= res["file_entropy"] <= 8.0
    
    # Headers
    assert "header_machine" in res
    assert res["header_machine"] in [0x014c, 0x8664] # x86 or x64
    assert "header_number_of_sections" in res
    assert res["header_number_of_sections"] > 0
    
    # Sections
    assert "sec_entropy_max" in res
    assert "sec_executable_count" in res
    assert res["sec_executable_count"] >= 0
    
    # Imports
    assert "imports_dll_count" in res
    assert res["imports_dll_count"] >= 0
    
    # Opcodes (should disassemble correctly if section exists and code is readable)
    assert "opcode_total_count" in res
    assert res["opcode_total_count"] >= 0

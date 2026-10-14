import os
import shutil
import tempfile
import pytest
import pandas as pd
from src.data.dataset import DatasetManager

@pytest.fixture
def temp_data_dir():
    # Setup temporary directory for test outputs
    d = tempfile.mkdtemp()
    yield d
    # Teardown
    if os.path.exists(d):
        shutil.rmtree(d)

def test_feature_schema(temp_data_dir):
    manager = DatasetManager(data_root=temp_data_dir)
    schema = manager.get_feature_schema()
    
    # Assert critical columns exist in the feature schema
    assert "file_size" in schema
    assert "file_entropy" in schema
    assert "header_machine" in schema
    assert "sec_entropy_max" in schema
    assert "imports_dll_count" in schema
    assert "yara_anti_debugging" in schema
    
    # Check dynamic import & opcode vocab columns exist
    assert "import_VirtualAlloc" in schema
    assert "opcode_freq_mov" in schema
    assert "opcode_2gram_push_mov" in schema
    assert "opcode_3gram_push_mov_sub" in schema

def test_synthetic_data_generation(temp_data_dir):
    manager = DatasetManager(data_root=temp_data_dir)
    
    # Generate a small synthetic dataset of 100 samples
    df = manager.generate_synthetic_dataset(n_samples=100, benign_ratio=0.6)
    
    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "label" in df.columns
    
    # Verify label distribution (60 benign, 40 malicious)
    labels = df["label"].value_counts()
    assert labels[0] == 60
    assert labels[1] == 40
    
    # Verify columns match schema plus label
    expected_cols = sorted(manager.get_feature_schema() + ["label"])
    actual_cols = sorted(list(df.columns))
    assert expected_cols == actual_cols

def test_split_and_save(temp_data_dir):
    manager = DatasetManager(data_root=temp_data_dir)
    df = manager.generate_synthetic_dataset(n_samples=200, benign_ratio=0.5)
    
    # Split and save
    proc_dir, train_sz, val_test_sz = manager.split_and_save(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    
    # Verify files created
    assert os.path.exists(os.path.join(proc_dir, "train.csv"))
    assert os.path.exists(os.path.join(proc_dir, "val.csv"))
    assert os.path.exists(os.path.join(proc_dir, "test.csv"))
    
    assert os.path.exists(os.path.join(proc_dir, "train.parquet"))
    assert os.path.exists(os.path.join(proc_dir, "val.parquet"))
    assert os.path.exists(os.path.join(proc_dir, "test.parquet"))
    
    # Read back and check size and balance
    train_df = pd.read_parquet(os.path.join(proc_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(proc_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(proc_dir, "test.parquet"))
    
    # 70% of 200 = 140
    assert len(train_df) == 140
    # 15% of 200 = 30
    assert len(val_df) == 30
    assert len(test_df) == 30
    
    # Verify stratification (balance is preserved)
    assert train_df["label"].value_counts()[0] == 70
    assert train_df["label"].value_counts()[1] == 70
    
    assert val_df["label"].value_counts()[0] == 15
    assert val_df["label"].value_counts()[1] == 15
    
    assert test_df["label"].value_counts()[0] == 15
    assert test_df["label"].value_counts()[1] == 15

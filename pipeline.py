import os
import argparse
import sys
import logging
import pandas as pd
from .dataset import DatasetManager

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("malguard_xai.pipeline")

def parse_args():
    parser = argparse.ArgumentParser(
        description="MalGuard-XAI Stage 1 Pipeline: Feature Extraction, Generation, and Splitter"
    )
    parser.add_argument(
        "--action", 
        choices=["extract", "generate", "run-all"], 
        default="run-all",
        help="Action to perform: extract (scan raw files), generate (generate synthetic), run-all (extract and generate)"
    )
    parser.add_argument(
        "--synthetic-samples", 
        type=int, 
        default=5000,
        help="Number of synthetic samples to generate if 'generate' or 'run-all' is selected"
    )
    parser.add_argument(
        "--benign-ratio", 
        type=float, 
        default=0.5,
        help="Ratio of benign files to generate/balance"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--yara-rules", 
        type=str, 
        default=None,
        help="Path to custom YARA rules file"
    )
    parser.add_argument(
        "--train-ratio", 
        type=float, 
        default=0.70,
        help="Train split ratio"
    )
    parser.add_argument(
        "--val-ratio", 
        type=float, 
        default=0.15,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio", 
        type=float, 
        default=0.15,
        help="Test split ratio"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure directories are absolute or resolved from current working dir
    data_dir = os.path.abspath(args.data_dir)
    yara_rules = os.path.abspath(args.yara_rules) if args.yara_rules else None
    
    logger.info("Initializing DatasetManager...")
    manager = DatasetManager(data_root=data_dir, yara_rules_path=yara_rules)

    df_real = pd.DataFrame()
    df_synth = pd.DataFrame()

    # 1. Feature Extraction (if 'extract' or 'run-all')
    if args.action in ["extract", "run-all"]:
        logger.info("Scanning raw folders and extracting PE features...")
        df_real = manager.scan_and_extract()
        logger.info(f"Extracted {len(df_real)} samples from local raw folders.")

    # 2. Synthetic Generation (if 'generate' or 'run-all')
    if args.action in ["generate", "run-all"]:
        logger.info(f"Generating {args.synthetic_samples} high-fidelity synthetic samples...")
        df_synth = manager.generate_synthetic_dataset(
            n_samples=args.synthetic_samples, 
            benign_ratio=args.benign_ratio
        )
        logger.info(f"Generated {len(df_synth)} synthetic samples.")

    # Combine datasets
    if not df_real.empty and not df_synth.empty:
        logger.info("Combining extracted and synthetic features...")
        combined_df = pd.concat([df_real, df_synth], ignore_index=True)
    elif not df_real.empty:
        combined_df = df_real
    elif not df_synth.empty:
        combined_df = df_synth
    else:
        logger.error("No data collected or generated! Please add files to data/raw/ or enable synthetic generation.")
        sys.exit(1)

    # Clean duplicates if any (only check features, ignore label)
    feat_cols = manager.get_feature_schema()
    combined_df = combined_df.drop_duplicates(subset=feat_cols, keep="first")
    
    logger.info(f"Final combined dataset shape: {combined_df.shape}")
    label_counts = combined_df["label"].value_counts()
    logger.info(f"Label distribution:\n{label_counts.to_string()}")

    # 3. Partitioning and Saving
    logger.info("Splitting dataset and saving files...")
    processed_dir, train_count, test_count = manager.split_and_save(
        combined_df, 
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio
    )
    
    logger.info(f"Successfully processed and split dataset!")
    logger.info(f"Processed features saved to: {processed_dir}")
    logger.info(f"Files saved: train.csv/parquet, val.csv/parquet, test.csv/parquet")

if __name__ == "__main__":
    main()

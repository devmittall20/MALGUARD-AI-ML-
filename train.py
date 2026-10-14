import os
import sys
import argparse
import logging
from .classical_trainer import ModelTrainer
from .deep_trainer import DeepModelTrainer
from .ensemble import PEVotingEnsemble
from .evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("malguard_xai.training_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(
        description="MalGuard-XAI Stage 4 Pipeline: Model Training and Ensemble Evaluation"
    )
    parser.add_argument(
        "--action", 
        choices=["train-all", "eval-only"], 
        default="train-all",
        help="Action to perform: train-all (train sub-models, build ensemble, evaluate) or eval-only (evaluate existing)"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/processed",
        help="Path to processed features directory"
    )
    parser.add_argument(
        "--models-dir", 
        type=str, 
        default="models",
        help="Path to models directory"
    )
    parser.add_argument(
        "--reports-dir", 
        type=str, 
        default="reports",
        help="Path to reports directory"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    data_dir = os.path.abspath(args.data_dir)
    models_dir = os.path.abspath(args.models_dir)
    reports_dir = os.path.abspath(args.reports_dir)

    # Initialize evaluator
    evaluator = ModelEvaluator(data_dir=data_dir, reports_dir=reports_dir)
    
    # Declare dictionary to store evaluation results
    results = {}

    try:
        if args.action == "train-all":
            logger.info("=== Starting Classical Model Training & GridSearchCV Pipeline ===")
            
            # Initialize trainer
            trainer = ModelTrainer(data_dir=data_dir, models_dir=models_dir)
            
            # Load training & validation data (static features only)
            X_train, y_train, X_val, y_val = trainer.load_data()
            
            # Train classical models (load if exists to save time)
            import joblib
            rf_path = os.path.join(models_dir, "rf_model.pkl")
            xgb_path = os.path.join(models_dir, "xgb_model.pkl")
            
            if os.path.exists(rf_path):
                logger.info("Loaded pre-trained Random Forest model.")
                rf_model = joblib.load(rf_path)
            else:
                rf_model = trainer.train_random_forest(X_train, y_train)
                
            if os.path.exists(xgb_path):
                logger.info("Loaded pre-trained XGBoost model.")
                xgb_model = joblib.load(xgb_path)
            else:
                xgb_model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
                
            gb_model = None  # Skipped to save CPU training time on EMBER

            # Train PyTorch deep learning models (load if exists to save time)
            logger.info("=== Starting PyTorch Deep Learning Model Training Pipeline ===")
            lstm_path = os.path.join(models_dir, "lstm_model.pth")
            cnn_path = os.path.join(models_dir, "cnn_model.pth")
            
            deep_trainer = DeepModelTrainer(data_dir=data_dir, models_dir=models_dir)
            
            import torch
            if os.path.exists(lstm_path) and os.path.exists(cnn_path):
                logger.info("Loaded pre-trained PyTorch deep learning models.")
                from .deep_models import APISequenceLSTM, RawByteCNN
                lstm_model = APISequenceLSTM().to(deep_trainer.device)
                lstm_model.load_state_dict(torch.load(lstm_path, map_location=deep_trainer.device))
                
                cnn_model = RawByteCNN().to(deep_trainer.device)
                cnn_model.load_state_dict(torch.load(cnn_path, map_location=deep_trainer.device))
            else:
                deep_data = deep_trainer.load_multimodal_data()
                lstm_model = deep_trainer.train_lstm(deep_data)
                cnn_model = deep_trainer.train_cnn(deep_data)

            # Build and save voting ensemble
            logger.info("Building soft voting ensemble...")
            ensemble = PEVotingEnsemble(
                rf_model=rf_model, 
                xgb_model=xgb_model, 
                gb_model=gb_model,
                lstm_model=lstm_model,
                cnn_model=cnn_model
            )
            ensemble_path = os.path.join(models_dir, "ensemble_model.pkl")
            ensemble.save(ensemble_path)
            logger.info(f"Ensemble model saved to {ensemble_path}")

        elif args.action == "eval-only":
            logger.info("=== Running Evaluation on Existing Models ===")
            
            # Load models from directory
            ensemble_path = os.path.join(models_dir, "ensemble_model.pkl")
            if not os.path.exists(ensemble_path):
                logger.error(f"Ensemble model not found at {ensemble_path}. Run train-all first.")
                sys.exit(1)
            
            logger.info("Loading ensemble and sub-models...")
            ensemble = PEVotingEnsemble.load(ensemble_path)
            # Re-ensure submodels are loaded inside
            ensemble.load_submodels(models_dir)

        # === Evaluation Phase ===
        logger.info("=== Loading Test Dataset for Scoring ===")
        X_test, y_test = evaluator.load_test_data()

        # Score individual classical models
        results["Random Forest"] = evaluator.evaluate_model(ensemble.rf_model, "Random Forest", X_test, y_test)
        results["XGBoost"] = evaluator.evaluate_model(ensemble.xgb_model, "XGBoost", X_test, y_test)
        if ensemble.gb_model is not None:
            results["Gradient Boosting"] = evaluator.evaluate_model(ensemble.gb_model, "Gradient Boosting", X_test, y_test)
        
        # Score individual PyTorch models
        if ensemble.lstm_model is not None:
            results["LSTM API Sequence"] = evaluator.evaluate_model(ensemble.lstm_model, "LSTM API Sequence", X_test, y_test)
        if ensemble.cnn_model is not None:
            results["Raw Byte CNN"] = evaluator.evaluate_model(ensemble.cnn_model, "Raw Byte CNN", X_test, y_test)
        
        # Score voting ensemble
        results["Voting Ensemble"] = evaluator.evaluate_model(ensemble, "Voting Ensemble", X_test, y_test)

        # Save metrics to JSON file
        metrics_json_path = os.path.join(reports_dir, "model_metrics.json")
        evaluator.save_metrics_to_json(results, metrics_json_path)

        # Plot and save curves/matrices
        roc_path = evaluator.plot_roc_curves(results, y_test)
        cm_path = evaluator.plot_confusion_matrices(results, y_test)

        # Print final console summary table
        print("\n" + "=" * 90)
        print(f"{'Classifier Comparison Metrics (Test Set)':^90}")
        print("=" * 90)
        print(f"{'Model Name':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
        print("-" * 90)
        
        for name in results.keys():
            res = results[name]
            print(f"{name:<20} | {res['accuracy']:<10.4f} | {res['precision']:<10.4f} | {res['recall']:<10.4f} | {res['f1_score']:<10.4f} | {res['roc_auc']:<10.4f}")
            
        print("=" * 90)
        print(f"ROC Curves saved to: {roc_path}")
        print(f"Confusion Matrices saved to: {cm_path}")
        print(f"JSON Performance report saved to: {metrics_json_path}")
        print("=" * 90 + "\n")

    except Exception as e:
        logger.exception(f"Training pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

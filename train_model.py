# # train_model.py
# import pandas as pd
# import xgboost as xgb
# from sklearn.metrics import classification_report, confusion_matrix
# import pickle
# from data_preprocessing import load_and_preprocess_data  # Import your preprocessing function
# from sklearn.model_selection import GridSearchCV
# # --- Load Preprocessed Data ---
# X_train_res, X_test, y_train_res, y_test = load_and_preprocess_data()

# # --- Train XGBoost Model ---
# model = xgb.XGBClassifier(
#     scale_pos_weight=100,  # Penalize fraud class (adjust based on your dataset)
#     objective="binary:logistic",
#     eval_metric="aucpr",  # Better for imbalanced data than 'auc'
#     n_estimators=200,
#     max_depth=5,
#     subsample=0.8,
#     random_state=42
# )

# model.fit(X_train_res, y_train_res)

# # --- Evaluate Model ---
# y_pred = model.predict(X_test)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # --- Save Model ---
# with open("fraud_detection_model.pkl", "wb") as f:
#     pickle.dump(model, f)
# print("Model saved as 'fraud_detection_model.pkl'")

# param_grid = {
#     "max_depth": [3, 5, 7],
#     "learning_rate": [0.01, 0.1],
#     "scale_pos_weight": [50, 100, 200]  # Test different weights
# }

# grid = GridSearchCV(model, param_grid, scoring="precision", cv=3)
# grid.fit(X_train_res, y_train_res)
# print("Best params:", grid.best_params_)
# train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
import pickle
import logging
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data
from sklearn.model_selection import GridSearchCV, train_test_split

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def train_and_evaluate_model():
    """Main function to train and evaluate XGBoost model."""
    try:
        # --- Load Data ---
        logger.info("Loading and preprocessing data...")
        X_train_res, X_test, y_train_res, y_test = load_and_preprocess_data()

        # --- Base Model Training ---
        logger.info("Training base XGBoost model...")
        base_model = xgb.XGBClassifier(
            scale_pos_weight=100,
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=200,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        base_model.fit(X_train_res, y_train_res)

        # --- Base Model Evaluation ---
        logger.info("Evaluating base model...")
        evaluate_model(base_model, X_test, y_test, "Base Model")

        # --- Hyperparameter Tuning ---
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "scale_pos_weight": [50, 100, 200]
        }
        
        grid = GridSearchCV(
            estimator=xgb.XGBClassifier(objective="binary:logistic", random_state=42),
            param_grid=param_grid,
            scoring="average_precision",
            cv=3,
            n_jobs=-1
        )
        grid.fit(X_train_res, y_train_res)
        
        logger.info(f"Best parameters: {grid.best_params_}")
        best_model = grid.best_estimator_

        # --- Best Model Evaluation ---
        logger.info("Evaluating tuned model...")
        evaluate_model(best_model, X_test, y_test, "Tuned Model")

        # --- Threshold Tuning ---
        logger.info("Finding optimal threshold...")
        y_scores = best_model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        
        # Find threshold that maximizes F1-score
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
        optimal_idx = f1_scores.argmax()
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        plot_pr_curve(precision, recall, average_precision_score(y_test, y_scores))

        # --- Save Model ---
        logger.info("Saving best model...")
        with open("fraud_detection_model.pkl", "wb") as f:
            pickle.dump({
                'model': best_model,
                'threshold': optimal_threshold
            }, f)
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and log metrics."""
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"\n{model_name} Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    logger.info(f"\n{model_name} Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    logger.info(f"AUPRC: {average_precision_score(y_test, y_scores):.4f}")

def plot_pr_curve(precision, recall, auprc):
    """Plot Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUPRC = {auprc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('pr_curve.png')
    plt.close()

if __name__ == "__main__":
    train_and_evaluate_model()
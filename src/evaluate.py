import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier, plot_importance
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

def load_test_data():
    print("Loading test data...")
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).squeeze()
    return X_test, y_test

def load_models():
    print("Loading trained models...")
    models = {}
    for name in ["Logistic_Regression", "Decision_Tree", "Random_Forest"]:
        models[name] = joblib.load(os.path.join(MODELS_DIR, f'{name}.pkl'))
    xgb = XGBClassifier()
    xgb.load_model(os.path.join(MODELS_DIR, 'XGBoost_Tuned.json'))
    models["XGBoost_Tuned"] = xgb
    return models

def calculate_physics_significance(y_test, y_prob):
    """Finds the optimal threshold that maximizes s/sqrt(b)."""
    thresholds = np.linspace(0.01, 0.99, 100)
    significances = []
    for t in thresholds:
        s = np.sum((y_prob >= t) & (y_test == 1))
        b = np.sum((y_prob >= t) & (y_test == 0))
        z = s / np.sqrt(b) if b > 0 else 0
        significances.append(z)
    max_z = max(significances)
    opt_threshold = thresholds[np.argmax(significances)]
    return thresholds, significances, max_z, opt_threshold

def evaluate_and_plot(models, X_test, y_test):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("Generating ROC Curve comparison...")
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, '01_roc_curve.png'))
    plt.close()

    xgb_model = models["XGBoost_Tuned"]
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    print("Maximizing Physics Discovery Significance...")
    ts, zs, max_z, opt_t = calculate_physics_significance(y_test, y_prob_xgb)
    plt.figure(figsize=(10, 6))
    plt.plot(ts, zs, color='purple', linewidth=2)
    plt.axvline(opt_t, color='black', linestyle='--', label=f'Opt Threshold: {opt_t:.2f}')
    plt.title(f'Physics Discovery Potential (Max Z = {max_z:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Significance (s/sqrt(b))')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, '06_physics_significance.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_importance(xgb_model, ax=ax, max_num_features=10, importance_type='gain', color='teal')
    plt.savefig(os.path.join(PLOTS_DIR, '04_feature_importance.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(y_prob_xgb[y_test == 0], bins=50, alpha=0.5, color='blue', label='Background', density=True)
    plt.hist(y_prob_xgb[y_test == 1], bins=50, alpha=0.5, color='red', label='Signal', density=True)
    plt.title('Model Discriminant Distribution')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, '05_score_distribution.png'))
    plt.close()

    print(f"\n--- SUCCESS ---")
    print(f"Max Significance Z: {max_z:.2f} at Threshold: {opt_t:.2f}")

if __name__ == "__main__":
    X_test, y_test = load_test_data()
    models = load_models()
    evaluate_and_plot(models, X_test, y_test)
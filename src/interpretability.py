import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def run_shap_analysis():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("Loading data and model for SHAP analysis...")
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    model = XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, 'XGBoost_Tuned.json'))
    
    X_sample = X_test.sample(500, random_state=42)
    print("Calculating SHAP values (Beeswarm and Feature Impact)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Physics Feature Impact (Quantum Kinematics)", fontsize=14)
    plt.savefig(os.path.join(PLOTS_DIR, '07_shap_beeswarm.png'), bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(os.path.join(PLOTS_DIR, '08_shap_global_bar.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Interpretability plots 07 and 08 saved to {PLOTS_DIR}! 🔍")

if __name__ == "__main__":
    run_shap_analysis()
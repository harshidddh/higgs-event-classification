import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def load_training_data():
    """Loads the processed training and testing data."""
    print("Loading processed phase space data...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).squeeze()

    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains Baseline and Ensemble models, evaluates them, and saves them."""

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost_Tuned": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, 
                                       use_label_encoder=False, eval_metric='logloss', 
                                       tree_method='hist', random_state=42)
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("\n--- Starting Model Benchmark ---")

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Results for {name}: AUC = {auc:.4f} | Accuracy = {acc:.4f}")

        if name == "XGBoost_Tuned":
            model_path = os.path.join(MODELS_DIR, f'{name}.json')
            model.save_model(model_path)
        else:
            model_path = os.path.join(MODELS_DIR, f'{name}.pkl')
            joblib.dump(model, model_path)
            
    print("\nAll models successfully trained and saved to the models/ directory. 🧠")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_training_data()
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

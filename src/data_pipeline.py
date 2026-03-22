import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'training.csv')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def engineer_physics_features(df):
    """Calculates Invariant Mass and Transverse Mass."""
    print("Reconstructing Physics Features (M and MT)...")
    
    # 1. Invariant Mass
    pt1, eta1, phi1 = df['PRI_tau_pt'], df['PRI_tau_eta'], df['PRI_tau_phi']
    pt2, eta2, phi2 = df['PRI_lep_pt'], df['PRI_lep_eta'], df['PRI_lep_phi']
    
    m_sq = 2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))
    df['phys_inv_mass'] = np.sqrt(np.maximum(m_sq, 0))
    
    # 2. Transverse Mass
    mt_sq = 2 * df['PRI_lep_pt'] * df['PRI_met'] * (1 - np.cos(df['PRI_lep_phi'] - df['PRI_met_phi']))
    df['phys_trans_mass'] = np.sqrt(np.maximum(mt_sq, 0))
    
    return df

def load_and_process():
    print(f"Loading: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH).replace(-999.0, np.nan)
    
    cols_to_drop = df.columns[df.isna().mean() > 0.70].tolist() + ['EventId', 'Weight']
    df = df.drop(columns=cols_to_drop)
    
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    df = engineer_physics_features(df)
    
    df['Label'] = df['Label'].map({'s': 1, 'b': 0})
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    X_train_s.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
    X_test_s.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'standard_scaler.pkl'))
    
    print("Verification: Columns in saved X_test:")
    print(X_test_s.columns.tolist())
    print("\nPipeline Success! 🚀")

if __name__ == "__main__":
    load_and_process()
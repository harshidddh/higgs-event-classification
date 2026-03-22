import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'training.csv')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def calculate_invariant_mass(df):
    """
    Calculates the invariant mass of the tau-tau system using 
    relativistic kinematics.
    M = sqrt(2 * pT1 * pT2 * (cosh(eta1 - eta2) - cos(phi1 - phi2)))
    """
    print("Reconstructing Invariant Mass from raw kinematics...")
    
    pt1, eta1, phi1 = df['PRI_tau_pt'], df['PRI_tau_eta'], df['PRI_tau_phi']
    pt2, eta2, phi2 = df['PRI_lep_pt'], df['PRI_lep_eta'], df['PRI_lep_phi']
    
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2
    
    mass_sq = 2 * pt1 * pt2 * (np.cosh(delta_eta) - np.cos(delta_phi))
    df['reconstructed_inv_mass'] = np.sqrt(np.maximum(mass_sq, 0))
    
    df['reconstructed_transverse_mass'] = np.sqrt(
        2 * df['PRI_lep_pt'] * df['PRI_met'] * (1 - np.cos(df['PRI_lep_phi'] - df['PRI_met_phi']))
    )
    
    return df

if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH).replace(-999.0, np.nan)
    df_enhanced = calculate_invariant_mass(df)
    
    print(df_enhanced[['reconstructed_inv_mass', 'reconstructed_transverse_mass']].head())
    print("\nPhysics features engineered successfully! ⚛️")
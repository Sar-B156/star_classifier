import pandas as pd
import torch

FEATURE_COLUMNS = [
    "Temperature (K)",
    "Luminosity(L/Lo)",
    "Radius(R/Ro)",
    "Absolute magnitude(Mv)",
    "Star color",
    "Spectral Class"
]

def preprocess_input(data_dict, scaler, le_color, le_spectral):
    df = pd.DataFrame([data_dict])

    # pastikan urutan & nama kolom IDENTIK
    df = df[FEATURE_COLUMNS]

    df["Star color"] = le_color.transform(df["Star color"])
    df["Spectral Class"] = le_spectral.transform(df["Spectral Class"])

    X_scaled = scaler.transform(df)
    return torch.tensor(X_scaled, dtype=torch.float32)
import streamlit as st
import joblib
from src.predict import load_model, predict_star
from src.preprocess import preprocess_input







@st.cache_resource
def load_all():
    model = load_model("models/star_classifier.pth")
    scaler = joblib.load("models/scaler.pkl")
    le_color = joblib.load("models/color_encoder.pkl")
    le_spectral = joblib.load("models/spectral_encoder.pkl")
    return model, scaler, le_color, le_spectral

model, scaler, le_color, le_spectral = load_all()



st.title("Star Classification AI")

temp = st.number_input("Temperature (K)", 2000, 40000, 5800)
lum = st.number_input("Luminosity (L/Lo)", 0.0001, 100000.0, 1.0)
radius = st.number_input("Radius (R/Ro)", 0.01, 1000.0, 1.0)
mag = st.number_input("Absolute Magnitude (Mv)", -10.0, 20.0, 4.83)

color = st.selectbox("Star color", le_color.classes_)
spectral = st.selectbox("Spectral Class", le_spectral.classes_)




if st.button("Predict Star Type"):
    input_data = {
        "Temperature (K)": temp,
        "Luminosity(L/Lo)": lum,
        "Radius(R/Ro)": radius,
        "Absolute magnitude(Mv)": mag,
        "Star color": color,
        "Spectral Class": spectral
    }


    X_tensor = preprocess_input(input_data, scaler, le_color, le_spectral)
    result = predict_star(model, X_tensor)

    st.success(f"Predicted Star Type: {result}")

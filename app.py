import streamlit as st
import pandas as pd
import joblib  # ✅ CORRECT SPELLING
import plotly.graph_objects as go

# Load model (FIXED)
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')  # ✅ joblib (not joblit)
    preprocessor = joblib.load('preprocessor.pkl')  # ✅ joblib (not joblit)
    return model, preprocessor

model, preprocessor = load_model()

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("🚀 Customer Churn Predictor")

# Inputs
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 118.0, 70.0)
with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("🔮 Predict Churn", type="primary"):
    # Create input data
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'Contract': [contract],
        'InternetService': [internet_service]
    })
    
    # Predict
    X_scaled = preprocessor.transform(input_data)
    churn_prob = model.predict_proba(X_scaled)[0, 1]
    
    # Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{churn_prob:.1%}")
    with col2:
        if churn_prob > 0.5:
            st.error("❌ **High Risk**")
        else:
            st.success("✅ **Low Risk**")
    
    st.balloons()

st.markdown("**Telco Churn Prediction using XGBoost**")

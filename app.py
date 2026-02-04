import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load('models/churn_model_xgb.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("🚀 Customer Churn Prediction App")
st.markdown("**Predict if a telecom customer will churn** using XGBoost model trained on 7K+ records.")

# Sidebar inputs
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 118.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 8684.0, 1000.0)

# Categorical selections (exact from dataset)
internet_service = st.sidebar.selectbox("Internet Service", 
    ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", 
    ["Month-to-month", "One year", "Two year"])
online_security = st.sidebar.selectbox("Online Security", 
    ["No internet service", "No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", 
    ["No internet service", "No", "Yes"])
streaming_tv = st.sidebar.selectbox("Streaming TV", 
    ["No internet service", "No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", 
     "Credit card (automatic)"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

if st.sidebar.button("🔮 Predict Churn"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'InternetService': [internet_service],
        'Contract': [contract],
        'OnlineSecurity': [online_security],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'PaymentMethod': [payment_method],
        'PaperlessBilling': [paperless_billing]
    })
    
    # Predict
    X_scaled = preprocessor.transform(input_data)
    prob = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]
    
    # Results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if pred == 1:
            st.error("❌ **Will Churn**")
        else:
            st.success("✅ **Will Stay**")
    
    with col2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob*100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)
    
    st.balloons()

# Model info
with st.expander("📊 Model Performance"):
    st.write("""
    - **Models Compared**: Logistic Regression, Random Forest, XGBoost  
    - **Best Model**: XGBoost (Accuracy ~91%, AUC ~0.88)  
    - **Preprocessing**: Label encoding, scaling, SMOTE for imbalance  
    - **Dataset**: 7,043 telecom customers  
    """)

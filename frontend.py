import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="XGBoost Obesity Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfigurasi API Backend
API_BASE_URL = "http://localhost:8000"

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }

    .prediction-card, .metric-card, .success-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
    }

    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .success-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }

    .error-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .recommendation-item {
        background-color: #f8f9fa;
        color: #333333;  
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }

    .sidebar-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .xgboost-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection Error: {str(e)}"

def make_prediction(data):
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return False, error_detail
    except requests.exceptions.RequestException as e:
        return False, f"Request Error: {str(e)}"

def get_risk_color(risk_level):
    colors = {
        "MINIMAL": "#28a745",
        "LOW": "#ffc107", 
        "MODERATE": "#fd7e14",
        "HIGH": "#dc3545",
        "VERY HIGH": "#6f42c1"
    }
    return colors.get(risk_level, "#6c757d")

# Header
st.markdown('<h1 class="main-header">XGBoost Obesity Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Obesity Level Prediction System Based on Lifestyle Using XGBoost</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### 📊 Status XGboost")
    api_status, api_info = check_api_connection()
    if api_status:
        st.success("✅ XGBoost API Connected")
    else:
        st.error("❌ XGBoost API Disconnected")
        st.write(f"Error: {api_info}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 📋 User Guide ")
    st.info("""
    1. Complete all the required fields  
    2. Ensure the information is correct  
    3. Click the 'Predict' button  
    4. Check the results and suggestions
    """)
    st.markdown("### 🏥 BMI Classification ")
    st.write("""
    - **Underweight:** < 18.5
    - **Normal:** 18.5 - 24.9
    - **Overweight:** 25.0 - 29.9
    - **Obese:** ≥ 30.0
    """)

# Form input
st.markdown('<h2 class="sub-header">📝 Health Information Form </h2>', unsafe_allow_html=True)
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PERSONAL INFORMATION")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
        height = st.number_input("Height (meter)", min_value=0.5, max_value=3.0, value=1.70, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
        family_history = st.selectbox("Obesity in family", ["yes", "no"])
        st.subheader(" 🔥 Lifestyle")
        smoke = st.selectbox("Smoke", ["yes", "no"])
        calc = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently"])
        faf = st.slider("Physical Activity Frequency (per week)", 0.0, 3.0, 1.0, step=0.1)
        tue = st.slider("Technology Usage (hours/day)", 0.0, 2.0, 1.0, step=0.1)
    with col2:
        st.subheader("🍽️ Eating Habits")
        favc = st.selectbox("High-Calorie Food Consumption", ["yes", "no"])
        scc = st.selectbox("Calorie Monitoring", ["yes", "no"])  
        fcvc = st.slider("Vegetable Consumption Frequency", 1.0, 3.0, 2.0, step=0.1)
        ncp = st.slider("Main Meals per Day", 1.0, 4.0, 3.0, step=0.1)
        caec = st.selectbox("Snacking Between Meals", ["no", "Sometimes", "Frequently", "Always"])
        ch20 = st.slider("Water Intake per Day (Liters)", 1.0, 3.0, 2.0, step=0.1)
        st.subheader("🚗 Transportation")
        mtrans = st.selectbox("Main Transportation Mode", ["Public_Transportation", "Automobile", "Motorbike", "Bike", "Walking"])
    st.markdown("---")
    submitted = st.form_submit_button("Predict ", use_container_width=True)

# Proses prediksi
if submitted:
    if not api_status:
        st.markdown('<div class="error-card"> Unable to connect to the XGBoost API. Please make sure the backend is running at http://localhost:8000</div>', unsafe_allow_html=True)
    else:
        prediction_data = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history_with_overweight": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch20,  
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }
        with st.spinner('Hold on! XGBoost is working on your prediction...'):
            success, result = make_prediction(prediction_data)
        if success:
            st.markdown('<h2 class="sub-header"> XGBoost Prediction Result</h2>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                prediction_clean = result['prediction'].replace('_', ' ')
                st.markdown(f'''
                <div class="prediction-card">
                    <h3>🎯 XGboost Prediction</h3>
                    <h2>{prediction_clean}</h2>
                    <p>Confidence: {result['confidence']:.2%}</p>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                risk_color = get_risk_color(result['risk_level'])
                st.markdown(f'''
                <div class="metric-card" style="background-color: {risk_color};">
                    <h3>⚠️ Risk Level</h3>
                    <h2>{result['risk_level']}</h2>
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                <div class="success-card">
                    <h3>📏 BMI</h3>
                    <h2>{result['bmi']}</h2>
                    <p>Kategori: {result['bmi_category']}</p>
                </div>
                ''', unsafe_allow_html=True)

            # Tampilkan probabilitas sebagai teks
            st.markdown("### XGBoost Probabilities for Each Category:")
            for category, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {category.replace('_', ' ')}: {prob:.2%}")

            # Rekomendasi
            st.markdown('<h3 class="sub-header">💡 Health Recommendations </h3>', unsafe_allow_html=True)
            for i, recommendation in enumerate(result['recommendations'], 1):
                st.markdown(f'''
                <div class="recommendation-item">
                    <strong>{i}.</strong> {recommendation}
                </div>
                ''', unsafe_allow_html=True)

            # Detail probabilitas dalam tabel
            with st.expander("📈 Detailed Probabilities for each categories"):
                prob_df = pd.DataFrame(
                    list(result['probabilities'].items()),
                    columns=['Kategori', 'Probabilitas']
                )
                prob_df['Probabilitas'] = prob_df['Probabilitas'].apply(lambda x: f"{x:.2%}")
                prob_df['Kategori'] = prob_df['Kategori'].str.replace('_', ' ')
                prob_df = prob_df.sort_values('Probabilitas', ascending=False)
                st.dataframe(prob_df, use_container_width=True)

            # Informasi tambahan
            st.markdown("---")
            st.markdown(f"**Prediction Time : ** {result['timestamp']}")

            # Download hasil
            result_json = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download Results(JSON)",
                data=result_json,
                file_name=f"xgboost_obesity_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.markdown(f'<div class="error-card"> Oops! Something Went Wrong: {result}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>XGBoost Obesity Prediction System | Powered by FastAPI & Streamlit</p>
</div>
""", unsafe_allow_html=True)


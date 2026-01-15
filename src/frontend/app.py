"""
Streamlit dashboard for churn prediction.
Interactive UI for predictions, monitoring, and model insights.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_prediction(customer_data):
    """Get prediction from API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def get_drift_status():
    """Get drift monitoring status."""
    try:
        response = requests.post(
            f"{API_URL}/monitoring/check-drift",
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def main():
    """Main dashboard application."""
    
    # Header
    st.title("🎯 Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 MLOps Churn Prediction")
        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["🔮 Predictions", "📊 Model Insights", "🔍 Monitoring", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### API Status")
        
        if check_api_health():
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
            st.info("Start API with: `uvicorn serving.api:app --reload`")
    
    # Pages
    if page == "🔮 Predictions":
        prediction_page()
    elif page == "📊 Model Insights":
        insights_page()
    elif page == "🔍 Monitoring":
        monitoring_page()
    else:
        about_page()


def prediction_page():
    """Single prediction page."""
    st.header("🔮 Make Predictions")
    
    # Tabs for single vs batch
    tab1, tab2 = st.tabs(["Single Customer", "Batch Upload"])
    
    with tab1:
        st.markdown("### Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demographics")
            customer_id = st.text_input("Customer ID", "CUST-001")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        with col2:
            st.markdown("#### Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        with col3:
            st.markdown("#### Account")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)
        
        st.markdown("---")
        
        if st.button("🎯 Predict Churn", type="primary", use_container_width=True):
            # Prepare data
            customer_data = {
                "customerID": customer_id,
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
            
            with st.spinner("Making prediction..."):
                result = get_prediction(customer_data)
            
            if result:
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Customer ID", result["customerID"])
                
                with col2:
                    churn_prob = result["churn_probability"]
                    st.metric("Churn Probability", f"{churn_prob:.1%}")
                
                with col3:
                    prediction = result["churn_prediction"]
                    color = "🔴" if prediction == "Yes" else "🟢"
                    st.metric("Prediction", f"{color} {prediction}")
                
                with col4:
                    risk = result["risk_level"]
                    risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    st.metric("Risk Level", f"{risk_emoji.get(risk, '')} {risk}")
                
                # Probability gauge
                st.markdown("### Churn Probability")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if churn_prob > 0.7 else "orange" if churn_prob > 0.4 else "green"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                if prediction == "Yes":
                    st.error("⚠️ **High Churn Risk** - Consider retention actions:")
                    st.markdown("""
                    - 📞 Reach out with personalized offer
                    - 💰 Provide loyalty discount
                    - 📱 Upgrade to better service plan
                    - 🎁 Offer additional features at no cost
                    """)
                else:
                    st.success("✅ **Low Churn Risk** - Customer likely to stay")
    
    with tab2:
        st.markdown("### Batch Predictions")
        st.info("Upload a CSV file with customer data for batch predictions")
        
        # Sample CSV download
        if st.button("📥 Download Sample CSV Template"):
            sample_data = pd.DataFrame([{
                "customerID": "SAMPLE-001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20,
            }])
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sample_customers.csv",
                mime="text/csv",
            )
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"📊 Loaded {len(df)} customers")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🎯 Predict All", type="primary"):
                with st.spinner("Making predictions..."):
                    predictions = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        customer_data = row.to_dict()
                        result = get_prediction(customer_data)
                        if result:
                            predictions.append(result)
                        progress_bar.progress((idx + 1) / len(df))
                    
                    if predictions:
                        results_df = pd.DataFrame(predictions)
                        
                        st.success(f"✅ Predicted {len(predictions)} customers")
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            churn_count = (results_df['churn_prediction'] == 'Yes').sum()
                            st.metric("Predicted Churners", churn_count)
                        with col2:
                            avg_prob = results_df['churn_probability'].mean()
                            st.metric("Avg Churn Prob", f"{avg_prob:.1%}")
                        with col3:
                            high_risk = (results_df['risk_level'] == 'High').sum()
                            st.metric("High Risk Customers", high_risk)
                        
                        # Results table
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )


def insights_page():
    """Model insights page."""
    st.header("📊 Model Insights")
    
    # Model info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Champion Model", "CatBoost")
        st.metric("Version", "v1")
    
    with col2:
        st.metric("ROC-AUC Score", "0.8485")
        st.metric("Accuracy", "81.10%")
    
    with col3:
        st.metric("F1 Score", "0.5832")
        st.metric("Precision", "70.28%")
    
    st.markdown("---")
    
    # Model comparison
    st.markdown("### Model Comparison")
    
    model_data = pd.DataFrame({
        'Model': ['CatBoost', 'LightGBM', 'XGBoost'],
        'ROC-AUC': [0.8485, 0.8441, 0.8403],
        'Accuracy': [0.8110, 0.8075, 0.8057],
        'F1 Score': [0.5832, 0.5819, 0.5764],
        'Precision': [0.7028, 0.6864, 0.6835],
        'Recall': [0.4983, 0.5050, 0.4983],
    })
    
    fig = px.bar(
        model_data,
        x='Model',
        y=['ROC-AUC', 'Accuracy', 'F1 Score'],
        barmode='group',
        title='Model Performance Comparison'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(model_data, use_container_width=True, hide_index=True)


def monitoring_page():
    """Monitoring dashboard page."""
    st.header("🔍 Model Monitoring")
    
    # Get drift status
    with st.spinner("Checking for drift..."):
        drift_data = get_drift_status()
    
    if drift_data:
        # Status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_emoji = "✅" if drift_data["health_status"] == "HEALTHY" else "⚠️"
            st.metric("Health Status", f"{status_emoji} {drift_data['health_status']}")
        
        with col2:
            drift_emoji = "❌" if drift_data["drift_detected"] else "✅"
            st.metric("Drift Detected", f"{drift_emoji} {'Yes' if drift_data['drift_detected'] else 'No'}")
        
        with col3:
            st.metric("Drift Share", f"{drift_data['drift_share']:.1%}")
        
        with col4:
            st.metric("Drifted Features", f"{drift_data['number_of_drifted_columns']}/{drift_data['total_columns']}")
        
        st.markdown("---")
        
        # Drift visualization
        st.markdown("### Drift Analysis")
        
        drift_pct = drift_data['drift_share'] * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=drift_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Drift Percentage"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "red" if drift_pct > 50 else "orange" if drift_pct > 30 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 50], 'color': "lightyellow"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.markdown("### 💡 Recommendation")
        if drift_data["drift_detected"]:
            st.warning(f"⚠️ {drift_data['recommendation']}")
            st.info("Consider triggering model retraining via webhook: `/webhook/retrain`")
        else:
            st.success(f"✅ {drift_data['recommendation']}")
    else:
        st.error("❌ Could not fetch monitoring data. Make sure the API is running.")


def about_page():
    """About page."""
    st.header("ℹ️ About This Project")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 MLOps Churn Prediction Pipeline
        
        A **production-ready** end-to-end ML system for predicting customer churn, 
        built with modern MLOps practices and designed for real-world deployment.
        """)
    
    with col2:
        st.metric("Project Status", "90% Complete", "🟢")
        st.metric("Code Lines", "~2,500", "✅")
    
    st.markdown("---")
    
    # Key metrics
    st.markdown("### 📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Models Trained",
            value="3",
            delta="CatBoost Champion"
        )
    
    with col2:
        st.metric(
            label="ROC-AUC Score",
            value="0.8485",
            delta="+0.44% vs baseline"
        )
    
    with col3:
        st.metric(
            label="API Endpoints",
            value="15+",
            delta="REST + Webhooks"
        )
    
    with col4:
        st.metric(
            label="Features Engineered",
            value="46",
            delta="Automated pipeline"
        )
    
    st.markdown("---")
    
    # Architecture diagram
    st.markdown("### 🏗️ System Architecture")
    
    st.markdown("""
```
    Data Pipeline → Feature Store → Training → Model Registry
                                                    ↓
                                        ┌───────────┴──────────┐
                                        ↓                      ↓
                                   Serving API           Webhooks
                                        ↓                      ↓
                                   Monitoring ← Drift Detection
```
    """)
    
    st.markdown("---")
    
    # Features grid
    st.markdown("### ⚡ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 ML Pipeline**
        - Automated data validation
        - 46 engineered features
        - Multi-model training (XGBoost, LightGBM, CatBoost)
        - MLflow experiment tracking
        
        **🔧 MLOps**
        - Feature Store for consistency
        - Automated model selection
        - Webhook-driven automation
        - Background job processing
        """)
    
    with col2:
        st.markdown("""
        **🌐 Serving**
        - FastAPI REST endpoints
        - Batch prediction support
        - Interactive Swagger docs
        - Real-time predictions
        
        **📈 Monitoring**
        - Statistical drift detection
        - Automated alerts
        - Health status tracking
        - Performance dashboards
        """)
    
    st.markdown("---")
    
    # Tech stack with icons
    st.markdown("### 🛠️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **Machine Learning**
        - 🤖 XGBoost 2.0+
        - ⚡ LightGBM 4.1+
        - 🐱 CatBoost 1.2+
        - 📊 scikit-learn 1.3+
        """)
    
    with tech_col2:
        st.markdown("""
        **MLOps & Infrastructure**
        - 📦 MLflow 2.9+
        - 🔍 Drift Detection
        - 🚀 FastAPI 0.104+
        - 🎨 Streamlit 1.29+
        """)
    
    with tech_col3:
        st.markdown("""
        **Data & Tools**
        - 🐼 Pandas 2.0+
        - 📈 Plotly 5.18+
        - 🔧 Pydantic 2.5+
        - 📊 NumPy 1.24+
        """)
    
    st.markdown("---")
    
    # Quick start
    st.markdown("### 🚀 Quick Start")
    
    with st.expander("💻 See Installation & Usage", expanded=False):
        st.code("""
# 1. Clone repository
git clone https://github.com/EstebanM-M/mlops-churn-prediction
cd mlops-churn-prediction

# 2. Install dependencies
pip install -e .

# 3. Start API
uvicorn serving.api:app --reload

# 4. Start Dashboard (in another terminal)
streamlit run src/frontend/app.py
        """, language="bash")
    
    st.markdown("---")
    
    # Author section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        ### 👤 About the Author
        
        **Esteban Morales Mahecha**  
        Electronic Engineer | Escuela Colombiana de Ingeniería Julio Garavito (2024)  
        Transitioning to ML/AI Engineering
        
        **Demonstrated Skills:**
        - End-to-end ML pipeline design
        - MLOps best practices & automation
        - Feature engineering & model training
        - API development & deployment
        - Production system monitoring
        """)
    
    with col2:
        st.markdown("""
        **Connect**
        
        📧 Email  
        💼 [LinkedIn](https://www.linkedin.com/in/esteban-morales-mahecha/)  
        🐙 [GitHub](https://github.com/EstebanM-M)
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with ❤️ using Python, FastAPI, MLflow, and Streamlit</p>
        <p>📝 MIT License | 🚀 Production Ready | ⭐ 90% Complete</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
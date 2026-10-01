import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction App | FA2 Case Study",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2a5298;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #dc3545;
        margin-top: 15px;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #28a745;
        margin-top: 15px;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        background-color: #2a5298;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Models & Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    models = {
        'Tuned Random Forest (Best)': joblib.load('best_tuned_rf_model.pkl'),
        'Support Vector Machine (SVM)': joblib.load('support_vector_machine_model.pkl'),
        'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
        'Random Forest': joblib.load('random_forest_model.pkl'),
        'Decision Tree': joblib.load('decision_tree_model.pkl')
    }
    
    metrics = joblib.load('model_metrics.pkl') if os.path.exists('model_metrics.pkl') else None
    dataset = pd.read_csv('heart_disease_fa2.csv') if os.path.exists('heart_disease_fa2.csv') else None
    
    return scaler, feature_names, models, metrics, dataset

scaler, feature_names, models, metrics, dataset = load_artifacts()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">🫀 Heart Disease Risk Prediction & ML Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Clinical Decision Support System powered by Machine Learning & Streamlit</div>', unsafe_allow_html=True)

with col_h2:
    st.markdown("""
    <div style="text-align: right; font-size: 0.85rem; color: #444; background: #eef2f7; padding: 10px; border-radius: 8px;">
        <b>Student:</b> Nishant Nitin Wadkar<br>
        <b>PRN:</b> 125M1H060 | <b>Class:</b> SYMCA<br>
        <b>Course:</b> ADS Lab (MCA33PE21) - FA2
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# Sidebar Controls & Feature Inputs
# ---------------------------------------------------------
st.sidebar.header("⚙️ Model & Clinical Parameters")

selected_model_name = st.sidebar.selectbox(
    "Select ML Algorithm for Prediction:",
    list(models.keys()) + ["Compare All Models Simultaneously"]
)

st.sidebar.subheader("👤 Patient Clinical Inputs")

# Patient Demographics
with st.sidebar.expander("1. Demographics & Vitals", expanded=True):
    age = st.slider("Age (years)", 20, 90, 54)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)", index=1)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Serum Cholesterol (mg/dl)", 120, 570, 240)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False (0)" if x == 0 else "True (1)", index=0)

# Vitals & Symptoms
with st.sidebar.expander("2. Symptoms & Diagnostic Tests", expanded=False):
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], 
                      format_func=lambda x: {
                          0: "0: Typical Angina",
                          1: "1: Atypical Angina",
                          2: "2: Non-Anginal Pain",
                          3: "3: Asymptomatic"
                      }[x])
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                           format_func=lambda x: {
                               0: "0: Normal",
                               1: "1: ST-T Wave Abnormality",
                               2: "2: Left Ventricular Hypertrophy"
                           }[x])
    thalach = st.slider("Maximum Heart Rate Achieved (bpm)", 70, 210, 150)
    exang = st.radio("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", index=0)

# Cardiac Imaging & ST Segment
with st.sidebar.expander("3. Cardiac Imaging & ST Segment", expanded=False):
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox("Peak Exercise ST Segment Slope", options=[0, 1, 2],
                         format_func=lambda x: {
                             0: "0: Upsloping",
                             1: "1: Flat",
                             2: "2: Downsloping"
                         }[x])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3], index=0)
    thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3],
                        format_func=lambda x: {
                            1: "1: Normal",
                            2: "2: Fixed Defect",
                            3: "3: Reversable Defect"
                        }[x], index=1)

# Format input data
input_df = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])[feature_names]

input_scaled = scaler.transform(input_df)

# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Risk Prediction", 
    "📊 Model Benchmarks", 
    "📈 Exploratory Data Analysis (EDA)", 
    "ℹ️ Project Documentation"
])

# ---------------------------------------------------------
# TAB 1: Risk Prediction
# ---------------------------------------------------------
with tab1:
    st.subheader("🔍 Patient Clinical Risk Assessment")
    
    col_input, col_pred = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("##### 📋 Input Summary Profile")
        st.dataframe(input_df.T.rename(columns={0: 'Patient Feature Value'}), use_container_width=True)
    
    with col_pred:
        st.markdown("##### 🎯 Prediction Output")
        
        predict_btn = st.button("🚀 Analyze Heart Disease Risk", type="primary", use_container_width=True)
        
        if selected_model_name != "Compare All Models Simultaneously":
            model = models[selected_model_name]
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else (0.95 if pred == 1 else 0.05)
            
            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>🚨 HIGH RISK OF HEART DISEASE DETECTED</h3>
                    <p style="font-size: 1.1rem; margin-bottom: 5px;">Selected Model: <b>{selected_model_name}</b></p>
                    <p style="font-size: 1.3rem; font-weight: bold;">Estimated Disease Probability: {prob*100:.1f}%</p>
                    <p>⚠️ Clinical Recommendation: Consult a cardiologist immediately for further diagnostic evaluation (Angiography / Stress Test).</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ LOW RISK OF HEART DISEASE</h3>
                    <p style="font-size: 1.1rem; margin-bottom: 5px;">Selected Model: <b>{selected_model_name}</b></p>
                    <p style="font-size: 1.3rem; font-weight: bold;">Estimated Healthy Probability: {(1-prob)*100:.1f}%</p>
                    <p>💚 Clinical Recommendation: Patient vitals appear within expected normal bounds. Maintain healthy lifestyle habits.</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Heart Disease Probability (%)", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#dc3545" if prob > 0.5 else "#28a745"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#d4edda'},
                        {'range': [40, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.markdown("#### ⚖️ Multi-Model Consensus Comparison")
            comparison_results = []
            for m_name, m_obj in models.items():
                m_pred = m_obj.predict(input_scaled)[0]
                m_prob = m_obj.predict_proba(input_scaled)[0][1] if hasattr(m_obj, "predict_proba") else (1.0 if m_pred == 1 else 0.0)
                comparison_results.append({
                    'Model Name': m_name,
                    'Prediction': '🚨 Heart Disease Present' if m_pred == 1 else '✅ Healthy / No Disease',
                    'Risk Probability': f"{m_prob*100:.1f}%",
                    'Confidence': 'High' if abs(m_prob - 0.5) > 0.25 else 'Moderate'
                })
            
            comp_df = pd.DataFrame(comparison_results)
            st.dataframe(comp_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Model Benchmarks
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Machine Learning Model Evaluation & Metrics")
    
    if metrics:
        metrics_df = pd.DataFrame(metrics).T
        
        st.markdown("##### 🏆 Algorithm Performance Comparison Table")
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color='#d4edda').format("{:.4f}"),
            use_container_width=True
        )
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### 📈 Accuracy & F1-Score Comparison")
            fig_bar = px.bar(
                metrics_df.reset_index().melt(id_vars='index', value_vars=['Accuracy', 'F1-Score']),
                x='index', y='value', color='variable', barmode='group',
                labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'},
                color_discrete_sequence=['#2a5298', '#2ecc71']
            )
            fig_bar.update_layout(height=350, yaxis_range=[0.5, 1.0])
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_chart2:
            st.markdown("##### 🎯 ROC-AUC Score Comparison")
            fig_auc = px.bar(
                metrics_df.reset_index(),
                x='index', y='ROC-AUC', color='ROC-AUC',
                color_continuous_scale='Viridis',
                labels={'index': 'Model'}
            )
            fig_auc.update_layout(height=350, yaxis_range=[0.5, 1.0])
            st.plotly_chart(fig_auc, use_container_width=True)

    st.markdown("##### 🔍 GridSearchCV & Cross-Validation Insights")
    st.info("""
    - **GridSearchCV Hyperparameter Optimization**: Applied to `RandomForestClassifier` tuning `n_estimators`, `max_depth`, `min_samples_split`, and `criterion`.
    - **5-Fold Stratified Cross-Validation**: Validates model stability across folds without data leakage.
    - **Model Selection Summary**: Support Vector Classifier (SVM) and Tuned Random Forest achieved superior ROC-AUC scores (>0.74), showing high generalization accuracy on unseen clinical data.
    """)

# ---------------------------------------------------------
# TAB 3: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
with tab3:
    st.subheader("📈 Exploratory Data Analysis (EDA)")
    
    if dataset is not None:
        st.markdown("##### 📊 Dataset Overview")
        st.write(f"Total Patient Records: **{len(dataset)}** | Total Features: **{dataset.shape[1]-1}**")
        
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            st.markdown("##### Target Class Distribution")
            fig_target = px.pie(
                dataset, names='heart_disease', 
                title='Heart Disease Diagnosis (0: Absence, 1: Presence)',
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                hole=0.4
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
        with col_eda2:
            st.markdown("##### Age vs Max Heart Rate (thalach)")
            fig_scatter = px.scatter(
                dataset, x='age', y='thalach', color='heart_disease',
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                labels={'age': 'Age (Years)', 'thalach': 'Max Heart Rate (bpm)'},
                title='Age vs Max Heart Rate by Disease Target'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("##### 🗃️ Raw Dataset Preview")
        st.dataframe(dataset.head(15), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: Project Documentation & Architecture
# ---------------------------------------------------------
with tab4:
    st.subheader("ℹ️ Case Study Documentation & Submission Details")
    
    st.markdown("""
    ### 📌 Assignment Overview: FA2 Activity
    - **Course**: Advanced Data Science Lab [MCA33PE21]
    - **Student Name**: Nishant Nitin Wadkar
    - **PRN**: 125M1H060
    - **Class / Division**: SYMCA (Semester-I)
    
    ---
    ### 🔬 Machine Learning Pipeline Architecture
    1. **Data Acquisition & Preprocessing (Part 1)**:
       - UCI Heart Disease Dataset containing 13 clinical attributes.
       - Missing values handling, feature standardization (`StandardScaler`), and 80-20 train-test split.
    2. **Model Implementation & Evaluation (Part 2)**:
       - Implemented 4 distinct scikit-learn classifiers: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM).
       - Performed GridSearchCV hyperparameter optimization and 5-Fold Stratified Cross-Validation.
    3. **Model Deployment using Streamlit (Part 3)**:
       - Interactive web interface with sidebar feature controls, multi-model consensus prediction, probability gauges, performance metric charts, and live EDA.
    4. **Submission Artifacts (Part 4)**:
       - `.ipynb` Jupyter Notebook with complete executed code & visualizations.
       - `app.py` Streamlit Web Application script.
       - `125M1H060_ADS_Lab_FA2_Report.pdf` PDF Summary Report.
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 *Built with Streamlit & Scikit-Learn*")

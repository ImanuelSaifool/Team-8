import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------------
# PAGE SETUP & FORMAL CLINICAL THEME
# -------------------------------------------------------------------------
st.set_page_config(page_title="Medicaid Allocation Risk Model", layout="wide")

st.markdown("""
    <style>
    /* Base font to standard enterprise sans-serif */
    html, body, [class*="st-"] {
        font-family: Arial, Helvetica, sans-serif;
    }
    /* Formal serif for all headers */
    h1, h2, h3, h4 {
        font-family: 'Georgia', serif !important;
        color: #1a252f !important;
        font-weight: 500 !important;
        border-bottom: 1px solid #c0c0c0;
        padding-bottom: 4px;
        margin-bottom: 16px;
    }
    /* Square off all buttons and inputs for a strict software feel */
    .stButton>button {
        background-color: #2c3e50;
        color: #ffffff;
        border-radius: 0px;
        border: none;
        font-family: Arial, sans-serif;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #1a252f;
        color: #ffffff;
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-radius: 0px !important;
    }
    /* Override Streamlit's bright boxes to clinical slate gray */
    div[data-testid="stAlert"] {
        background-color: #f8f9fa;
        color: #212529;
        border: 1px solid #ced4da;
        border-left: 5px solid #495057;
        border-radius: 0px;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-family: 'Georgia', serif;
        color: #1a252f;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_data():
    return joblib.load('meps_model_data.pkl')

try:
    model_data = load_model_data()
    final_rf_model = model_data['model']
    selected_features_list = model_data['selected_features']
    X_train_selected = model_data['X_train_selected']
except FileNotFoundError:
    st.error("System Error: 'meps_model_data.pkl' missing from deployment directory.")
    st.stop()

# -------------------------------------------------------------------------
# SIDEBAR: EHR CONTEXT
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Patient Record Details")
    st.text_input("Medical Record Number (MRN)", value="MRN-84729", disabled=True)
    st.text_input("Attending Physician", value="G. House, MD", disabled=True)
    st.divider()
    st.markdown("**Assessment Objective:**")
    st.caption("Determine statutory Medicaid eligibility and project statistical public burden to mitigate oncology treatment non-adherence.")

# -------------------------------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------------------------------
st.title("Proactive Medicaid Allocation & Financial Toxicity Model")
st.markdown("Department of Biostatistics | Medicaid Decision Support System")
st.divider()

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("### I. Demographics")
    patient_age = st.number_input("Age (Years)", min_value=0, max_value=120, value=55)
    patient_sex = st.radio("Sex at Birth", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)
    
    region_choice = st.selectbox("Geographic Region", options=["Northeast", "Midwest", "South", "West"])
    region_map = {"Northeast": 1, "Midwest": 2, "South": 3, "West": 4}
    region_idx = region_map[region_choice]
    
    st.markdown("### II. Coverage Parameters")
    patient_vet = 1 if st.checkbox("Veterans Affairs (VA)") else 0
    patient_mil = 1 if st.checkbox("Military / Tricare") else 0
    patient_fed = 1 if st.checkbox("Federal Employee (FEHB)") else 0

with col2:
    st.markdown("### III. Socioeconomic Data")
    patient_faminc = st.number_input("Annual Family Income (USD)", min_value=0.0, value=45000.0, step=1000.0)
    patient_totslf = st.number_input("Current Out-of-Pocket Burden (USD)", min_value=0.0, value=2500.0, step=100.0)
    
    patient_famsze = st.number_input("Household Size", min_value=1, value=2)
    patient_pov = st.slider("Federal Poverty Category (1=Poor, 5=High)", min_value=1, max_value=5, value=3)
    
    st.markdown("### IV. SDoH Risk Factors")
    patient_foodst = 1 if st.checkbox("Food Insecurity (SNAP)") else 2
    patient_adl = 1 if st.checkbox("ADL Assistance Required") else 2
    patient_ddnwrk = st.number_input("Days Missed Work (Illness)", min_value=0, value=0)
    patient_phq2 = st.slider("PHQ-2 Depression Screener", min_value=0, max_value=6, value=0)

with col3:
    st.markdown("### V. Clinical Pathology")
    cancer_list = ["Bladder", "Breast", "Cervix", "Colon", "Lung", "Lymphoma", "Melanoma", "Other", "Prostate", "Skin (Non-Melanoma)", "Skin (Unknown)", "Uterus", "None"]
    cancer_choice = st.selectbox("Primary Oncology Diagnosis", options=cancer_list, index=3) 
    
    disease_list = ["Diabetes", "High Blood Pressure", "Coronary Heart Disease", "Angina", "Heart Attack", "Other Heart Disease", "Stroke", "High Cholesterol", "Emphysema", "Asthma", "Chronic Bronchitis", "Arthritis"]
    selected_diseases = st.multiselect("Documented Comorbidities", options=disease_list)
    
    st.markdown("### VI. Utilization (Trailing 12 Mo)")
    patient_ipdis = st.number_input("Inpatient Admissions", min_value=0, value=0)
    patient_ipngtd = st.number_input("Total Inpatient Days", min_value=0, value=0)
    patient_ertot = st.number_input("Emergency Department Visits", min_value=0, value=0)

# -------------------------------------------------------------------------
# BACKEND & RESULTS
# -------------------------------------------------------------------------
st.divider()
if st.button("Evaluate Medicaid Eligibility & Calculate Subsidy", type="primary", use_container_width=True):
    with st.spinner("Processing data vector against Medicaid thresholds..."):
        
        # Base Logic
        catastrophic_cost = 1 if patient_totslf > (0.10 * patient_faminc) else 0
        patient_medicare = 1 if patient_age >= 65 else 0
        patient_chip = 1 if patient_age <= 19 else 0
        
        cancer_features = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER", "CAPROSTA", "CASKINNM", "CASKINDK", "CAUTERUS"]
        patient_cancers = {col: 2 for col in cancer_features}
        if cancer_choice != "None":
            selected_idx = cancer_list.index(cancer_choice)
            patient_cancers[cancer_features[selected_idx]] = 1

        disease_features = ["DIABDX", "HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "CHOLDX", "EMPHDX", "ASTHDX", "CHBRON", "ARTHDX"]
        patient_diseases = {col: 2 for col in disease_features}
        patient_age_diag = {}
        for col in disease_features:
            if col != "CHBRON": 
                age_col = col.replace("DX", "AGED")
                patient_age_diag[age_col] = 0
                
        for d in selected_diseases:
            idx = disease_list.index(d)
            feat_name = disease_features[idx]
            patient_diseases[feat_name] = 1
            if feat_name != "CHBRON":
                age_col = feat_name.replace("DX", "AGED")
                patient_age_diag[age_col] = patient_age 
            
        total_visits_calc = patient_ertot 
        inpatient_burden_calc = patient_ipdis + patient_ipngtd
        care_intensity_calc = total_visits_calc + inpatient_burden_calc
        er_dependency_calc = patient_ertot / (total_visits_calc + 1e-6)
        
        has_cancer = 1 if cancer_choice != "None" else 0
        cancer_dep_calc = 1 if (has_cancer == 1 and patient_phq2 > 2) else 0
        disease_count = len(selected_diseases)
        elderly_multi_calc = 1 if (patient_age >= 65 and disease_count >= 2) else 0
        fin_spiral_calc = catastrophic_cost * patient_ddnwrk
        
        pov_reverse = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}.get(patient_pov, 3)
        sdoh_score_calc = pov_reverse + (2 if patient_foodst == 1 else 0) + (2 if patient_adl == 1 else 0)
        avg_nights_calc = patient_ipngtd / max(1, patient_ipdis)

        patient_data = {
            'FAMINC': [patient_faminc], 'TOTSLF': [patient_totslf], 'CATASTROPHIC_COST': [catastrophic_cost],
            'AGELAST': [patient_age], 'SEX': [patient_sex], 'IS_MEDICARE_AGE': [patient_medicare],
            'IS_CHIP_AGE': [patient_chip], 'IS_VETERAN': [patient_vet], 'IS_MILITARY_FAM': [patient_mil],
            'IS_FED_WORKER': [patient_fed], 'FAMILY_SIZE': [patient_famsze],
            'INCOME_PER_CAPITA': [patient_faminc / max(1, patient_famsze)],
            'POVERTY_CATEGORY': [patient_pov], 'FOOD_STAMPS': [patient_foodst],
            'DAYS_MISSED_WORK': [patient_ddnwrk], 'ADL_HELP_NEEDED': [patient_adl],
            'PHQ2_DEPRESSION_SCORE': [patient_phq2],
            'REGION_NORTHEAST': [1 if region_idx == 1 else 0],
            'REGION_MIDWEST': [1 if region_idx == 2 else 0],
            'REGION_SOUTH': [1 if region_idx == 3 else 0],
            'REGION_WEST': [1 if region_idx == 4 else 0],
            'TOTAL_VISITS': [total_visits_calc], 'INPATIENT_BURDEN': [inpatient_burden_calc],
            'CARE_INTENSITY_INDEX': [care_intensity_calc], 'ER_DEPENDENCY': [er_dependency_calc],
            'CANCER_AND_DEPRESSION': [cancer_dep_calc], 'ELDERLY_MULTIMORBIDITY': [elderly_multi_calc],
            'FINANCIAL_SPIRAL_RISK': [fin_spiral_calc], 'SDOH_VULNERABILITY_SCORE': [sdoh_score_calc],
            'AVG_NIGHTS_PER_STAY': [avg_nights_calc]
        }
        
        patient_data.update({k: [v] for k, v in patient_cancers.items()})
        patient_data.update({k: [v] for k, v in patient_diseases.items()})
        patient_data.update({k: [v] for k, v in patient_age_diag.items()})
        
        if hasattr(final_rf_model, "feature_names_in_"):
            expected_features = final_rf_model.feature_names_in_
        else:
            expected_features = selected_features_list
            
        for col in expected_features:
            if col not in patient_data:
                patient_data[col] = [0]
                
        new_patient_df = pd.DataFrame(patient_data)[expected_features]
        new_patient_df_selected = new_patient_df.values
        
        # Medicaid Eligibility Gate Logic
        approx_fpl = 14580 + (5140 * (max(1, patient_famsze) - 1))
        medicaid_income_limit = approx_fpl * 1.38 
        is_medicaid_eligible = True
        
        if patient_age > 19 and patient_faminc > medicaid_income_limit:
            if patient_adl == 2 and patient_medicare == 0:
                is_medicaid_eligible = False
        if patient_chip == 1 and patient_faminc > (approx_fpl * 3.0):
            is_medicaid_eligible = False

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("### Medicaid Assessment")
            if is_medicaid_eligible:
                recommended_subsidy = final_rf_model.predict(new_patient_df_selected)[0]
                recommended_subsidy = max(0, recommended_subsidy)
                st.metric(label="Projected Medicaid Allocation", value=f"${recommended_subsidy:,.2f}")
                
                if patient_phq2 > 2 or catastrophic_cost == 1:
                    st.info("Clinical Note: Eligible for Medicaid. Patient flags positive for depressive symptoms or catastrophic out-of-pocket costs. Preemptive adherence intervention required.")
                else:
                    st.info("Clinical Note: Eligible for Medicaid. Standard allocation path. No acute SDoH interventions flagged.")
            else:
                st.metric(label="Projected Medicaid Allocation", value="$0.00")
                st.warning("Coverage Flag: Patient income exceeds statutory Federal Poverty Level (FPL) eligibility thresholds for Medicaid/CHIP allocation.")

        with res_col2:
            st.markdown("### Cost Driver Analysis (SHAP)")
            explainer = shap.Explainer(final_rf_model, X_train_selected)
            shap_values = explainer(new_patient_df)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], show=False, max_display=8)
            plt.tight_layout()
            st.pyplot(fig)
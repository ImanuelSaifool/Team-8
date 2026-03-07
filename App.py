import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

# -------------------------------------------------------------------------
# PAGE SETUP & ASSETS
# -------------------------------------------------------------------------
st.set_page_config(page_title="Proactive Medicaid Subsidy Calculator", layout="wide")

@st.cache_resource
def load_model_data():
    return joblib.load('meps_model_data.pkl')

try:
    model_data = load_model_data()
    final_rf_model = model_data['model']
    selected_features_list = model_data['selected_features']
    X_train_selected = model_data['X_train_selected']
except FileNotFoundError:
    st.error("Error: 'meps_model_data.pkl' not found. Please ensure it is uploaded to your GitHub repository.")
    st.stop()

# -------------------------------------------------------------------------
# UI FRONTEND
# -------------------------------------------------------------------------
st.title("🏥 Proactive Medicaid Subsidy Calculator")
st.markdown("### A Clinical Decision Support Tool for Financial Toxicity Prevention")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Patient Demographics")
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=55)
    patient_sex = st.radio("Assigned Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    
    region_choice = st.selectbox("US Region", options=["Northeast", "Midwest", "South", "West"])
    region_map = {"Northeast": 1, "Midwest": 2, "South": 3, "West": 4}
    region_idx = region_map[region_choice]
    
    st.subheader("Coverage Status")
    patient_vet = 1 if st.checkbox("US Veteran") else 0
    patient_mil = 1 if st.checkbox("Military Family (Tricare)") else 0
    patient_fed = 1 if st.checkbox("Federal Worker") else 0

with col2:
    st.header("Financial & SDoH")
    patient_faminc = st.number_input("Family Income ($)", min_value=0.0, value=45000.0, step=1000.0)
    patient_totslf = st.number_input("Out-of-Pocket Costs ($)", min_value=0.0, value=2500.0, step=100.0)
    
    patient_famsze = st.number_input("Family Size", min_value=1, value=2)
    patient_pov = st.slider("Poverty Category", min_value=1, max_value=5, value=3, help="1=Negative/Poor, 5=High Income")
    
    st.subheader("Hardship Flags")
    patient_foodst = 1 if st.checkbox("Receives Food Stamps/SNAP") else 2
    patient_adl = 1 if st.checkbox("Needs ADL Help (Bathing, etc.)") else 2
    patient_ddnwrk = st.number_input("Days Missed Work", min_value=0, value=0)
    patient_phq2 = st.slider("PHQ-2 Depression Score", min_value=0, max_value=6, value=0)

with col3:
    st.header("Clinical Profile")
    cancer_list = ["Bladder", "Breast", "Cervix", "Colon", "Lung", "Lymphoma", "Melanoma", "Other", "Prostate", "Skin (Non-Melanoma)", "Skin (Unknown)", "Uterus", "None"]
    cancer_choice = st.selectbox("Primary Cancer Diagnosis", options=cancer_list)
    
    disease_list = ["Diabetes", "High Blood Pressure", "Coronary Heart Disease", "Angina", "Heart Attack", "Other Heart Disease", "Stroke", "High Cholesterol", "Emphysema", "Asthma", "Chronic Bronchitis", "Arthritis"]
    selected_diseases = st.multiselect("Comorbidities", options=disease_list)
    
    st.subheader("Healthcare Utilization (Past 12 Mos)")
    patient_ipdis = st.number_input("Hospital Discharges", min_value=0, value=0)
    patient_ipngtd = st.number_input("Total Hospital Nights", min_value=0, value=0)
    patient_ertot = st.number_input("ER Visits", min_value=0, value=0)

# -------------------------------------------------------------------------
# BACKEND CALCULATION LOGIC
# -------------------------------------------------------------------------
st.divider()
if st.button("Calculate Statistical Subsidy", type="primary", use_container_width=True):
    with st.spinner("Processing clinical profile..."):
        
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
        patient_age_diag = {col.replace("DX", "AGED"): 0 for col in disease_features}
        
        for d in selected_diseases:
            idx = disease_list.index(d)
            feat_name = disease_features[idx]
            patient_diseases[feat_name] = 1
            patient_age_diag[feat_name.replace("DX", "AGED")] = patient_age 
            
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
        
        for col in selected_features_list:
            if col not in patient_data:
                patient_data[col] = [0]
                
        new_patient_df = pd.DataFrame(patient_data)[selected_features_list]
        
        approx_fpl = 14580 + (5140 * (max(1, patient_famsze) - 1))
        medicaid_income_limit = approx_fpl * 1.38 
        is_medicaid_eligible = True
        
        if patient_age > 19 and patient_faminc > medicaid_income_limit:
            if patient_adl == 2 and patient_medicare == 0:
                is_medicaid_eligible = False
        if patient_chip == 1 and patient_faminc > (approx_fpl * 3.0):
            is_medicaid_eligible = False

        if is_medicaid_eligible:
            recommended_subsidy = final_rf_model.predict(new_patient_df)[0]
            recommended_subsidy = max(0, recommended_subsidy)
            st.success(f"### Computed Statistical Subsidy Expectation: ${recommended_subsidy:,.2f}")
            st.caption("Estimated public burden based on full-year patient realities and historical MEPS data.")
        else:
            st.error("### Computed Statistical Subsidy Expectation: $0.00")
            st.caption("FLAG: Patient income exceeds Medicaid/CHIP eligibility thresholds for their demographic.")

        st.markdown("### Cost Drivers Analysis (SHAP)")
        explainer = shap.Explainer(final_rf_model, X_train_selected)
        shap_values = explainer(new_patient_df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)
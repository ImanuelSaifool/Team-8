# Proactive Medicaid Allocation & Financial Toxicity Risk Model
### Department of Biostatistics | Oncology Resource Management Tool

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://team8iscool.streamlit.app/)

## 🏥 Project Overview
This clinical decision support tool utilizes historical **Medical Expenditure Panel Survey (MEPS)** data to predict the required Medicaid subsidy allocation for oncology patients. The model identifies the intersection between financial burden, depressive symptoms, and treatment non-adherence to provide practitioners with a quantitative "financial toxicity" risk profile on Day 1 of diagnosis.

## 🔬 Clinical Impact
Financial toxicity is a documented side effect of cancer treatment that directly correlates with treatment abandonment. This model provides three key clinical interventions:
1. **Early Navigation:** Predicts statutory Medicaid eligibility and allocation before a patient incurs catastrophic debt.
2. **Adherence Risk Profiling:** Uses SDoH (Social Determinants of Health) and PHQ-2 depression screeners to flag patients at high risk of quitting treatment.
3. **Data-Driven Advocacy:** Aggregates predicted subsidy requirements to justify hospital grant applications and community resource funding.

## 🛠️ Technical Stack
- **Modeling:** Random Forest Regressor / HistGradientBoosting
- **Explainability:** SHAP (SHapley Additive exPlanations) for local feature impact
- **Interface:** Streamlit (Enterprise Clinical UI)
- **Data Source:** AHRQ Medical Expenditure Panel Survey (MEPS)

## 📁 Repository Structure
- `App.py`: The main Streamlit web application.
- `meps_model_data.pkl`: The fully trained machine learning "brain" (exported from the training phase).
- `requirements.txt`: Environment dependencies.
- `.streamlit/config.toml`: Custom clinical theme and color palette settings.

## 🚀 Deployment
The application is deployed via Streamlit Community Cloud and can be accessed at:
**[https://team8iscool.streamlit.app/](https://team8iscool.streamlit.app/)**

---
*Developed by Safwan, Mehraf, Imanuel* University College London*

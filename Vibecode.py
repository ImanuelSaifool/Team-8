import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.feature_selection import mutual_info_classif

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------------------------------------------------------------------------
df2019 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/OneDrive/Desktop/Coding%20Projects/h216.csv")
df2020 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/OneDrive/Desktop/Coding%20Projects/H224.csv")
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2. STANDARDIZING
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Treating inflation
    # Family income
df2019["FAMINC19"] = df2019["FAMINC19"] * 1.19
df2020["FAMINC20"] = df2020["FAMINC20"] * 1.17
df2021p1['FAMINC21'] = df2021p1['FAMINC21'] * 1.12
df2021p2['FAMINC21'] = df2021p2['FAMINC21'] * 1.12
df2022['FAMINC22'] = df2022['FAMINC22'] * 1.04

df2019["TOTSLF19"] = df2019["TOTSLF19"] * 1.19
df2020["TOTSLF20"] = df2020["TOTSLF20"] * 1.17
df2021p1['TOTSLF21'] = df2021p1['TOTSLF21'] * 1.12
df2021p2['TOTSLF21'] = df2021p2['TOTSLF21'] * 1.12
df2022['TOTSLF22'] = df2022['TOTSLF22'] * 1.04
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Out of pocket cost
df2019 = df2019.rename(columns={"TOTSLF19": "TOTSLF"})
df2020 = df2020.rename(columns={"TOTSLF20": "TOTSLF"})
df2021p1 = df2021p1.rename(columns={"TOTSLF21": "TOTSLF"})
df2021p2 = df2021p2.rename(columns={"TOTSLF21": "TOTSLF"})
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF"})

# Family income
df2019 = df2019.rename(columns={"FAMINC19": "FAMINC"})
df2020 = df2020.rename(columns={"FAMINC20": "FAMINC"})
df2021p1 = df2021p1.rename(columns={"FAMINC21": "FAMINC"})
df2021p2 = df2021p2.rename(columns={"FAMINC21": "FAMINC"})
df2022 = df2022.rename(columns={"FAMINC22": "FAMINC"})
df2023 = df2023.rename(columns={"FAMINC23": "FAMINC"})

# Renaming Medicaid
df2019 = df2019.rename(columns={"TOTMCD19": "TOTMCD"})
df2020 = df2020.rename(columns={"TOTMCD20": "TOTMCD"})
df2021p1 = df2021p1.rename(columns={"TOTMCD21": "TOTMCD"})
df2021p2 = df2021p2.rename(columns={"TOTMCD21": "TOTMCD"})
df2022 = df2022.rename(columns={"TOTMCD22": "TOTMCD"})
df2023 = df2023.rename(columns={"TOTMCD23": "TOTMCD"})

# Renaming Region
df2019 = df2019.rename(columns={"REGION19": "REGION"})
df2020 = df2020.rename(columns={"REGION20": "REGION"})
df2021p1 = df2021p1.rename(columns={"REGION21": "REGION"})
df2021p2 = df2021p2.rename(columns={"REGION21": "REGION"})
df2022 = df2022.rename(columns={"REGION22": "REGION"})
df2023 = df2023.rename(columns={"REGION23": "REGION"})
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Combining datasets
main_df = pd.concat([df2019, df2020, df2021p1, df2021p2, df2022, df2023], axis=0)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 3. FILTERING
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Define feature lists
demog_features = ["FAMINC", "TOTSLF", "AGELAST", "SEX", "REGION"]
adherance_features = ["DLAYCA42", "AFRDCA42", "DLAYPM42", "AFRDPM42"]
cancer_features = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER", "CAPROSTA", "CASKINNM", "CASKINDK", "CAUTERUS"]
other_disease_features = ["DIABDX_M18", "HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "CHOLDX", "EMPHDX", "ASTHDX", "CHBRON31", "ARTHDX"]
insurance_features = ["TOTMCD"]
Financial_Subjectivity_features = ["PROBPY42", "PYUNBL42", "CRFMPY42"]
features = demog_features + cancer_features + other_disease_features + adherance_features + insurance_features + Financial_Subjectivity_features
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Filters ONLY for patients who actually received Medicaid funding
clean_df = main_df[(main_df['CANCERDX'] == 1) & (main_df['TOTMCD'] > 0)].copy()

# Dropping duplicates for same person
clean_df = clean_df.drop_duplicates(subset=['DUPERSID'], keep='first')

# Filter negative values for demographics to prevent logic error
clean_df = clean_df[(clean_df[demog_features] >= 0).all(axis=1)]

# Filter negative values for cancer features to prevent logic error
clean_df[cancer_features] = clean_df[cancer_features].replace([-1,-7, -8, -9], 2)

for col in Financial_Subjectivity_features:
    if col in clean_df.columns:
        clean_df[col] = clean_df[col].replace([-1, -7, -8, -9], np.nan)
# ----------------------------------------------------------------------------------------------------------------------------------------------
clean_df['PUBLIC_TOTAL'] = clean_df[insurance_features].sum(axis=1)
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Standardize the adherence features (1 = Issue, 0 = No Issue, NaN = Missing)
def clean_adherence(val):
    if val == 1: 
        return 1  # Yes, experienced financial barrier
    elif val == 2: 
        return 0  # No, did not experience barrier
    else: 
        return np.nan # Treat negatives as missing data

# Apply cleaning to all four features
for col in adherance_features:
    clean_df[col] = clean_df[col].apply(clean_adherence)

# Drop rows where we don't have valid adherence data to avoid skewed math
clean_df = clean_df.dropna(subset=adherance_features)
# ----------------------------------------------------------------------------------------------------------------------------------------------
clean_df['TOXICITY_SCORE'] = clean_df[adherance_features].sum(axis=1)
def calculate_toxicity_tier(row):
    # If they are completely UNABLE to afford either care or meds -> Severe
    if row['AFRDCA42'] == 1 or row['AFRDPM42'] == 1:
        return "Severe (Forgone Care/Meds)"
    # Else, if they DELAYED care or meds -> Moderate
    elif row['DLAYCA42'] == 1 or row['DLAYPM42'] == 1:
        return "Moderate (Delayed Care/Meds)"
    # Otherwise -> None
    else:
        return "None (Fully Adherent)"

clean_df['TOXICITY_TIER'] = clean_df.apply(calculate_toxicity_tier, axis=1)
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Total Known Cost (What the patient paid + What public insurance paid)
clean_df['TOTAL_KNOWN_COST'] = clean_df['PUBLIC_TOTAL'] + clean_df['TOTSLF']

# 2. Calculate the Coverage Ratio
clean_df['COVERAGE_RATIO'] = clean_df['PUBLIC_TOTAL'] / (clean_df['TOTAL_KNOWN_COST'] + 1e-9)
clean_df['COVERAGE_RATIO_PCT'] = clean_df['COVERAGE_RATIO'] * 100

# ----------------------------------------------------------------------------------------------------------------------------------------------
clean_df['CATASTROPHIC_COST'] = (clean_df['TOTSLF'] > (0.10 * clean_df['FAMINC'])).astype(int)
# ----------------------------------------------------------------------------------------------------------------------------------------------
disease_map = {
    "HIBPDX": "High Blood Pressure",
    "ARTHDX": "Arthritis",
    "CHOLDX": "High Cholesterol",
    "OHRTDX": "Other Heart Disease",
    "DIABDX_M18": "Diabetes",
    "ASTHDX": "Asthma",
    "CHDDX": "Coronary Heart Disease",
    "STRKDX": "Stroke",
    "MIDX": "Heart Attack",
    "EMPHDX": "Emphysema",
    "ANGIDX": "Angina",
    "CHBRON31": "Chronic Bronchitis"
}
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 15. PRESCRIPTIVE MODELING (Random Forest Regressor)
#        Predicting the Optimal Medicaid Subsidy (Comprehensive Day 1 Baseline)
# ----------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("\n" + "="*80)
print("INITIALIZING PROACTIVE SUBSIDY CALCULATOR (CLINICAL & ECONOMIC)")
print("="*80)

# 1. Filter for Success (Only learn from the Fully Adherent patients)
success_df = clean_df[clean_df['TOXICITY_TIER'] == "None (Fully Adherent)"].copy()

# 2. Define Comprehensive "Day 1" Features
# We bring back Sex, Cancer Type, and Comorbidities since these are known on Day 1
ml_features = ['FAMINC', 'TOTSLF', 'AGELAST', 'CATASTROPHIC_COST', 'SEX'] + cancer_features + other_disease_features

# Drop rows with NaNs in our specific feature set
ml_df = success_df.dropna(subset=ml_features + ['PUBLIC_TOTAL']).copy()

X = ml_df[ml_features]
y = ml_df['PUBLIC_TOTAL']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"--- Model Ready ---")
print(f"Average variation from historical successful MEPS subsidies: ${mae:,.2f}")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 16. INTERACTIVE CLINICAL DECISION SUPPORT TOOL
#        Terminal Input with Clinical/Demographic Menus
# ----------------------------------------------------------------------------------------------------------------------------------------------

def run_subsidy_calculator():
    print("\n" + "="*80)
    print(" CLINICAL DECISION SUPPORT: DAY-1 SUBSIDY CALCULATOR")
    print("="*80)
    print("Type 'quit' at any prompt to exit the tool.\n")

    # Helper lists for the menus
    cancer_list = ["Bladder", "Breast", "Cervix", "Colon", "Lung", "Lymphoma", "Melanoma", "Other", "Prostate", "Skin (Non-Melanoma)", "Skin (Unknown)", "Uterus"]
    disease_list = ["Diabetes", "High Blood Pressure", "Coronary Heart Disease", "Angina", "Heart Attack", "Other Heart Disease", "Stroke", "High Cholesterol", "Emphysema", "Asthma", "Chronic Bronchitis", "Arthritis"]

    while True:
        try:
            # --- ECONOMIC DEMOGRAPHICS ---
            faminc_in = input("Enter Patient's Current Family Income ($): ").strip().replace(',', '').replace('$', '')
            if faminc_in.lower() == 'quit': break
            patient_faminc = float(faminc_in)

            totslf_in = input("Enter Projected Out-of-Pocket Cost for Treatment Plan ($): ").strip().replace(',', '').replace('$', '')
            if totslf_in.lower() == 'quit': break
            patient_totslf = float(totslf_in)

            age_in = input("Enter Patient Age: ").strip()
            if age_in.lower() == 'quit': break
            patient_age = int(age_in)

            sex_in = input("Enter Assigned Sex (1 = Male, 2 = Female): ").strip()
            if sex_in.lower() == 'quit': break
            patient_sex = int(sex_in)

            # --- CLINICAL DEMOGRAPHICS ---
            print("\n--- PRIMARY CANCER DIAGNOSIS ---")
            for i, c in enumerate(cancer_list):
                print(f"{i+1}. {c}")
            cancer_choice = input("Select Primary Cancer Type (1-12): ").strip()
            if cancer_choice.lower() == 'quit': break
            
            # Create a dictionary setting all cancers to 2 (No), then set the chosen one to 1 (Yes)
            patient_cancers = {col: 2 for col in cancer_features}
            if 1 <= int(cancer_choice) <= 12:
                selected_cancer_col = cancer_features[int(cancer_choice) - 1]
                patient_cancers[selected_cancer_col] = 1

            print("\n--- COMORBIDITIES ---")
            for i, d in enumerate(disease_list):
                print(f"{i+1}. {d}")
            disease_choice = input("Enter Comorbidities by number (comma separated, e.g., '1, 2, 8') or '0' for None: ").strip()
            if disease_choice.lower() == 'quit': break
            
            # Create a dictionary setting all diseases to 2 (No), then set the chosen ones to 1 (Yes)
            patient_diseases = {col: 2 for col in other_disease_features}
            if disease_choice != '0':
                choices = [int(x.strip()) for x in disease_choice.split(',') if x.strip().isdigit()]
                for choice in choices:
                    if 1 <= choice <= 12:
                        selected_disease_col = other_disease_features[choice - 1]
                        patient_diseases[selected_disease_col] = 1

            # Auto-calculate catastrophic cost risk
            catastrophic_cost = 1 if patient_totslf > (0.10 * patient_faminc) else 0

            # --- PACKAGE DATA FOR MODEL ---
            patient_data = {
                'FAMINC': [patient_faminc],
                'TOTSLF': [patient_totslf],
                'AGELAST': [patient_age],
                'CATASTROPHIC_COST': [catastrophic_cost],
                'SEX': [patient_sex]
            }
            # Merge the clinical dictionaries into the main patient data dictionary
            patient_data.update({k: [v] for k, v in patient_cancers.items()})
            patient_data.update({k: [v] for k, v in patient_diseases.items()})

            # Create DataFrame ensuring columns match exactly the training features order
            new_patient_df = pd.DataFrame(patient_data)[ml_features]

            # Predict
            recommended_subsidy = rf_model.predict(new_patient_df)[0]

            # Output Recommendation
            cancer_name = cancer_list[int(cancer_choice) - 1] if 1 <= int(cancer_choice) <= 12 else "Unknown"
            
            print("\n" + "-" * 70)
            print(" BASELINE PATIENT PROFILE:")
            print(f" Demographics: Age {patient_age} | Sex: {'Male' if patient_sex == 1 else 'Female'}")
            print(f" Clinical: {cancer_name} Cancer | Comorbidities Logged: {'None' if disease_choice == '0' else disease_choice}")
            print(f" Financial: Income ${patient_faminc:,.2f} | Expected OOP: ${patient_totslf:,.2f}")
            print(f" Catastrophic Cost Risk (>10% income): {'YES' if catastrophic_cost == 1 else 'NO'}")
            print("-" * 70)
            print(f">>> RECOMMENDED PROACTIVE SUBSIDY: ${recommended_subsidy:,.2f} <<<")
            print("    (Amount required to statistically ensure treatment adherence)")
            print("-" * 70 + "\n")
            
            run_again = input("Calculate for another patient? (y/n): ").strip().lower()
            if run_again != 'y':
                break
            print("\n" + "="*80 + "\n")

        except ValueError:
            print("\n[ERROR] Invalid input. Please enter numbers appropriately.\n")
            print("="*80 + "\n")

    print("\nExiting Clinical Decision Support Tool. Goodbye!")

# Launch the interactive tool
run_subsidy_calculator()
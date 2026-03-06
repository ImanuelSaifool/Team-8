import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------------------------------------------------------------------------
df2014 = pd.read_sas("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/h171.ssp", format='xport', encoding='utf-8')
df2015 = pd.read_sas("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/h181.ssp", format='xport', encoding='utf-8')
df2016 = pd.read_sas("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/h192.ssp", format='xport', encoding='utf-8')
df2017 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/h201.csv")
df2018 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/h209.csv")
df2019 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/h216.csv")
df2020 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/H224.csv")
df2021 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/h233.csv")
df2022 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/h243.csv")
df2023 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/h251.csv")
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2. STANDARDIZING
# ----------------------------------------------------------------------------------------------------------------------------------------------

# Treating inflation
df2014["FAMINC14"] = df2014["FAMINC14"] * 1.30
df2014["TOTSLF14"] = df2014["TOTSLF14"] * 1.30

df2015["FAMINC15"] = df2015["FAMINC15"] * 1.28
df2015["TOTSLF15"] = df2015["TOTSLF15"] * 1.28

df2016["FAMINC16"] = df2016["FAMINC16"] * 1.26
df2016["TOTSLF16"] = df2016["TOTSLF16"] * 1.26

df2017["FAMINC17"] = df2017["FAMINC17"] * 1.25
df2017["TOTSLF17"] = df2017["TOTSLF17"] * 1.25

df2018["FAMINC18"] = df2018["FAMINC18"] * 1.22
df2018["TOTSLF18"] = df2018["TOTSLF18"] * 1.22

df2019["FAMINC19"] = df2019["FAMINC19"] * 1.19
df2019["TOTSLF19"] = df2019["TOTSLF19"] * 1.19

df2020["FAMINC20"] = df2020["FAMINC20"] * 1.17
df2020["TOTSLF20"] = df2020["TOTSLF20"] * 1.17

df2021['FAMINC'] = df2021['FAMINC'] * 1.12
df2021['TOTSLF'] = df2021['TOTSLF'] * 1.12

df2022['FAMINC'] = df2022['FAMINC'] * 1.04
df2022['TOTSLF'] = df2022['TOTSLF'] * 1.04

df2014 = df2014.rename(columns={"TOTSLF14": "TOTSLF", "FAMINC14": "FAMINC", "TOTMCD14": "TOTMCD", "TOTMCR14": "TOTMCR", "TOTVA14": "TOTVA", "TOTTRI14": "TOTTRI", "TOTOFD14": "TOTOFD", "TOTSTL14": "TOTSTL", "REGION14": "REGION", "PRVEV14": "PRVEV", "POVCAT14": "POVCAT", "FOODST14": "FOODST", "DDNWRK14": "DDNWRK", "FAMSZE14": "FAMSZE"})
df2015 = df2015.rename(columns={"TOTSLF15": "TOTSLF", "FAMINC15": "FAMINC", "TOTMCD15": "TOTMCD", "TOTMCR15": "TOTMCR", "TOTVA15": "TOTVA", "TOTTRI15": "TOTTRI", "TOTOFD15": "TOTOFD", "TOTSTL15": "TOTSTL", "REGION15": "REGION", "PRVEV15": "PRVEV", "POVCAT15": "POVCAT", "FOODST15": "FOODST", "DDNWRK15": "DDNWRK", "FAMSZE15": "FAMSZE"})
df2016 = df2016.rename(columns={"TOTSLF16": "TOTSLF", "FAMINC16": "FAMINC", "TOTMCD16": "TOTMCD", "TOTMCR16": "TOTMCR", "TOTVA16": "TOTVA", "TOTTRI16": "TOTTRI", "TOTOFD16": "TOTOFD", "TOTSTL16": "TOTSTL", "REGION16": "REGION", "PRVEV16": "PRVEV", "POVCAT16": "POVCAT", "FOODST16": "FOODST", "DDNWRK16": "DDNWRK", "FAMSZE16": "FAMSZE"})
df2017 = df2017.rename(columns={"TOTSLF17": "TOTSLF", "FAMINC17": "FAMINC", "TOTMCD17": "TOTMCD", "TOTMCR17": "TOTMCR", "TOTVA17": "TOTVA", "TOTTRI17": "TOTTRI", "TOTOFD17": "TOTOFD", "TOTSTL17": "TOTSTL", "REGION17": "REGION", "PRVEV17": "PRVEV", "POVCAT17": "POVCAT", "FOODST17": "FOODST", "DDNWRK17": "DDNWRK", "FAMSZE17": "FAMSZE"})
df2018 = df2018.rename(columns={"TOTSLF18": "TOTSLF", "FAMINC18": "FAMINC", "TOTMCD18": "TOTMCD", "TOTMCR18": "TOTMCR", "TOTVA18": "TOTVA", "TOTTRI18": "TOTTRI", "TOTOFD18": "TOTOFD", "TOTSTL18": "TOTSTL", "REGION18": "REGION", "PRVEV18": "PRVEV", "POVCAT18": "POVCAT", "FOODST18": "FOODST", "DDNWRK18": "DDNWRK", "FAMSZE18": "FAMSZE"})
df2019 = df2019.rename(columns={"TOTSLF19": "TOTSLF", "FAMINC19": "FAMINC", "TOTMCD19": "TOTMCD", "TOTMCR19": "TOTMCR", "TOTVA19": "TOTVA", "TOTTRI19": "TOTTRI", "TOTOFD19": "TOTOFD", "TOTSTL19": "TOTSTL", "REGION19": "REGION", "PRVEV19": "PRVEV", "POVCAT19": "POVCAT", "FOODST19": "FOODST", "DDNWRK19": "DDNWRK", "FAMSZE19": "FAMSZE"})
df2020 = df2020.rename(columns={"TOTSLF20": "TOTSLF", "FAMINC20": "FAMINC", "TOTMCD20": "TOTMCD", "TOTMCR20": "TOTMCR", "TOTVA20": "TOTVA", "TOTTRI20": "TOTTRI", "TOTOFD20": "TOTOFD", "TOTSTL20": "TOTSTL", "REGION20": "REGION", "PRVEV20": "PRVEV", "POVCAT20": "POVCAT", "FOODST20": "FOODST", "DDNWRK20": "DDNWRK", "FAMSZE20": "FAMSZE"})
df2021 = df2021.rename(columns={"TOTSLF21": "TOTSLF", "FAMINC21": "FAMINC", "TOTMCD21": "TOTMCD", "TOTMCR21": "TOTMCR", "TOTVA21": "TOTVA", "TOTTRI21": "TOTTRI", "TOTOFD21": "TOTOFD", "TOTSTL21": "TOTSTL", "REGION21": "REGION", "PRVEV21": "PRVEV", "POVCAT21": "POVCAT", "FOODST21": "FOODST", "DDNWRK21": "DDNWRK", "FAMSZE21": "FAMSZE"})
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF", "FAMINC22": "FAMINC", "TOTMCD22": "TOTMCD", "TOTMCR22": "TOTMCR", "TOTVA22": "TOTVA", "TOTTRI22": "TOTTRI", "TOTOFD22": "TOTOFD", "TOTSTL22": "TOTSTL", "REGION22": "REGION", "PRVEV22": "PRVEV", "POVCAT22": "POVCAT", "FOODST22": "FOODST", "DDNWRK22": "DDNWRK", "FAMSZE22": "FAMSZE"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF", "FAMINC23": "FAMINC", "TOTMCD23": "TOTMCD", "TOTMCR23": "TOTMCR", "TOTVA23": "TOTVA", "TOTTRI23": "TOTTRI", "TOTOFD23": "TOTOFD", "TOTSTL23": "TOTSTL", "REGION23": "REGION", "PRVEV23": "PRVEV", "POVCAT23": "POVCAT", "FOODST23": "FOODST", "DDNWRK23": "DDNWRK", "FAMSZE23": "FAMSZE"})

main_df = pd.concat([df2014, df2015, df2016, df2017, df2018, df2019, df2020, df2021, df2022, df2023], axis=0)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 3. LABELLING
# ----------------------------------------------------------------------------------------------------------------------------------------------
    # 3a. Demog/Status -------------------------------------------------------------------------------------------------------------------------
demog_features = ["TOTSLF", "AGELAST", "SEX", "REGION", "RACEV1X", "RACEV2X"]
status_features = ["PSTAT31", "PSTAT43", "PSTAT53"]
marriage_features = ["MARRY31X", "MARRY42X", "MARRY53X"]
military_features = ["ACTDTY31", "ACTDTY42", "ACTDTY53"]
work_features = ["DDNWRK23", "EMPST31", "EMPST42", "EMPST53", "EVRWRK"]
income_features = ["TTLP23X", "FAMINC", "POVCAT23", "POVLEV23"]
wage_features = ["HRWG31X", "HRWG42X", "HRWG53X"]
how_wage_features = ["HRHOW31", "HRHOW42", "HRHOW53"]
wage_change_features = ["DIFFWG31", "DIFFWG42", "DIFFWG53"]
foodstamps_features = ["FOODST23", "FOODMN23", "FOODVL23"]
social_security_features = ["SSECP23X"]
veteran_income = ["VETSP23X"]
child_support_income = ["CHLDP23X"]

    # 3b. Cancer -------------------------------------------------------------------------------------------------------------------------------
cancer_features = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER", "CAPROSTA", "CASKINNM", "CASKINDK", "CAUTERUS"]

    # 3c. Comorbidities ------------------------------------------------------------------------------------------------------------------------
other_disease_features = ["OHRTDX", "STRKDX", "CHOLDX", "EMPHDX", "ASTHDX", "CHBRON31", "ARTHDX"]
coronary_heart_disease_features = ["CHDDX", "CHDAGED"]
high_blood_pressure_features = ["HIBPAGED", "HIBPDX", "BPMLDX"]
angina_features = ["ANGIDX", "ANGIAGED"]
heart_attack_features = ["MIDX", "MIAGED", "OHRTAGED", "OHRTTYPE"]
diabetes_features = ["DIABDX_M18", "DIABAGED"]
joint_features = ["JTPAIN31_M18", "ARTHDX", "ARTHTYPE", "ARTHAGED"]
asthma_features = ["ASTHDX", "ASTHAGED", "ASSTIL31", "ASATAK31", "ASTHEP31"]
adhd_features = ["ADHDADDX", "ADHDAGED"]
meantal_health_features = ["MNHLTH31", "MNHLTH42", "MNHLTH53", "UNHAP42", "ADSAD42" "HAVFUN42", "NERVAF42", "ADNERV42", "ADHOPE42", "ADWRTH42", "ADDPRS42", "PHQ242"]
physical_ability_features = ["ACTLIM31", "AIDHLP31", "WLKLIM31", "LFTDIF31", "STPDIF31", "WLKDIF31", "MILDIF31", "STNDIF31", "BENDIF31", "RCHDIF31", "FNGRDF31"]

    # 3d. Insurance ---------------------------------------------------------------------------------------------------------------------------
insurance_coverage_features = ["INSCOV23", "INSURC23"]
insurance_type_features = ["TOTMCD", "TOTMCR", "TOTVA", "TOTTRI", "TOTOFD", "TOTSTL", "ADINSA42"]
anytime_cover = ["TRICH31X", "TRICH42X"]
medicaid = ["TOTMCD"]
adherance_features = ["DLAYCA42", "AFRDCA42", "DLAYPM42", "AFRDPM42"]
covered_by_tricare_features = ["TRIJA23X", "TRIFE23X", "TRIMA23X", "TRIAP23X", "TRIMY23X", "TRIJU23X", "TRIJL23X", "TRIAU23X", "TRISE23X", "TRIOC23X", "TRINO23X", "TRIDE23X"]
covered_by_medicare_features = ["MCRJA23", "MCRFE23", "MCRMA23", "MCRAP23", "MCRMY23", "MCRJU23", "MCRJL23", "MCRAU23", "MCRSE23", "MCROC23", "MCRNO23", "MCRDE23"]
covered_by_medicareorshcip_features = ["MCDJA23", "MCDFE23", "MCDMA23", "MCDAP23", "MCDMY23", "MCDJU23", "MCDJL23", "MCDAU23", "MCDSE23", "MCDOC23", "MCDNO23", "MCDDE23"]

    # 3e. All ---------------------------------------------------------------------------------------------------------------------------
features = demog_features + cancer_features + other_disease_features + adherance_features + insurance_features

clean_df = main_df[(main_df['CANCERDX'] == 1) & (main_df['TOTMCD'] > 0)].copy()
clean_df = clean_df.drop_duplicates(subset=['DUPERSID'], keep='first')
clean_df = clean_df[(clean_df[demog_features] >= 0).all(axis=1)]
clean_df[cancer_features] = clean_df[cancer_features].replace([-1,-7, -8, -9], 2)

def clean_adherence(val):
    if val == 1: return 1  
    elif val == 2: return 0  
    else: return np.nan 

for col in adherance_features:
    clean_df[col] = clean_df[col].apply(clean_adherence)

clean_df = clean_df.dropna(subset=adherance_features)
clean_df['TOXICITY_SCORE'] = clean_df[adherance_features].sum(axis=1)

def calculate_toxicity_tier(row):
    if row['AFRDCA42'] == 1 or row['AFRDPM42'] == 1:
        return "Severe (Forgone Care/Meds)"
    elif row['DLAYCA42'] == 1 or row['DLAYPM42'] == 1:
        return "Moderate (Delayed Care/Meds)"
    else:
        return "None (Fully Adherent)"

clean_df['TOXICITY_TIER'] = clean_df.apply(calculate_toxicity_tier, axis=1)
clean_df['PUBLIC_TOTAL'] = clean_df[insurance_features].sum(axis=1)
clean_df['MCD_TOTAL'] = clean_df[medicaid].sum(axis=1)
clean_df['TOTAL_KNOWN_COST'] = clean_df['PUBLIC_TOTAL'] + clean_df['TOTSLF']
clean_df['COVERAGE_RATIO'] = clean_df['MCD_TOTAL'] / (clean_df['TOTAL_KNOWN_COST'] + 1e-9)
clean_df['COVERAGE_RATIO_PCT'] = clean_df['COVERAGE_RATIO'] * 100
clean_df['CATASTROPHIC_COST'] = (clean_df['TOTSLF'] > (0.10 * clean_df['FAMINC'])).astype(int)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 14.5 FEATURE ENGINEERING (Insurance & Geography Proxies)
# ----------------------------------------------------------------------------------------------------------------------------------------------
clean_df['IS_MEDICARE_AGE'] = (clean_df['AGELAST'] >= 65).astype(int)
clean_df['IS_CHIP_AGE'] = (clean_df['AGELAST'] <= 19).astype(int)
clean_df['IS_VETERAN'] = (clean_df['TOTVA'] > 0).astype(int)
clean_df['IS_MILITARY_FAM'] = (clean_df['TOTTRI'] > 0).astype(int)
clean_df['IS_FED_WORKER'] = (clean_df['TOTOFD'] > 0).astype(int)
clean_df['REGION_NORTHEAST'] = (clean_df['REGION'] == 1).astype(int)
clean_df['REGION_MIDWEST'] = (clean_df['REGION'] == 2).astype(int)
clean_df['REGION_SOUTH'] = (clean_df['REGION'] == 3).astype(int)
clean_df['REGION_WEST'] = (clean_df['REGION'] == 4).astype(int)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 14.6 FEATURE ENGINEERING (Safe Socioeconomic & Health Proxies)
# ----------------------------------------------------------------------------------------------------------------------------------------------
print("\n[*] Scanning dataset for MEPS socioeconomic and depression variables...")

target_extra_cols = {
    'FAMSZE': 'FAMILY_SIZE', 'PRVEV': 'HAS_PRIVATE_INS', 'RTHLTH': 'PERCEIVED_PHYS_HLTH', 
    'MNHLTH': 'PERCEIVED_MENTAL_HLTH', 'POVCAT': 'POVERTY_CATEGORY', 'FOODST': 'FOOD_STAMPS', 
    'EMPST': 'EMPLOYMENT_STATUS', 'DDNWRK': 'DAYS_MISSED_WORK', 'ADLHLP': 'ADL_HELP_NEEDED', 
    'PHQ2': 'PHQ2_DEPRESSION_SCORE' 
}

available_extras = []
for original_col, new_name in target_extra_cols.items():
    matching_cols = [c for c in clean_df.columns if original_col in c]
    if matching_cols:
        actual_col = matching_cols[0]
        # Clean negative codes, but DO NOT impute medians here to prevent data leakage
        clean_df[actual_col] = clean_df[actual_col].replace([-1, -7, -8, -9], np.nan)
        clean_df = clean_df.rename(columns={actual_col: new_name})
        available_extras.append(new_name)
        print(f"    - Found and cleaned: {new_name}")

if 'FAMILY_SIZE' in available_extras:
    clean_df['INCOME_PER_CAPITA'] = clean_df['FAMINC'] / clean_df['FAMILY_SIZE'].replace(0, 1)
    available_extras.append('INCOME_PER_CAPITA')
    print("    - Engineered: INCOME_PER_CAPITA")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 15. PREDICTIVE MODELING
# ----------------------------------------------------------------------------------------------------------------------------------------------
print("\n" + "="*80)
print("EVALUATING BIOSTATISTICAL MODELS (10-YEAR DATA)")
print("="*80)

# Binarize Target
clean_df['RISK_FLAG'] = clean_df['TOXICITY_TIER'].apply(lambda x: 0 if x == "None (Fully Adherent)" else 1)

regional_features = ['REGION_NORTHEAST', 'REGION_MIDWEST', 'REGION_SOUTH', 'REGION_WEST']
ml_features = [
    'FAMINC', 'AGELAST', 'SEX', 'IS_MEDICARE_AGE', 'IS_CHIP_AGE', 
    'IS_VETERAN', 'IS_MILITARY_FAM', 'IS_FED_WORKER'
] + regional_features + cancer_features + other_disease_features + available_extras

# Data Prep
ml_df = clean_df.dropna(subset=['RISK_FLAG']).copy()

# Ensure we drop any features that are 100% empty across the decade to prevent imputer crashes
clean_df = clean_df.dropna(axis=1, how='all')
ml_features = [col for col in ml_features if col in clean_df.columns]

X = ml_df[ml_features]
y = ml_df['RISK_FLAG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# We need an imputer AND a scaler. We fit them on training data ONLY to prevent leakage.
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Transform training data for the model bake-off
X_train_imp = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_imp = scaler.transform(imputer.transform(X_test))

# To store our models for comparison
models_dict = {}

print("\n[*] Training Model 1: Gradient Boosting (Manual 1:4 Weight Ratio)...")
manual_weights = np.where(y_train == 1, 4.0, 1.0) 
gb_manual = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
gb_manual.fit(X_train_imp, y_train, sample_weight=manual_weights)
models_dict['GB_Manual_Weights'] = gb_manual

print("[*] Training Model 2: Gradient Boosting (Random Undersampling)...")
train_df = pd.DataFrame(X_train_imp, columns=ml_features)
train_df['RISK_FLAG'] = y_train.values

at_risk_subset = train_df[train_df['RISK_FLAG'] == 1]
adherent_subset = train_df[train_df['RISK_FLAG'] == 0].sample(n=len(at_risk_subset), random_state=42)

undersampled_train = pd.concat([at_risk_subset, adherent_subset]).sample(frac=1, random_state=42) 
X_train_under = undersampled_train[ml_features]
y_train_under = undersampled_train['RISK_FLAG']

gb_under = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
gb_under.fit(X_train_under, y_train_under)
models_dict['GB_Undersampled'] = gb_under

print("[*] Training Model 3: Random Forest (Balanced Subsample)...")
rf_model = RandomForestClassifier(n_estimators=150, class_weight='balanced_subsample', random_state=42)
rf_model.fit(X_train_imp, y_train)
models_dict['Random_Forest'] = rf_model

print("[*] Training Model 4: Penalized Logistic Regression...")
log_reg = LogisticRegression(class_weight='balanced', max_iter=2000, solver='liblinear', random_state=42)
log_reg.fit(X_train_imp, y_train)
models_dict['Logistic_Regression'] = log_reg

# EVALUATION & COMPARISON
print("\n" + "="*80)
print(" MODEL COMPARISON RESULTS (Threshold = 50%)")
print("="*80)

for name, model in models_dict.items():
    y_pred = model.predict(X_test_imp)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) 
    specificity = tn / (tn + fp) 
    
    print(f"\n--- {name} ---")
    print(f"Sensitivity (Caught At-Risk): {sensitivity*100:.1f}%")
    print(f"Specificity (Correctly Adherent): {specificity*100:.1f}%")
    print(f"False Alarm Rate (FP Rate): {(1 - specificity)*100:.1f}%")

# Explicitly selecting Logistic Regression for its clinical interpretability
best_model_name = 'Logistic_Regression'
best_model = models_dict['Logistic_Regression']

print("\n" + "="*80)
print(f" WINNING MODEL SELECTED FOR CLINICAL TOOL: {best_model_name}")
print("="*80)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 15.5 EXTRACTING CLINICAL ODDS RATIOS
# ----------------------------------------------------------------------------------------------------------------------------------------------
print("\n[*] Extracting Odds Ratios for Clinical Interpretation...\n")

coefficients = best_model.coef_[0]
odds_ratios = np.exp(coefficients)

or_df = pd.DataFrame({
    'Feature': ml_features,
    'Coefficient': coefficients,
    'Odds_Ratio': odds_ratios
})

or_df = or_df.sort_values(by='Odds_Ratio', ascending=False)

print("Top 10 Risk Multipliers (OR > 1 means HIGHER risk of non-adherence):")
print("-" * 65)
for index, row in or_df.head(10).iterrows():
    print(f"{row['Feature']:<25} | Odds Ratio: {row['Odds_Ratio']:.2f}x")

print("\nTop 5 Protective Factors (OR < 1 means LOWER risk of non-adherence):")
print("-" * 65)
for index, row in or_df.tail(5).sort_values(by='Odds_Ratio', ascending=True).iterrows():
    print(f"{row['Feature']:<25} | Odds Ratio: {row['Odds_Ratio']:.2f}x")
print("-" * 65)

# --- THE CRITICAL FIX FOR SECTION 16 ---
# Build a complete pipeline so raw user input gets imputed AND scaled before hitting the model
final_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', best_model)
])

# Fit the entire pipeline once on the original training data so it is ready for live predictions
final_pipeline.fit(X_train, y_train)

# Plot the winning model's confusion matrix
y_pred_final = final_pipeline.predict(X_test)
cm_final = confusion_matrix(y_test, y_pred_final, normalize='true')
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=["Adherent", "At-Risk"])
disp.plot(cmap='Blues', ax=ax, values_format='.2f') 
plt.title(f"Winning Model: {best_model_name}", fontsize=14, fontweight='bold')
plt.grid(False)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 17. ADVANCED MODEL EVALUATION (Clinical ML Standard)
# ----------------------------------------------------------------------------------------------------------------------------------------------

print("\n" + "="*80)
print(" ADVANCED MODEL PERFORMANCE ANALYSIS")
print("="*80)

# Predicted probabilities
y_probs = final_pipeline.predict_proba(X_test)[:,1]

# ------------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Financial Toxicity Risk Model")
plt.legend()
plt.grid()
plt.show()

print(f"ROC AUC: {roc_auc:.3f}")

# ------------------------------------------------------------------
# Precision Recall Curve
# ------------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)

plt.figure(figsize=(7,6))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid()
plt.show()

print(f"PR AUC: {pr_auc:.3f}")

# ------------------------------------------------------------------
# Calibration Plot
# ------------------------------------------------------------------
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

plt.figure(figsize=(7,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--')
plt.xlabel("Predicted Risk")
plt.ylabel("Observed Risk")
plt.title("Calibration Curve")
plt.grid()
plt.show()

# ------------------------------------------------------------------
# Predicted Risk Distribution
# ------------------------------------------------------------------
plt.figure(figsize=(8,6))
sns.histplot(y_probs[y_test==0], label="Adherent", bins=30, stat="density", kde=True)
sns.histplot(y_probs[y_test==1], label="At-Risk", bins=30, stat="density", kde=True)
plt.xlabel("Predicted Probability of Non-Adherence")
plt.title("Risk Score Distribution")
plt.legend()
plt.show()

# ------------------------------------------------------------------
# Odds Ratio Forest Plot
# ------------------------------------------------------------------
or_plot_df = or_df.copy()
or_plot_df = or_plot_df.sort_values("Odds_Ratio", ascending=False).head(15)

plt.figure(figsize=(8,10))
sns.barplot(
    x="Odds_Ratio",
    y="Feature",
    data=or_plot_df,
    orient="h"
)

plt.axvline(1, linestyle="--")
plt.title("Top Risk Factors for Treatment Non-Adherence")
plt.xlabel("Odds Ratio")
plt.ylabel("Clinical Feature")
plt.show()

# ------------------------------------------------------------------
# Model Performance Summary Table
# ------------------------------------------------------------------
y_pred = final_pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
precision_metric = tp/(tp+fp)
f1 = 2*(precision_metric*sensitivity)/(precision_metric+sensitivity)

summary_df = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Sensitivity (Recall)",
        "Specificity",
        "Precision",
        "F1 Score",
        "ROC AUC",
        "PR AUC"
    ],
    "Value":[
        accuracy,
        sensitivity,
        specificity,
        precision_metric,
        f1,
        roc_auc,
        pr_auc
    ]
})

print("\nModel Performance Summary")
print(summary_df)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 16. INTERACTIVE CLINICAL DECISION SUPPORT TOOL (Fixed Pipeline Integration)
# ----------------------------------------------------------------------------------------------------------------------------------------------

def run_risk_screener():
    print("\n" + "="*80)
    print(" CLINICAL DECISION SUPPORT: DAY-1 TOXICITY RISK SCREENER")
    print("="*80)
    print("Type 'quit' at any prompt to exit the tool.\n")

    cancer_list = ["Bladder", "Breast", "Cervix", "Colon", "Lung", "Lymphoma", "Melanoma", "Other", "Prostate", "Skin (Non-Melanoma)", "Skin (Unknown)", "Uterus"]
    disease_list = ["Diabetes", "High Blood Pressure", "Coronary Heart Disease", "Angina", "Heart Attack", "Other Heart Disease", "Stroke", "High Cholesterol", "Emphysema", "Asthma", "Chronic Bronchitis", "Arthritis"]
    region_names = ["Northeast", "Midwest", "South", "West"]

    while True:
        try:
            # --- ECONOMIC DEMOGRAPHICS ---
            faminc_in = input("Enter Patient's Current Family Income ($): ").strip().replace(',', '').replace('$', '')
            if faminc_in.lower() == 'quit': break
            patient_faminc = float(faminc_in)

            age_in = input("Enter Patient's Current Age: ").strip()
            if age_in.lower() == 'quit': break
            patient_age = int(age_in)

            sex_in = input("Enter Assigned Sex (1 = Male, 2 = Female): ").strip()
            if sex_in.lower() == 'quit': break
            patient_sex = int(sex_in)

            # --- DYNAMIC SOCIOECONOMIC & DEPRESSION INPUTS ---
            print("\n--- SOCIAL DETERMINANTS OF HEALTH (SDoH) ---")
            
            patient_famsze = 1
            if 'FAMILY_SIZE' in available_extras:
                fs_in = input("Enter Family Size (number of people in household): ").strip()
                if fs_in.lower() == 'quit': break
                patient_famsze = int(fs_in) if fs_in.isdigit() else 1

            patient_prv = 2
            if 'HAS_PRIVATE_INS' in available_extras:
                prv_in = input("Does the patient have any Private Insurance? (y/n): ").strip().lower()
                if prv_in == 'quit': break
                patient_prv = 1 if prv_in == 'y' else 2
                
            patient_pov = 3
            if 'POVERTY_CATEGORY' in available_extras:
                pov_in = input("Enter Poverty Category (1=Poor to 5=High Income): ").strip()
                if pov_in.lower() == 'quit': break
                patient_pov = int(pov_in) if pov_in.isdigit() else 3

            patient_foodst = 2
            if 'FOOD_STAMPS' in available_extras:
                fs_in = input("Does the patient receive Food Stamps/SNAP? (y/n): ").strip().lower()
                if fs_in == 'quit': break
                patient_foodst = 1 if fs_in == 'y' else 2
                
            patient_ddnwrk = 0
            if 'DAYS_MISSED_WORK' in available_extras:
                dw_in = input("Estimated days of work missed due to illness this year: ").strip()
                if dw_in.lower() == 'quit': break
                patient_ddnwrk = int(dw_in) if dw_in.isdigit() else 0

            patient_adl = 2
            if 'ADL_HELP_NEEDED' in available_extras:
                adl_in = input("Does the patient need help with daily activities (bathing, etc)? (y/n): ").strip().lower()
                if adl_in == 'quit': break
                patient_adl = 1 if adl_in == 'y' else 2
                
            patient_phq2 = 0
            if 'PHQ2_DEPRESSION_SCORE' in available_extras:
                phq_in = input("PHQ-2 Depression Score (0 to 6): ").strip()
                if phq_in.lower() == 'quit': break
                patient_phq2 = int(phq_in) if phq_in.isdigit() else 0

            patient_ph, patient_mh, patient_empst = 3, 3, 1

            # --- REGIONAL DATA ---
            print("\n--- GEOGRAPHY ---")
            for i, r in enumerate(region_names): print(f"{i+1}. {r}")
            region_choice = input("Select Patient's US Region (1-4): ").strip()
            if region_choice.lower() == 'quit': break
            region_idx = int(region_choice)
            
            patient_region = {
                'REGION_NORTHEAST': [1 if region_idx == 1 else 0],
                'REGION_MIDWEST': [1 if region_idx == 2 else 0],
                'REGION_SOUTH': [1 if region_idx == 3 else 0],
                'REGION_WEST': [1 if region_idx == 4 else 0]
            }

            # --- INSURANCE ELIGIBILITY DEMOGRAPHICS ---
            vet_in = input("Is the patient a US Veteran? (y/n): ").strip().lower()
            if vet_in == 'quit': break
            patient_vet = 1 if vet_in == 'y' else 0

            mil_in = input("Is the patient/family in the military [Tricare eligible]? (y/n): ").strip().lower()
            if mil_in == 'quit': break
            patient_mil = 1 if mil_in == 'y' else 0

            fed_in = input("Does the patient work for the Federal Government? (y/n): ").strip().lower()
            if fed_in == 'quit': break
            patient_fed = 1 if fed_in == 'y' else 0

            patient_medicare = 1 if patient_age >= 65 else 0
            patient_chip = 1 if patient_age <= 19 else 0

            # --- CLINICAL DEMOGRAPHICS ---
            print("\n--- PRIMARY CANCER DIAGNOSIS ---")
            for i, c in enumerate(cancer_list): print(f"{i+1}. {c}")
            cancer_choice = input("Select Primary Cancer Type (1-12): ").strip()
            if cancer_choice.lower() == 'quit': break
            
            patient_cancers = {col: 2 for col in cancer_features}
            if 1 <= int(cancer_choice) <= 12:
                selected_cancer_col = cancer_features[int(cancer_choice) - 1]
                patient_cancers[selected_cancer_col] = 1

            print("\n--- COMORBIDITIES ---")
            for i, d in enumerate(disease_list): print(f"{i+1}. {d}")
            disease_choice = input("Enter Comorbidities by number (comma separated, e.g., '1, 2, 8') or '0' for None: ").strip()
            if disease_choice.lower() == 'quit': break
            
            patient_diseases = {col: 2 for col in other_disease_features}
            if disease_choice != '0':
                choices = [int(x.strip()) for x in disease_choice.split(',') if x.strip().isdigit()]
                for choice in choices:
                    if 1 <= choice <= 12:
                        selected_disease_col = other_disease_features[choice - 1]
                        patient_diseases[selected_disease_col] = 1

            # --- PACKAGE DATA FOR MODEL ---
            patient_data = {
                'FAMINC': [patient_faminc], 'AGELAST': [patient_age], 'SEX': [patient_sex],
                'IS_MEDICARE_AGE': [patient_medicare], 'IS_CHIP_AGE': [patient_chip],
                'IS_VETERAN': [patient_vet], 'IS_MILITARY_FAM': [patient_mil], 'IS_FED_WORKER': [patient_fed]
            }
            
            if 'FAMILY_SIZE' in available_extras:
                patient_data['FAMILY_SIZE'] = [patient_famsze]
                patient_data['INCOME_PER_CAPITA'] = [patient_faminc / max(1, patient_famsze)]
            if 'HAS_PRIVATE_INS' in available_extras: patient_data['HAS_PRIVATE_INS'] = [patient_prv]
            if 'PERCEIVED_PHYS_HLTH' in available_extras: patient_data['PERCEIVED_PHYS_HLTH'] = [patient_ph]
            if 'PERCEIVED_MENTAL_HLTH' in available_extras: patient_data['PERCEIVED_MENTAL_HLTH'] = [patient_mh]
            if 'POVERTY_CATEGORY' in available_extras: patient_data['POVERTY_CATEGORY'] = [patient_pov]
            if 'FOOD_STAMPS' in available_extras: patient_data['FOOD_STAMPS'] = [patient_foodst]
            if 'DAYS_MISSED_WORK' in available_extras: patient_data['DAYS_MISSED_WORK'] = [patient_ddnwrk]
            if 'ADL_HELP_NEEDED' in available_extras: patient_data['ADL_HELP_NEEDED'] = [patient_adl]
            if 'EMPLOYMENT_STATUS' in available_extras: patient_data['EMPLOYMENT_STATUS'] = [patient_empst]
            if 'PHQ2_DEPRESSION_SCORE' in available_extras: patient_data['PHQ2_DEPRESSION_SCORE'] = [patient_phq2]

            patient_data.update(patient_region)
            patient_data.update({k: [v] for k, v in patient_cancers.items()})
            patient_data.update({k: [v] for k, v in patient_diseases.items()})

            new_patient_df = pd.DataFrame(patient_data)[ml_features]

            # --- PREDICT CLASSIFICATION AND PROBABILITIES ---
            predicted_tier = final_pipeline.predict(new_patient_df)[0]
            probabilities = final_pipeline.predict_proba(new_patient_df)[0]
            
            prob_dict = {"0: Adherent": probabilities[0], "1: At-Risk": probabilities[1]}

            cancer_name = cancer_list[int(cancer_choice) - 1] if 1 <= int(cancer_choice) <= 12 else "Unknown"
            region_name = region_names[region_idx - 1] if 1 <= region_idx <= 4 else "Unknown"
            
            print("\n" + "-" * 80)
            print(" PROGNOSTIC PATIENT PROFILE:")
            print(f" Demographics: Age {patient_age} | Sex: {'Male' if patient_sex == 1 else 'Female'} | Region: {region_name}")
            print(f" Clinical: {cancer_name} Cancer | Comorbidities Logged: {'None' if disease_choice == '0' else disease_choice}")
            print(f" Financial: Income ${patient_faminc:,.2f} | Household Size: {patient_famsze}")
            print(f" Overlapping Coverage: Vet({vet_in.upper()}) | Mil({mil_in.upper()}) | Fed({fed_in.upper()}) | Medicare({'Y' if patient_medicare else 'N'})")
            if 'PHQ2_DEPRESSION_SCORE' in available_extras:
                print(f" SDoH Flags: PHQ-2 Score [{patient_phq2}] | Missed Work [{patient_ddnwrk} Days] | ADL Help [{'YES' if patient_adl == 1 else 'NO'}]")
            print("-" * 80)
            
            if predicted_tier == 1:
                print(f">>> ALERT: HIGH RISK OF TREATMENT ABANDONMENT/DELAY <<<")
                tier_label = "AT-RISK"
            else:
                print(f">>> CLEAR: LOW RISK OF NON-ADHERENCE <<<")
                tier_label = "ADHERENT"
                
            print(f" Predicted Category: {tier_label}")
            print("\n Statistical Confidence Profile:")
            for tier, prob in prob_dict.items():
                print(f"  - {tier}: {prob*100:.1f}%")
            print("-" * 80 + "\n")
            
            run_again = input("Screen another patient? (y/n): ").strip().lower()
            if run_again != 'y':
                break
            print("\n" + "="*80 + "\n")

        except ValueError:
            print("\n[ERROR] Invalid input. Please enter numbers appropriately.\n")
            print("="*80 + "\n")

    print("\nExiting Clinical Decision Support Tool. Goodbye!")

if __name__ == "__main__":
    run_risk_screener()
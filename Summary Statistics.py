import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.feature_selection import mutual_info_classif

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------------------------------
df2019 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/OneDrive/Desktop/Coding%20Projects/h216.csv")
df2020 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/OneDrive/Desktop/Coding%20Projects/H224.csv")
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2. Standardizing column names
    # we do this so that we can easily integrate multiple data files without changing the name on the raw data
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
medicaid = ["TOTMCD"]
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
clean_df['PUBLIC_TOTAL'] = clean_df[insurance_features].sum(axis=1)
clean_df['MCD_TOTAL'] = clean_df[medicaid].sum(axis=1)

# Total Known Cost
clean_df['TOTAL_KNOWN_COST'] = clean_df['PUBLIC_TOTAL'] + clean_df['TOTSLF']

# 2. Calculate the Coverage Ratio
clean_df['COVERAGE_RATIO'] = clean_df['MCD_TOTAL'] / (clean_df['TOTAL_KNOWN_COST'] + 1e-9)
clean_df['COVERAGE_RATIO_PCT'] = clean_df['COVERAGE_RATIO'] * 100

# ----------------------------------------------------------------------------------------------------------------------------------------------
clean_df['CATASTROPHIC_COST'] = (clean_df['TOTSLF'] > (0.10 * clean_df['FAMINC'])).astype(int)
# ----------------------------------------------------------------------------------------------------------------------------------------------
cancer_map = {
    "CABLADDR": "Bladder Cancer",
    "CABREAST": "Breast Cancer",
    "CACERVIX": "Cervix Cancer",
    "CACOLON": "Colon Cancer",
    "CALUNG": "Lung Cancer",
    "CALYMPH": "Lymph Cancer",
    "CAMELANO": "Melano Cancer",
    "CAOTHER": "Other Cancer",
    "CAPROSTA": "Prostate Cancer",
    "CASKINNM": "Skin Cancer 1",
    "CASKINDK": "SKin Cancer 2",
    "CAUTERUS": "Uterus Cancer"
}

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
# 4. SUMMARY STATISTICS
# ----------------------------------------------------------------------------------------------------------------------------------------------

print("--- GENERAL SUMMARY STATISTICS (General) ---")
print(clean_df[features].describe()) 

# Grouping by the new TOXICITY_SCORE instead of UNABLE
print("\n--- Average Out-of-Pocket Cost by Toxicity Score ---")
print(clean_df.groupby('TOXICITY_SCORE')['TOTSLF'].mean())

print("\n--- Average Family Income by Toxicity Score ---")
print(clean_df.groupby('TOXICITY_SCORE')['FAMINC'].mean())

print("\n--- Average Public Insurance Coverage by Toxicity Score ---")
print(clean_df.groupby('TOXICITY_SCORE')['PUBLIC_TOTAL'].mean())

summary_list = []
total_patients = len(clean_df)

for col, name in disease_map.items():
    if col in clean_df.columns:
        # 1. Filter for people with this disease
        disease_subgroup = clean_df[clean_df[col] == 1]
        
        # 2. Calculate Stats
        num_patients = len(disease_subgroup)
        percent = (num_patients / total_patients) * 100
        
        # Calculate AVERAGE cost/income for this group 
        avg_oop = disease_subgroup['TOTSLF'].mean()
        avg_income = disease_subgroup['FAMINC'].mean()
        avg_public = disease_subgroup['PUBLIC_TOTAL'].mean()

        # Added: Average Toxicity Score for patients with this comorbidity
        avg_toxicity = disease_subgroup['TOXICITY_SCORE'].mean()

        summary_list.append({
            "Comorbidity": name,
            "Count (N)": num_patients,
            "Prevalence (%)": round(percent, 2),
            "Avg OOP Cost ($)": round(avg_oop, 2),
            "Avg Family Income ($)": round(avg_income, 2),
            "Avg Public Pay ($)": round(avg_public, 2),
            "Avg Toxicity Score": round(avg_toxicity, 2)
        })

# Optional: Print the comorbidity summary nicely
summary_df = pd.DataFrame(summary_list).sort_values(by="Avg Toxicity Score", ascending=False)
print("\n--- Comorbidity Summary (Sorted by Worst Adherence) ---")
print(summary_df)

print("\n--- PATIENT COUNT BY TOXICITY TIER ---")
# 1. Get the raw number of patients in each category
tier_counts = clean_df['TOXICITY_TIER'].value_counts()
print(tier_counts)

print("\n--- PERCENTAGE BREAKDOWN ---")
# 2. Get the exact percentage to see the severity of the imbalance
tier_percentages = clean_df['TOXICITY_TIER'].value_counts(normalize=True) * 100
print(tier_percentages.round(2).astype(str) + '%')
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 5. V(Heatmap)
# ----------------------------------------------------------------------------------------------------------------------------------------------
X = clean_df[['FAMINC', 'AGELAST', 'COVERAGE_RATIO_PCT', 'CATASTROPHIC_COST']]
y = clean_df['TOXICITY_SCORE']

mi_scores = mutual_info_classif(X, y, random_state=42)

print("\n--- Mutual Information Scores (Target: TOXICITY_SCORE) ---")
for feature, score in zip(X.columns, mi_scores):
    print(f"{feature}: {score:.4f}")

plt.figure(figsize=(10, 8))

corr_features = ['TOXICITY_SCORE', 'TOTSLF', 'FAMINC', 'PUBLIC_TOTAL', 'COVERAGE_RATIO_PCT', 'CATASTROPHIC_COST', 'AGELAST']
corr_data = clean_df[corr_features].corr(method='spearman')

sns.heatmap(
    corr_data, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    linewidths=0.5,
    vmin=-1, vmax=1
)

plt.title("Correlation Heatmap: Economic Drivers of Non-Adherence", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 6. V(Boxplot of income vs cost)
# ----------------------------------------------------------------------------------------------------------------------------------------------
plt.rcParams['figure.figsize'] = (16, 6)

# We use boxplots to compare the distributions across the new Toxicity Tiers.
# Note: We use 'showfliers=False' because MEPS has massive outliers that ruin the visual scale.

fig, axes = plt.subplots(1, 2)
tier_order = ["None (Fully Adherent)", "Moderate (Delayed Care/Meds)", "Severe (Forgone Care/Meds)"]

# Plot 1: Out-of-Pocket Costs (Fixed the bug here: y='TOTSLF')
sns.boxplot(
    data=clean_df, 
    x='TOXICITY_TIER', 
    y='TOTSLF', 
    ax=axes[0], 
    order=tier_order,
    showfliers=False, 
    palette="Reds"
)
axes[0].set_title("Impact of Out-of-Pocket Costs on Adherence Severity", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Financial Toxicity Tier")
axes[0].set_ylabel("Total Out-of-Pocket Cost ($)")

# Plot 2: Family Income
sns.boxplot(
    data=clean_df, 
    x='TOXICITY_TIER', 
    y='FAMINC', 
    ax=axes[1], 
    order=tier_order,
    showfliers=False, 
    palette="Greens"
)
axes[1].set_title("Protective Effect of Income on Adherence", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Financial Toxicity Tier")
axes[1].set_ylabel("Family Income ($)")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 7. V(Comorbidity Multiplier)
#       Does having Diabetes or Heart Disease ALONG with Cancer increase financial risk?
# ----------------------------------------------------------------------------------------------------------------------------------------------

risk_data = []

# Calculate the baseline rate of ANY financial toxicity (Score > 0) for the average cancer patient
baseline_rate = (clean_df['TOXICITY_SCORE'] > 0).mean() * 100 

for code, name in disease_map.items():
    if code in clean_df.columns:
        # Get patients with this specific disease
        subset = clean_df[clean_df[code] == 1]
        
        # Calculate % who have ANY financial adherence issue (Score > 0)
        risk = (subset['TOXICITY_SCORE'] > 0).mean() * 100
        
        # We can also capture the average severity score just to have it
        avg_score = subset['TOXICITY_SCORE'].mean()
        
        risk_data.append({
            'Condition': name, 
            'Risk_Percentage': risk, 
            'Avg_Severity': avg_score
        })

# Sort by the highest risk percentage
risk_df = pd.DataFrame(risk_data).sort_values('Risk_Percentage', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=risk_df, x='Risk_Percentage', y='Condition', palette="magma")

# Draw the baseline red dashed line for comparison
plt.axvline(x=baseline_rate, color='red', linestyle='--', label=f'Avg Cancer Patient ({baseline_rate:.1f}%)')

plt.title("Financial Toxicity Rate by Comorbidity", fontsize=14, fontweight='bold')
plt.xlabel("Percentage of Patients Reporting Financial Adherence Issues (%)")
plt.ylabel("") # Hiding the Y-axis label since the condition names are self-explanatory
plt.legend()
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------
# 8. V(Difference Between Age)
# ----------------------------------------------------------------------------------------------------------------------------------------------
from matplotlib.ticker import PercentFormatter

clean_df["Age Group"] = pd.cut( 
    clean_df["AGELAST"], 
    bins=[0, 17, 34, 49, 64, 120], 
    labels=["0–17", "18–34", "35–49", "50–64", "65+"] 
)

clean_df['ANY_TOXICITY'] = (clean_df['TOXICITY_SCORE'] > 0).astype(int)

plt.figure(figsize=(8,5)) 
sns.barplot(
    data=clean_df, 
    x="Age Group", 
    y="ANY_TOXICITY", 
    estimator="mean", 
    errorbar=None,
    palette="Blues"
) 
plt.title("Financial Toxicity Rate by Age Group", fontsize=14, fontweight='bold') 
plt.ylabel("Percentage Reporting Adherence Issues (%)") 
plt.xlabel("Age Group") 
plt.ylim(0,1) 
plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) 
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 9. VISUALIZATIONS (Difference Between Sex)
# ----------------------------------------------------------------------------------------------------------------------------------------------
sns.set_style("whitegrid")

sex_map = {1: "Male", 2: "Female"}
clean_df["Assigned Sex"] = clean_df["SEX"].map(sex_map)

plt.figure(figsize=(6,4))
ax = sns.barplot(
    data=clean_df,
    x="Assigned Sex",
    y="ANY_TOXICITY",
    estimator="mean",
    errorbar=None,
    palette="Pastel1"
)
ax.set_title("Financial Toxicity Rate by Assigned Sex", fontsize=14, fontweight='bold')
ax.set_ylabel("Percentage Reporting Adherence Issues (%)")
ax.set_xlabel("Assigned Sex")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 9. V(Toxicity Tier)
# ----------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

# Boxplot showing how Family Income protects against different TIERS of toxicity
sns.boxplot(
    data=clean_df, 
    x='TOXICITY_TIER', 
    y='FAMINC', 
    order=["None (Fully Adherent)", "Moderate (Delayed Care/Meds)", "Severe (Forgone Care/Meds)"],
    palette="YlOrRd", 
    showfliers=False
)

plt.title("How Family Income Impacts Adherence Severity", fontsize=14)
plt.ylabel("Family Income ($)")
plt.xlabel("Financial Toxicity Tier")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 10. V(Insurance Efficacy & Coverage Ratios)
# ----------------------------------------------------------------------------------------------------------------------------------------------

# A KDE plot shows where patients are concentrated based on their coverage percentage
sns.kdeplot(
    data=clean_df, 
    x='COVERAGE_RATIO_PCT', 
    hue='TOXICITY_TIER', 
    fill=True, 
    common_norm=False, 
    palette='Set1',
    alpha=0.5,
    linewidth=2
)

plt.title("The 'Tipping Point': How Coverage Ratio Impacts Adherence", fontsize=14, fontweight='bold')
plt.xlabel("Percentage of Healthcare Bill Covered by Public Insurance (%)", fontsize=12)
plt.ylabel("Density of Patients", fontsize=12)
plt.xlim(0, 100) # Lock the X-axis from 0% to 100% coverage
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 11. VISUALIZATIONS (Cancer-Specific Financial Toxicity)
#        Which specific cancer diagnoses carry the highest economic risk?
# ----------------------------------------------------------------------------------------------------------------------------------------------

cancer_risk_data = []

# Calculate the baseline rate of ANY financial toxicity for the average cancer patient
baseline_rate = (clean_df['TOXICITY_SCORE'] > 0).mean() * 100 

for code, name in cancer_map.items():
    if code in clean_df.columns:
        # Filter for patients with this specific cancer (1 = Yes in MEPS)
        subset = clean_df[clean_df[code] == 1]
        
        # Make sure there are actually patients in this bucket to avoid division-by-zero errors
        if len(subset) > 0:
            # Calculate % who have ANY financial adherence issue
            risk = (subset['TOXICITY_SCORE'] > 0).mean() * 100
            cancer_risk_data.append({
                'Cancer_Type': name, 
                'Risk_Percentage': risk, 
                'Patient_Count': len(subset)
            })

# Convert to DataFrame and sort by highest risk
cancer_risk_df = pd.DataFrame(cancer_risk_data).sort_values('Risk_Percentage', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(data=cancer_risk_df, x='Risk_Percentage', y='Cancer_Type', palette="mako")

# Add the baseline average line so you can clearly see who is above/below average
plt.axvline(x=baseline_rate, color='red', linestyle='--', label=f'Avg Cancer Patient ({baseline_rate:.1f}%)')

plt.title("Financial Toxicity Rate by Specific Cancer Diagnosis", fontsize=14, fontweight='bold')
plt.xlabel("Percentage Reporting Financial Adherence Issues (%)")
plt.ylabel("") # Hiding Y-axis label since the cancer names are self-explanatory
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 12. VISUALIZATIONS (Catastrophic Health Expenditure)
#        How does crossing the WHO 10% threshold correlate with abandoning care?
# ----------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(8, 6))

ax = sns.barplot(
    data=clean_df, 
    x='TOXICITY_TIER', 
    y='CATASTROPHIC_COST', 
    order=["None (Fully Adherent)", "Moderate (Delayed Care/Meds)", "Severe (Forgone Care/Meds)"],
    palette="flare",
    estimator="mean",
    errorbar=None
)

plt.title("Catastrophic Health Expenditure (>10% of Income) by Adherence Tier", fontsize=14, fontweight='bold')
plt.ylabel("Percentage Facing Catastrophic Costs (%)")
plt.xlabel("Financial Toxicity Tier")
plt.ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))

# Add value labels on top of the bars for easy reading in a presentation
for p in ax.patches:
    ax.annotate(f"{p.get_height()*100:.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=11, color='black', xytext=(0, 5), 
                textcoords='offset points')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 13. VISUALIZATIONS (Regional Medicaid Disparities)
#        Does geographical location dictate the efficacy of public insurance?
# ----------------------------------------------------------------------------------------------------------------------------------------------

# Map the MEPS region codes to actual names
region_map = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
clean_df['Region_Name'] = clean_df['REGION'].map(region_map)

plt.figure(figsize=(9, 6))

ax = sns.barplot(
    data=clean_df, 
    x='Region_Name', 
    y='ANY_TOXICITY', 
    order=["Northeast", "Midwest", "South", "West"],
    palette="viridis",
    estimator="mean",
    errorbar=None
)

plt.title("Financial Toxicity Rates for Medicaid Cancer Patients by Region", fontsize=14, fontweight='bold')
plt.ylabel("Percentage Reporting Adherence Issues (%)")
plt.xlabel("US Region")
plt.ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))

# Add value labels on top of the bars
for p in ax.patches:
    ax.annotate(f"{p.get_height()*100:.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=11, color='black', xytext=(0, 5), 
                textcoords='offset points')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 14. VISUALIZATIONS (The Optimization Frontier)
#        At what intersection of Income and Medicaid Subsidy do patients abandon care?
# ----------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(10, 7))

# We filter out the extreme 1% of outliers in income and subsidy just so the scatter plot isn't squished
q_inc = clean_df['FAMINC'].quantile(0.95)
q_mcd = clean_df['PUBLIC_TOTAL'].quantile(0.95)

scatter_df = clean_df[(clean_df['FAMINC'] < q_inc) & (clean_df['PUBLIC_TOTAL'] < q_mcd)]

sns.scatterplot(
    data=scatter_df, 
    x='FAMINC', 
    y='PUBLIC_TOTAL', 
    hue='TOXICITY_TIER', 
    palette=['#2ca02c', '#ff7f0e', '#d62728'], # Green, Orange, Red
    alpha=0.7,
    s=60
)

plt.title("The Optimization Frontier: Income vs. Medicaid Subsidy", fontsize=14, fontweight='bold')
plt.xlabel("Family Income ($)")
plt.ylabel("Medicaid Subsidy Provided ($)")
plt.legend(title="Financial Toxicity Tier")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
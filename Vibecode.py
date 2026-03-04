import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier #for categorical data, you wanna use RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, accuracy_score
# ----------------------------------------------------------------------------------------------------------------------------------------------
# database
git fetch --all
df2021 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h251.csv")
df2022 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h243.csv")
df2023 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h233.csv")
main_df = pd.concat([df2021, df2022, df2023], axis=0)


# filter-rooster
cancer_df = main_df[main_df['CANCERDX'] == 1].copy() 
    # CHOLDX is a survey marker for whether their doctor have diagnosed them with cancer
    # In the survey, 1 is used to indicate "yes" while 2 is "no"
cancer_df = cancer_df[(cancer_df["FAMINC"] > 0) & (cancer_df["K6SUM42"] >= 0)]
    # FAMINC refers to the family income of the patient while K6SUM42 is a numerical predictor in the survey for mental distress
    # here, it is very important that we remove the zeroes in the data to clean it
    # notice that im using "&" only peak coders to that
# ----------------------------------------------------------------------------------------------------------------------------------------------
# =====================================================
# First: Does Cancer Cause Financial Issues?
# =====================================================
# define variables
y1 = cancer_df["TOTSLF"] # for good luck, you define the y first
x1 = cancer_df[["CANCERDX"]]

# splitting training and testing data (pareto's rule)
X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=88)

# do the random forest gahh
rf1 = RandomForestRegressor(n_estimators=100, random_state=88)

# fit the data first
rf1.fit(X1_train, y1_train)
importances = rf1.feature_importances_
print(f"Feature Importances (Income vs Cost): {importances}")

avg_cost_cancer = main_df[main_df['CANCERDX']==1]['TOTSLF'].mean()
avg_cost_healthy = main_df[main_df['CANCERDX']==2]['TOTSLF'].mean()

print(f"Average Cost for Cancer Patients: ${avg_cost_cancer:,.2f}")
print(f"Average Cost for Others:          ${avg_cost_healthy:,.2f}")
print(f"Link Confirmed: Cancer patients pay {avg_cost_cancer/avg_cost_healthy:.1f}x more.")

#---------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================
# Second: Does the Financial Issues Lead to Mental Distress?
# =========================================================

# define variables
y2 = cancer_df["K6SUM42"] # for good luck, you define the y first
x2 = cancer_df[["FAMINC", "TOTSLF"]]

# splitting training and testing data (pareto's rule)
X2_train, X2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=88)

# do the random forest gahh
rf2 = RandomForestRegressor(n_estimators=100, random_state=88)

# fit the data first
rf2.fit(X2_train, y2_train)
importances = rf2.feature_importances_
print(f"Feature Importances (Income vs Cost): {importances}")

#---------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================
# Third: Does the Financial Issues Lead to Depression?
# =========================================================
X3 = cancer_df[["K6SUM42"]]   # Predictor: Distress
y3 = cancer_df["UNABLE31"] # Target: Quitting

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
rf3 = RandomForestClassifier(n_estimators=100, random_state=88)
rf3.fit(X_train3, y_train3)

# Accuracy of this link
acc_link3 = accuracy_score(y_test3, rf3.predict(X_test3))
print(f"Model Accuracy (Using only Distress to predict Quitting): {acc_link3:.2%}")
#---------------------------------------------------------------------------------------------------------------------------------------------
# ==========================================
# FINAL VISUALIZATION
# ==========================================
links = ['Cancer -> $$', '$$ -> Depression', 'Depression -> Quit']
# For visualization, we use normalized scores or simple boolean confirmations
strengths = [100, rf2.feature_importances_[0]*100, acc_link3*100] 

plt.figure(figsize=(10, 5))
plt.plot(links, strengths, marker='o', linestyle='-', color='b')
plt.title("The Causal Chain: Path Analysis")
plt.show()
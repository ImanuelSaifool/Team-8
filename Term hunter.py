import pandas as pd

# Load just a tiny chunk of your 2023 file to check names
df = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h251.csv", nrows=5)

print("--- SEARCHING FOR 'UNABLE' VARIABLES ---")
for col in df.columns:
    if "UNAB" in col:  # Searches for UNABle
        print(f"Found: {col}")

print("\n--- SEARCHING FOR 'DELAY' VARIABLES ---")
for col in df.columns:
    if "LAY" in col:   # Searches for deLAY or dLAY
        print(f"Found: {col}")
# Reads tourism.csv from the repository's data folder, checks that the
# columns the rest of the pipeline expects are present, and prints a short
# summary. This keeps the dataset validation step inside the repo, so no
# external dataset store is needed.

import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

df = pd.read_csv(DATA_PATH)

missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
assert not missing, f"Dataset is missing expected columns: {missing}"

print(f"Loaded {DATA_PATH}")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"All {len(EXPECTED_COLUMNS)} expected columns are present.")
print("\nTarget distribution (ProdTaken):")
print(df["ProdTaken"].value_counts(normalize=True).rename("share").round(4))
print("\nMissing values per column (top 5):")
print(df.isna().sum().sort_values(ascending=False).head(5))

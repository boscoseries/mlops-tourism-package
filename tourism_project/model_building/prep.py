# Loads tourism.csv from the repository data folder, removes columns that
# are not predictive (the stray index column and the customer identifier),
# fixes a known typo in Gender, and splits the data into a stratified
# train/test set. The split files are saved at the repository root as plain
# CSVs (Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv) because the workflow
# uploads them from there as an artifact for the next job - there is no
# need to upload them anywhere else.

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET_COLUMN = "ProdTaken"
RANDOM_STATE = 42
TEST_SIZE = 0.2

DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

# "Fe Male" is a data-entry typo for "Female"
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print(f"Train rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")
print(f"Train target rate: {y_train.mean():.4f}, Test target rate: {y_test.mean():.4f}")
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

# MLOps Tourism Package

A machine-learning project that predicts whether a customer is likely to purchase a newly introduced wellness tourism package. It includes a dataset validation step, data preparation, model training and evaluation, MLflow tracking, a GitHub Actions pipeline, and a Streamlit prediction app.

> **Note:** This project is a prediction example. Its output is a model estimate, not a guarantee of customer behavior.

## Project overview

The model predicts `ProdTaken`, the dataset’s purchase indicator, from customer, travel, and sales-pitch information. The training code builds a preprocessing-and-classification pipeline that:

- Scales numeric features and one-hot encodes categorical features.
- Trains an XGBoost classifier.
- Tunes selected model parameters with five-fold cross-validation, using ROC-AUC for model selection.
- Evaluates the selected model on a held-out test set.
- Logs the tuned parameters and evaluation metrics to MLflow.
- Saves the trained pipeline as `tourism_project/deployment/model.joblib`.

The Streamlit app loads that saved pipeline, collects one customer’s details, and displays a purchase prediction and estimated probability.

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── pipeline.yml             # GitHub Actions model pipeline
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv              # Dataset used by the pipeline
│   ├── deployment/
│   │   ├── app.py                   # Streamlit prediction app
│   │   ├── model.joblib             # Saved trained pipeline
│   │   └── requirements.txt         # App dependencies
│   ├── model_building/
│   │   ├── data_register.py         # Dataset checks and summary
│   │   ├── prep.py                  # Cleaning and train/test split
│   │   └── train.py                 # Preprocessing, tuning, evaluation, and saving
│   └── requirements.txt             # Model-pipeline dependencies
└── README.md
```

## Dataset and features

The included `tourism_project/data/tourism.csv` is read directly by the pipeline; no external dataset registry is configured. The validation script checks that the expected columns exist and prints the dataset shape, target distribution, and columns with the most missing values.

The target is `ProdTaken`. During preparation, `CustomerID` and the optional `Unnamed: 0` index column are removed, and the known `Fe Male` entry in `Gender` is normalized to `Female`. The remaining features are split into:

- **Numeric:** `Age`, `CityTier`, `DurationOfPitch`, `NumberOfPersonVisiting`, `NumberOfFollowups`, `PreferredPropertyStar`, `NumberOfTrips`, `Passport`, `PitchSatisfactionScore`, `OwnCar`, `NumberOfChildrenVisiting`, and `MonthlyIncome`.
- **Categorical:** `TypeofContact`, `Occupation`, `Gender`, `ProductPitched`, `MaritalStatus`, and `Designation`.

The split is stratified by the target, uses a fixed random state, and reserves 20% of the data for testing. Missing feature values are imputed during training using statistics calculated from the training set.

## Requirements

- Python 3.11 is used by the GitHub Actions workflow and is recommended for local runs.
- Model-pipeline dependencies are listed in `tourism_project/requirements.txt`.
- Streamlit app dependencies are listed separately in `tourism_project/deployment/requirements.txt`.

## Run the model pipeline locally

Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the pipeline dependencies and run the stages in order:

```bash
pip install -r tourism_project/requirements.txt

python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
```

Data preparation writes the following files to the repository root:

```text
Xtrain.csv
Xtest.csv
ytrain.csv
ytest.csv
```

Training expects those split files to be present. It also connects to an MLflow tracking server at `http://127.0.0.1:5000`. Start the server in a separate terminal:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Then, from the repository root, run:

```bash
python tourism_project/model_building/train.py
```

Training prints the selected parameters, cross-validation ROC-AUC, test metrics, and a classification report. It saves the complete trained pipeline to `tourism_project/deployment/model.joblib` and logs the model artifact and metrics to MLflow.

## Run the Streamlit app

The app expects the saved model to exist at `tourism_project/deployment/model.joblib`. Install its dependencies and launch it from the repository root:

```bash
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```

Enter the customer details in the app and select **Predict** to see whether the model predicts a purchase, along with its estimated probability.

## GitHub Actions pipeline

The workflow at `.github/workflows/pipeline.yml` runs when code is pushed to `main` or when manually started from the GitHub Actions tab. Its jobs run in sequence:

1. **Register dataset:** installs pipeline dependencies, checks the included dataset, and uploads it as an artifact.
2. **Prepare data:** creates the stratified train/test split and uploads the four CSV split files as an artifact.
3. **Train model:** downloads the split files, starts an MLflow UI process, tunes and evaluates the model, then commits any updated files in `tourism_project/deployment/` to the branch.

The workflow uses Python 3.11. Its training job needs permission to write repository contents so it can commit the resulting model artifact.

## Evaluation and tracking

The training script reports and logs the following held-out test metrics:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

It also logs the best cross-validation ROC-AUC and the selected XGBoost parameters. MLflow uses the experiment name `tourism-wellness-package`.

## Limitations

- The app and model use the categories and value ranges represented in the current implementation and dataset.
- Performance depends on the provided dataset and may not generalize to other populations or business settings.
- The repository contains a training and inference workflow; it does not configure a hosted production service or continuous monitoring.
- Review model quality, data suitability, and operational requirements before using predictions in business decisions.

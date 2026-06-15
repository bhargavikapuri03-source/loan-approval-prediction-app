# loan-approval-prediction-app
# Loan Prediction App

This project is a **Loan Prediction App** that uses a trained machine learning model to predict whether a loan application is likely to be approved or not based on applicant details. The app loads a pre-trained model, along with encoders and a scaler, and exposes a simple interface for making predictions.

## Project description

The goal of this project is to build an end-to-end machine learning pipeline for loan approval prediction and wrap it in an application that can be easily used. The pipeline typically includes:

- Loading and cleaning the loan dataset.
- Encoding categorical features (e.g., gender, marital status, education, property area).
- Scaling numerical features (e.g., applicant income, loan amount).
- Training a classification model to predict loan approval status.
- Saving the trained model, encoders, and scaler to disk.
- Creating an app script that loads these artifacts and serves predictions.

This repository contains the training code, the trained artifacts (`model.pkl`, `encoders.pkl`, `scaler.pkl`), and the application code.

## Files in this repository

- `.gitignore` – Git ignore rules for this project.
- `app` – Application entry point (e.g., Streamlit/FastAPI/Flask app script or folder).
- `train` – Training script or notebook used to train the model.
- `loan` – Loan dataset file (e.g., Excel/CSV) used for training and evaluation.
- `model.pkl` – Saved trained machine learning model.
- `encoders.pkl` – Saved encoders for categorical features.
- `scaler.pkl` – Saved scaler for numerical features.
- `requirements.txt` – Python dependencies required to run the project.

## Technology stack

- **Language**: Python 3
- **Typical libraries** (update according to your `requirements.txt`):
  - numpy, pandas
  - scikit-learn
  - joblib / pickle
  - streamlit / flask / fastapi (depending on how the app is built)
  - any other libraries listed in `requirements.txt`

## Setup and installation

1. **Clone the repository**

```bash
git clone https://github.com/bhargavikapuri03-source/loan-approval-prediction-app.git
cd loan-approval-prediction-app
```

2. **Create and activate a virtual environment** (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## How to run the training script

If you want to retrain the model (optional):

```bash
python train.py
```
## How to run the Loan Prediction App

If the app is a Streamlit app , you would run:

```bash
streamlit run app.py
```

or, if the file is just called `app`:

```bash
streamlit run app
```

If it is a Flask or FastAPI app, adjust to:

```bash
python app.py
```
Once the app is running:

- Open the URL shown in the terminal (http://localhost:8501/) 
- Enter applicant details (income, loan amount, credit history, etc.) in the UI.
- Click the button to get a **Loan Approved / Not Approved** prediction.

## Data

- **File**: `loan` (e.g., `loan.csv` / `loan.xlsx`)
- **Description**:
  - Each row represents a loan application.
  - Columns contain applicant financial and demographic information along with the loan approval status.

## Model details

- **Problem type**: Binary classification (Approved vs Not Approved).
- **Features**: Typical features may include:
  - Applicant income
  - Co-applicant income
  - Loan amount and term
  - Credit history
  - Gender, marital status, education, employment status
  - Property area, etc.
- **Preprocessing**:
  - Categorical features encoded and saved in `encoders.pkl`.
  - Numerical features scaled and saved in `scaler.pkl`.
- **Model**:
  - Trained model object saved as `model.pkl` (e.g., Logistic Regression / Random Forest / XGBoost).

## Usage notes

- Make sure `model.pkl`, `encoders.pkl`, and `scaler.pkl` are in the same directory as `app` (or in the paths expected by your code).
- If you change the dataset or retrain the model, regenerate and overwrite the `.pkl` files to keep the app consistent.
- If you deploy this app (e.g., on Streamlit Cloud, Heroku, etc.), ensure the dataset and model files are handled appropriately and do not expose sensitive data.

## Acknowledgements

This Loan Prediction App was developed as part of the Internspark AI Internship to practice building end-to-end machine learning applications and deploying predictive models.

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# LOAD DATA
df = pd.read_csv("loan.csv")

# DROP LOAN ID
df = df.drop('Loan_ID', axis=1)

# HANDLE DEPENDENTS (3+ issue)
df['Dependents'] = df['Dependents'].replace('3+', '3')

# CONVERT TO NUMERIC
df['Dependents'] = df['Dependents'].astype(float)

# FILL MISSING VALUES
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].median(), inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# ENCODING
le = LabelEncoder()
encoders = {}

categorical_cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# SPLIT FEATURES & TARGET
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# MODEL
model = RandomForestClassifier()
model.fit(X_train,y_train) 

# SAVE FILES
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
pickle.dump(encoders, open("encoders.pkl","wb"))

print("All Pickle files saved successfully!")
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
encoders = pickle.load(open("encoders.pkl","rb"))

st.title("Loan Approval Prediction App")

Gender = st.selectbox("Gender",["Male","Female"])
Married = st.selectbox("Married",["Yes","No"])
Dependents = st.selectbox("Dependents",[0,1,2,3])
Education = st.selectbox("Education",["Graduate","Not Graduate"])
Self_Employed = st.selectbox("Self Employed",["Yes","No"])
ApplicantIncome = st.number_input("Applicant Income")
CoapplicantIncome = st.number_input("Coapplicant Income")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.number_input("Loan Amount Term")
Credit_History = st.selectbox("Credit History",[0,1])
Property_Area = st.selectbox("Property Area",["Urban","Semiurban","Rural"])

if st.button("Predict Loan Status"):

    Gender = encoders['Gender'].transform([Gender])[0]
    Married = encoders['Married'].transform([Married])[0]
    Education = encoders['Education'].transform([Education])[0]
    Self_Employed = encoders['Self_Employed'].transform([Self_Employed])[0]
    Property_Area = encoders['Property_Area'].transform([Property_Area])[0]

    data = np.array([[Gender,Married,Dependents,Education,Self_Employed,
                      ApplicantIncome,CoapplicantIncome,LoanAmount,
                      Loan_Amount_Term,Credit_History,Property_Area]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0]==1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
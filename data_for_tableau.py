import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

data=pd.read_csv(r"C:\Users\User\DA proj\Credit Risk Analytics & Loan Approval Optimization\datasets\df_for_ml.csv")
model = joblib.load("calibrated_model.pkl")

X=data.drop(columns='default')
y=data['default']

X_train,X_test, y_train, y_test = train_test_split(X,y,random_state=42, stratify=y, test_size=0.2)

scaler = StandardScaler()
ohe = OneHotEncoder(drop = 'first', sparse_output = False)

prescal = ColumnTransformer(
    transformers=[
        ('num', scaler, ['loan_percent_income','person_income','cb_person_default_on_file','loan_int_rate','loan_amnt'] ),
        ('str', ohe, ['loan_intent','loan_grade'])
    ])

prescal.fit(X_train)

X_full_scaled=prescal.transform(X)

data['predicted_default_prob']=model.predict_proba(X_full_scaled)[:,1]

data['loan_status_text'] = data['default'].map({0: 'Оплачен', 1: 'Дефолт'})

data['past_default_text'] = data['cb_person_default_on_file'].map({
    0: 'Нет дефолтов в БКИ', 1: 'Были дефолты в БКИ',
    'N': 'Нет дефолтов в БКИ', 'Y': 'Были дефолты в БКИ'
})

data=data.drop(columns='cb_person_default_on_file')

data.to_csv('credit_risk_data_for_tableau.csv', index = False, encoding='utf-8-sig')


print(data.describe)



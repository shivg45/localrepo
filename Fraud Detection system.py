#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

file_path = "Fraud.csv" 
data = pd.read_csv(file_path)

def reduce_memory_usage(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df

data = reduce_memory_usage(data)

print(data.info())
print("Missing values:\n", data.isnull().sum())

data.fillna(data.median(), inplace=True)

label_encoder = LabelEncoder()

categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

print("After Label Encoding:\n", data.info())

X = data.drop(columns=['isFlaggedFraud'])
y = data['isFlaggedFraud']

print("Class distribution before resampling:\n", y.value_counts())

undersampler = RandomUnderSampler(random_state=42)

X_resampled, y_resampled = undersampler.fit_resample(X, y)

print("Class distribution after resampling:\n", pd.Series(y_resampled).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15, n_jobs=-1)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()


# 
# 
# 
# # Summary:
# The code implements a machine learning pipeline for detecting fraudulent transactions in a financial dataset. 
# 
# Key Algorithms Used:-
# 1)Label Encoding: This  converts categorical variables (e.g., 'CASH_OUT') into integer labels to make them suitable for the machine learning model.
# 
# 2)Random Under-sampling:- It basically balances the class distribution by undersampling the majority class, preventing the model from being biased towards the dominant class (non-fraud).
# 
# 3)Random Forest Classifier:- An ensemble learning algorithm that builds multiple decision trees and combines them for more accurate predictions. It's robust and works well with both classification and regression tasks.
# 
# 4)StandardScaler:- It scales the features to standardize them, ensuring each feature contributes equally to the model.
# 
# 5)Evaluation Metrics:
# -> Classification Report, Confusion Matrix, and ROC-AUC are used to evaluate how well the model performs in detecting fraudulent transactions.
# 
# 
# Purpose of model:-
# The goal of the code is to develop a machine learning model that can predict fraudulent transactions effectively while handling large-scale data efficiently and dealing with class imbalance. The Random Forest model, along with data preprocessing techniques like encoding and scaling, is used to build an accurate and balanced model for fraud detection.
# 
# 

# In[ ]:





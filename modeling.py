from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os
from feature_engine.encoding import MeanEncoder
from feature_engine.encoding import WoEEncoder
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Read CSVs
real_world_df = pd.read_csv('./data/real-world-data/healthcare-dataset-stroke-data.csv')
train_synthetic_df = pd.read_csv('./data/train.csv')

# Features
categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
binary = ["hypertension", "heart_disease", "stroke"] # Basically, categorical values with only 2 values. 1 or 0
continous_numerical = ["age", "avg_glucose_level", "bmi"]

encoder = MeanEncoder(variables=categorical)
encoder = WoEEncoder(variables=categorical)
encoder = CountFrequencyEncoder(variables=categorical, encoding_method='frequency')

### Using a Decision Tree to predict the missing BMI
DT_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=42))
                              ])
X = dt_df[['age','gender','bmi']].copy()
X.gender = X.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)

missing = X[X.bmi.isna()]
X = X[~X.bmi.isna()]
Y = X.pop('bmi')
DT_bmi_pipe.fit(X,Y)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(missing[['age','gender']]),index=missing.index)
dt_df.loc[missing.index,'bmi'] = predicted_bmi

print('Missing values after decision tree regressor: ',sum(dt_df.isnull().sum()))

### Using a Simple Imputer
mean_imputer = SimpleImputer(strategy='mean')
imputed_df['bmi'] = mean_imputer.fit_transform(imputed_df['bmi'].values.reshape(-1,1))
print('Missing values after imputing: ',sum(imputed_df.isnull().sum()))

# Baseline
cv = StratifiedKFold(shuffle=True, random_state=2023)
features = categorical + binary + continous_numerical

# Create a copy of train df to modify inplace
train_df = train_synthetic_df[features].copy()

# cast categorical features as categorical type
train_df[categorical] = (
    train_df[categorical].astype('category')
)

target = train_df['stroke']

oof_preds = pd.Series(0, index=train_df.index)
train_auc = []
val_auc = []
pipelines = []

for fold, (tr_ix, vl_ix) in enumerate(cv.split(train_df, target)):
    X_train, Y_train = train_df.iloc[tr_ix], target.iloc[tr_ix]
    X_val, Y_val = train_df.iloc[vl_ix], target.iloc[vl_ix]
    
    X_train = X_train.copy()
    X_val = X_val.copy()
    
    model = LogisticRegression()
    encoder = OneHotEncoder(drop_last=True, variables=categorical)
    scaler = SklearnTransformerWrapper(StandardScaler(), variables=binary+continous_numerical)
    
    X_train = encoder.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    
    X_val = encoder.transform(X_val)
    X_val = scaler.transform(X_val)

    print('_'*50)
    
    model.fit(X_train, Y_train)
    
    oof_preds.iloc[vl_ix] = model.predict_proba(X_val)[:, 1]
    train_auc.append(roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]))
    val_auc.append(roc_auc_score(Y_val, model.predict_proba(X_val)[:, 1]))
    pipelines.append([encoder, scaler, model])

    print(f'Val AUC: {val_auc[-1]}')

print(f'Mean Val AUC: {np.mean(val_auc)}')
print(f'OOF AUC: {roc_auc_score(target, oof_preds)}')
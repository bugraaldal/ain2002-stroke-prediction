from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import pandas as pd
from feature_engine.encoding import StringSimilarityEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

s_categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
binary = ["hypertension", "heart_disease"] # Basically, categorical values with only 2 values. 1 or 0
continous_numerical = ["age", "avg_glucose_level", "bmi"]

def generate_features(df):
    df['age/bmi'] = df.age / df.bmi
    df['age*bmi'] = df.age * df.bmi
    df['bmi/prime'] = df.bmi / 25
    df['obesity'] = df.avg_glucose_level * df.bmi / 1000
    df['blood_heart']= df.hypertension*df.heart_disease
    return df

categorical = s_categorical + binary
features = categorical + continous_numerical + ['age/bmi', 'age*bmi', 'bmi/prime', 'obesity', 'blood_heart']
rfe_features =[
    'age', 'avg_glucose_level', 'bmi', 'ever_married_Yes', 'gender_Male',
    'hypertension_0', 'smoking_status_Unknown', 'smoking_status_formerly smoked',
    'smoking_status_never smoked', 'work_type_Self-employed', 'work_type_children'
]

real_world_df = pd.read_csv('./data/real-world-data/healthcare-dataset-stroke-data.csv',)

train_df = pd.read_csv('./data/synthetic-data/train.csv', index_col=0)
test_df = pd.read_csv('./data/synthetic-data/test.csv', index_col=0)


@ignore_warnings(category=ConvergenceWarning)
def XGBoost_model(synthetic_train_df, real_world_df, train_df, test_df_c, real_world_target, encoder):
    # cast categorical features as categorical type
    synthetic_train_df[categorical] = (synthetic_train_df[categorical].astype('category'))
    real_world_df[categorical] = (real_world_df[categorical].astype('category'))


    target = train_df['stroke']
    # Grab Target
    oof_preds = pd.Series(0, index=train_df.index, name='stroke')
    test_preds = pd.Series(0, index=test_df.index, name='stroke')

    train_auc, val_auc = [], []
    pipelines = []

    for fold, (train_indx, val_indx) in enumerate(cv.split(train_df, target)):
        pipeline = []
        X_train, y_train = synthetic_train_df.iloc[train_indx], target.iloc[train_indx]
        X_val, y_val = synthetic_train_df.iloc[val_indx], target.iloc[val_indx]
        X_test = test_df_c.copy()

        # concat prev dataset
        X_train = pd.concat([X_train, real_world_df], axis=0)
        y_train = pd.concat([y_train, real_world_target])

        X_train = X_train.copy()
        X_val = X_val.copy()


        model = XGBClassifier(
            random_state=42, learning_rate=1e-2,
            n_estimators=3000,
            tree_method='gpu_hist',
            callbacks=[
                EarlyStopping(100, save_best=True, maximize=False),
            ]
        )

        # Fit encoder
        X_train = encoder.fit_transform(X_train)
        X_val = encoder.transform(X_val)
        X_test = encoder.transform(X_test)
        pipeline.append(encoder)

        # filter columns by RFE columns
        X_train = X_train[X_train.columns.intersection(rfe_features)]
        X_val = X_val[X_val.columns.intersection(rfe_features)]
        X_test = X_test[X_test.columns.intersection(rfe_features)]

        print('_'*50)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val,y_val)],
            verbose=1000,
        )

        oof_preds.iloc[val_indx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1]

        train_auc.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
        val_auc.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

        pipeline.append(model)
        pipelines.append(pipeline)

        print(f'Val AUC: {val_auc[-1]}')

    print()
    print(f'Mean Val AUC: {np.mean(val_auc)}')
    print(f'OOF AUC: {roc_auc_score(target, oof_preds)}')
    return test_preds, pipelines

def impute_mean(real_world_df):
    mean_imputer = SimpleImputer(strategy="mean")
    real_world_df = real_world_df.copy()
    real_world_df["bmi"] = mean_imputer.fit_transform(real_world_df["bmi"].values.reshape(-1,1))
    print("Missing values after imputing with mean: ",sum(real_world_df.isnull().sum()))
    return real_world_df

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Generate feature engineering
synthetic_train_df = generate_features(train_df.copy())
test_df_c = generate_features(test_df.copy())
real_world_df_mean = impute_mean(real_world_df)
real_world_df_mean = generate_features(real_world_df_mean.copy())


real_world_target = real_world_df["stroke"]

# filter features 
synthetic_train_df = synthetic_train_df[features]
test_df_c = test_df_c[features]
real_world_df_mean = real_world_df_mean[features]

preds_ss_mean, pipelines = XGBoost_model(synthetic_train_df, real_world_df_mean, train_df, test_df_c, real_world_target, encoder=StringSimilarityEncoder(variables=categorical))
preds_ss_mean /= len(pipelines)

preds_ss_mean.rename('stroke', inplace=True)
preds_ss_mean.to_csv('final_submission.csv')  # Public Score: 0.8672
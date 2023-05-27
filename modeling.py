### Using a Decision Tree to predict the missing BMI
dt_df = df.copy()
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
imputed_df = df.copy()
mean_imputer = SimpleImputer(strategy='mean')
imputed_df['bmi'] = mean_imputer.fit_transform(imputed_df['bmi'].values.reshape(-1,1))
print('Missing values after imputing: ',sum(imputed_df.isnull().sum()))

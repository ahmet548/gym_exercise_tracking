import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('gym_members_exercise_tracking.csv')

ohe = OneHotEncoder()
xd = ohe.fit_transform(df[['Workout_Type', 'Gender']]).toarray()

columns_name = ohe.get_feature_names_out()
df[columns_name] = xd

df = df.drop(columns=['Gender', 'Workout_Type'])

y = df['Calories_Burned']
x = df.drop('Calories_Burned',axis=1)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=6,train_size=0.77
)

xgb = XGBRegressor()
xgbModel = xgb.fit(x_train,y_train)

example = x.head(1)

print(f"XGBRF Regressor: {xgbModel.score(x_test,y_test)} --> {xgbModel.predict(example)}")

print(df.info())

joblib.dump(xgbModel, 'xgb_model.pkl')
joblib.dump(ohe, 'ohe_encoder.pkl')
joblib.dump(x.columns.tolist(), 'feature_columns.pkl')

print("Model ve bileşenler başarıyla kaydedildi.")
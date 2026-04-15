import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data.csv")

le_hospital = LabelEncoder()
le_test = LabelEncoder()
le_unit = LabelEncoder()

df['hospital'] = le_hospital.fit_transform(df['hospital'])
df['test_name'] = le_test.fit_transform(df['test_name'])
df['unit'] = le_unit.fit_transform(df['unit'])

X = df[['hospital', 'test_name', 'unit']]
y = df['value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))

with open("model.pkl", "wb") as f:
    pickle.dump((model, le_hospital, le_test, le_unit), f)

print("Model trained and saved!")

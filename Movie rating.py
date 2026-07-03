import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

path = kagglehub.dataset_download("adrianmcmahon/imdb-india-movies")

csv_file = os.path.join(path, os.listdir(path)[0])
df = pd.read_csv(csv_file, encoding="latin1")

df = df.dropna()

le = LabelEncoder()

# adjust columns safely (dataset columns may differ)
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Rating", axis=1)
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
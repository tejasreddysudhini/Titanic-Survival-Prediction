import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

path = kagglehub.dataset_download("yasserh/titanic-dataset")

csv_file = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(csv_file)

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Cabin"] = df["Cabin"].fillna("Unknown")

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])
df["Cabin"] = le.fit_transform(df["Cabin"])

df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100)

import matplotlib.pyplot as plt

survival = df["Survived"].value_counts()

plt.bar(["Not Survived", "Survived"], survival)
plt.title("Titanic Survival Count")
plt.xlabel("Status")
plt.ylabel("Passengers")
plt.show()
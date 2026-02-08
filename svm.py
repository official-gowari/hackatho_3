import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect("placement.db")
df = pd.read_sql_query("SELECT * FROM students", conn)
conn.close()

print(df.head())
print("Rows:", len(df))

print("Missing:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print("Ranges:\n", df.describe())

print("Placed vs Not Placed:\n", df['placed'].value_counts())

print(df.corr()['placed'].sort_values(ascending=False))


import matplotlib.pyplot as plt
df.boxplot(column='cgpa', by='placed')
plt.title("CGPA vs Placement")
plt.show()

df.boxplot(column='internships', by='placed')

plt.title("Internships vs Placement")
plt.suptitle("")  
plt.xlabel("Placed (0 = No, 1 = Yes)")
plt.ylabel("Number of Internships")

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

conn = sqlite3.connect("placement.db")
df = pd.read_sql_query("SELECT * FROM students", conn)
conn.close()

X = df[['cgpa', 'internships', 'projects', 'communication']]
y = df['placed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear"))
])

svm_pipeline.fit(X_train, y_train)

svm_pred = svm_pipeline.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_acc * 100)

import pickle

with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_pipeline, f)

print("SVM model saved!")
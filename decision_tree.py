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
plt.suptitle("")   # Extra title remove
plt.xlabel("Placed (0 = No, 1 = Yes)")
plt.ylabel("Number of Internships")

plt.show()

conn = sqlite3.connect("placement.db")
df = pd.read_sql_query("SELECT * FROM students", conn)
conn.close()

X = df[['cgpa', 'internships', 'projects', 'communication']]
y = df['placed']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_acc * 100)

import pickle

with open("dt_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

print("Decision Tree model saved!")

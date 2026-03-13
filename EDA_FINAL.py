import pandas as pd
import sqlite3

# connect database
conn = sqlite3.connect("placement_final.db")

# load data
df = pd.read_sql_query("SELECT * FROM students", conn)

conn.close()

print("Dataset Loaded Successfully")

print("\nFirst 5 rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nTotal Rows:", len(df))
print("Total Columns:", len(df.columns))

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values Check:")
print(df.isnull().sum())

print("\nDuplicate Rows Check:")
print(df.duplicated().sum())

print("\nPlacement Distribution:")

import matplotlib.pyplot as plt

df['placement'].value_counts().plot(kind='bar')

plt.title("Placement Distribution")
plt.xlabel("Placement (0 = Not Placed, 1 = Placed)")
plt.ylabel("Number of Students")

plt.show()

print("\nCGPA vs Placement")

import seaborn as sns

plt.figure(figsize=(8,5))

sns.boxplot(x='placement', y='cgpa', data=df)

plt.title("CGPA vs Placement")
plt.xlabel("Placement (0 = Not Placed, 1 = Placed)")
plt.ylabel("CGPA")

plt.show()

print("\nBacklogs vs Placement")

plt.figure(figsize=(8,5))

sns.boxplot(x='placement', y='backlogs', data=df)

plt.title("Backlogs vs Placement")
plt.xlabel("Placement (0 = Not Placed, 1 = Placed)")
plt.ylabel("Number of Backlogs")

plt.show()

print("\nInternships vs Placement")

plt.figure(figsize=(8,5))

sns.boxplot(x='placement', y='internship_relevance', data=df)

plt.title("Internships vs Placement")
plt.xlabel("Placement (0 = Not Placed, 1 = Placed)")
plt.ylabel("Internship Relevance Score")

plt.show()

print("\nProjects vs Placement")

plt.figure(figsize=(8,5))

sns.boxplot(x='placement', y='projects_count', data=df)

plt.title("Projects Count vs Placement")
plt.xlabel("Placement (0 = Not Placed, 1 = Placed)")
plt.ylabel("Number of Projects")

plt.show()

print("\nCommunication Skills vs Placement")

plt.figure(figsize=(8,5))

sns.boxplot(x='placement', y='communication', data=df)

plt.title("Communication Skills vs Placement")
plt.xlabel("Placement (0 = Not Placed, 1 = Placed)")
plt.ylabel("Communication Skill Score")

plt.show()

print("\nProblem Solving vs Placement")

plt.figure(figsize=(8,5))

sns.boxplot(x='placement', y='problem_solving', data=df)

plt.title("Problem Solving vs Placement")
plt.xlabel("Placement (0 = Not Placed, 1 = Placed)")
plt.ylabel("Problem Solving Score")

plt.show()

print("\nFeature Correlation Heatmap")

plt.figure(figsize=(10,6))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.show()

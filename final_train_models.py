import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_curve,
    roc_auc_score
)

import pickle


# -----------------------------
# LOAD DATA FROM DATABASE
# -----------------------------

conn = sqlite3.connect("placement_final.db")

df = pd.read_sql_query("SELECT * FROM students", conn)

conn.close()

print("Dataset Loaded")
print("Total rows:", len(df))


# -----------------------------
# FEATURE SELECTION
# -----------------------------

X = df[
    [
        "cgpa",
        "backlogs",
        "internship_relevance",
        "projects_count",
        "project_depth",
        "communication",
        "problem_solving"
    ]
]

y = df["placement"]


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training size:", len(X_train))
print("Testing size:", len(X_test))


# -----------------------------
# MODEL EVALUATION FUNCTION
# -----------------------------

def evaluate_model(name, model):

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)

    print(f"\n{name} Results")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    # ROC + AUC
    prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, prob)

    auc = roc_auc_score(y_test, prob)

    print("AUC Score:", auc)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")


# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

evaluate_model("Logistic Regression", log_model)


# -----------------------------
# DECISION TREE
# -----------------------------

dt_model = DecisionTreeClassifier(max_depth=5)

dt_model.fit(X_train, y_train)

evaluate_model("Decision Tree", dt_model)


# -----------------------------
# RANDOM FOREST
# -----------------------------

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)

evaluate_model("Random Forest", rf_model)


# -----------------------------
# ROC CURVE GRAPH
# -----------------------------

plt.plot([0,1], [0,1], linestyle="--")

plt.title("ROC Curve Comparison")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.show()


# -----------------------------
# SAVE MODELS
# -----------------------------

with open("final_logistic_model.pkl", "wb") as f:
    pickle.dump(log_model, f)

with open("final_decision_tree_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

with open("final_random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("\nFinal models saved successfully")

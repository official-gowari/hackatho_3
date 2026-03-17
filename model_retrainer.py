import sqlite3
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def retrain_model():

    # connect DB
    conn = sqlite3.connect("placement_final.db")

    df = pd.read_sql_query("SELECT * FROM students", conn)

    conn.close()

    # features & target
    X = df.drop(["placement","id","generated_at","readiness_score"], axis=1)
    y = df["placement"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    # save model (overwrite)
    pickle.dump(model, open("placement_model.pkl","wb"))

    return "Model retrained successfully"

if __name__ == "__main__":
    result = retrain_model()
    print(result)
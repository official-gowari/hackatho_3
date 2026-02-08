import streamlit as st
import sqlite3
import pandas as pd
import pickle



with open("placement_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)




conn = sqlite3.connect("placement.db")
df = pd.read_sql_query("SELECT * FROM students", conn)
conn.close()



st.set_page_config(
    page_title="Placement Prediction System",
    layout="centered"
)

st.title("🎓 Student Placement Prediction Dashboard")




st.sidebar.header("Enter Student Details")

cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)

internships = st.sidebar.slider("Internships", 0, 4, 1)

projects = st.sidebar.slider("Projects", 0, 6, 2)

communication = st.sidebar.slider("Communication Skill", 1, 10, 6)




st.sidebar.header("Select Model")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "SVM"
    ]
)



st.sidebar.markdown("### 📊 Model Accuracy")

st.sidebar.write("Logistic Regression: ~83%")
st.sidebar.write("Decision Tree: ~100%")
st.sidebar.write("SVM: ~82%")




if st.sidebar.button("Predict"):

    data = [[cgpa, internships, projects, communication]]

    result = None


    # Model Selection
    if model_choice == "Logistic Regression":
        result = lr_model.predict(data)

    elif model_choice == "Decision Tree":
        result = dt_model.predict(data)

    elif model_choice == "SVM":
        result = svm_model.predict(data)


    # Result Display
    st.subheader("📌 Prediction Result")

    if result is not None and result[0] == 1:
        st.success("✅ Student is Likely to be Placed")

    elif result is not None and result[0] == 0:
        st.error("❌ Student is Not Likely to be Placed")

    else:
        st.warning("⚠️ Prediction Failed. Try Again.")




st.subheader("📊 Dataset Overview")

total = len(df)
placed = df["placed"].sum()
not_placed = total - placed


col1, col2, col3 = st.columns(3)

col1.metric("Total Students", total)
col2.metric("Placed", placed)
col3.metric("Not Placed", not_placed)




st.subheader("📈 Placement Distribution")

st.bar_chart(df["placed"].value_counts())



st.subheader("📉 Feature Correlation with Placement")

corr = df.corr()["placed"].sort_values(ascending=False)

st.write(corr)




with st.expander("📁 View Raw Dataset"):
    st.dataframe(df)

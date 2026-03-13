import plotly.graph_objects as go
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Placement Dashboard", layout="wide")

st.title("🎓 Student Placement Prediction Dashboard")

# -------------------------
# LOAD MODELS
# -------------------------

log_model = pickle.load(open("final_logistic_model.pkl","rb"))
dt_model = pickle.load(open("final_decision_tree_model.pkl","rb"))
rf_model = pickle.load(open("final_random_forest_model.pkl","rb"))

# -------------------------
# LAYOUT
# -------------------------

col1, col2 = st.columns([2,1])

# -------------------------
# INPUT PANEL
# -------------------------

with col1:

    st.subheader("Student Attributes")

    cgpa = st.number_input("CGPA",0.0,10.0,7.0)
    backlogs = st.number_input("Backlogs",0,10,0)
    internship = st.slider("Internship Relevance",1,10,5)
    projects = st.number_input("Projects Count",0,10,2)
    depth = st.slider("Project Depth",1,10,5)
    communication = st.slider("Communication Skill",1,10,5)
    problem = st.slider("Problem Solving Skill",1,10,5)

    model_option = st.selectbox(
        "Select Prediction Model",
        ("Random Forest","Logistic Regression","Decision Tree")
    )

    predict_button = st.button("Predict Placement")

# -------------------------
# MODEL SELECTION
# -------------------------

if model_option == "Random Forest":
    model = rf_model
elif model_option == "Logistic Regression":
    model = log_model
else:
    model = dt_model

# -------------------------
# PREDICTION
# -------------------------

with col2:

    st.subheader("Prediction Result")

    if predict_button:

        data = pd.DataFrame([[cgpa, backlogs, internship, projects, depth, communication, problem]],
        columns=[
        "cgpa",
        "backlogs",
        "internship_relevance",
        "projects_count",
        "project_depth",
        "communication",
        "problem_solving"
        ])

        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        if prediction == 1:
            st.success("✅ Likely to be Placed")
        else:
            st.error("❌ Not Likely to be Placed")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Placement Probability"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0,40], 'color': "red"},
                    {'range': [40,70], 'color': "orange"},
                    {'range': [70,100], 'color': "green"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)


st.markdown("---")
st.subheader("🚀 Placement Chance Simulator")

sim_cgpa = st.slider("Simulate CGPA Improvement", 0.0, 10.0, cgpa)
sim_projects = st.slider("Simulate Projects Count", 0, 10, projects)
sim_problem = st.slider("Simulate Problem Solving Skill", 1, 10, problem)

simulate = st.button("Run Simulation")

if simulate:

    sim_data = pd.DataFrame([[
        sim_cgpa,
        backlogs,
        internship,
        sim_projects,
        depth,
        communication,
        sim_problem
    ]], columns=[
        "cgpa",
        "backlogs",
        "internship_relevance",
        "projects_count",
        "project_depth",
        "communication",
        "problem_solving"
    ])

    sim_prob = model.predict_proba(sim_data)[0][1]

    st.write("### New Placement Probability")

    gauge2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sim_prob*100,
        title={'text': "Simulated Probability"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0,40], 'color': "red"},
                {'range': [40,70], 'color': "orange"},
                {'range': [70,100], 'color': "green"}
            ]
        }
    ))

    st.plotly_chart(gauge2, use_container_width=True)

    # current probability
    current_data = pd.DataFrame([[cgpa, backlogs, internship, projects, depth, communication, problem]],
    columns=[
        "cgpa",
        "backlogs",
        "internship_relevance",
        "projects_count",
        "project_depth",
        "communication",
        "problem_solving"
    ])

    current_prob = model.predict_proba(current_data)[0][1]

    improvement = (sim_prob - current_prob) * 100

    st.write("### Improvement:", round(improvement,2), "%")

# -------------------------
# MODEL PERFORMANCE
# -------------------------

st.markdown("---")
st.subheader("📊 Model Accuracy Comparison")

models = ["Logistic Regression","Decision Tree","Random Forest"]
accuracy = [0.74,0.66,0.71]

fig, ax = plt.subplots()
ax.bar(models,accuracy)

ax.set_ylabel("Accuracy")
ax.set_title("Model Performance")

st.pyplot(fig)


# -------------------------
# FEATURE IMPORTANCE
# -------------------------

st.markdown("---")
st.subheader("🧠 Feature Importance ")

features=[
"CGPA",
"Backlogs",
"Internship",
"Projects",
"Project Depth",
"Communication",
"Problem Solving"
]

if model_option == "Random Forest":
    importance = rf_model.feature_importances_

elif model_option == "Decision Tree":
    importance = dt_model.feature_importances_

else:
    importance = abs(log_model.coef_[0])

fig2, ax2 = plt.subplots()

ax2.barh(features,importance)

ax2.set_title("Feature Importance")

st.pyplot(fig2)

import streamlit as st
import sqlite3
import pandas as pd

st.markdown("---")
st.subheader("📥 Download Generated Dataset")

conn = sqlite3.connect("placement_final.db")

df = pd.read_sql_query("SELECT * FROM students", conn)

conn.close()

st.download_button(
    label="Download Dataset as CSV",
    data=df.to_csv(index=False),
    file_name="placement_dataset.csv",
    mime="text/csv"
)

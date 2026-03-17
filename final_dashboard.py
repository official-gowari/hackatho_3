import plotly.graph_objects as go
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

from retrain_model import retrain_model  # ✅ NEW

st.set_page_config(page_title="Placement Dashboard", layout="wide")

st.title("🎓 Student Placement Prediction Dashboard")

# -------------------------
# MODEL CONTROL (NEW 🔥)
# -------------------------

st.markdown("---")
st.subheader("⚙️ Model Controls")

if st.button("🔄 Retrain Model"):
    with st.spinner("Retraining model..."):
        result = retrain_model()
    st.success(result)

# -------------------------
# LOAD MODEL
# -------------------------

def load_model():
    return pickle.load(open("placement_model.pkl","rb"))

model = load_model()

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

    predict_button = st.button("Predict Placement")

# -------------------------
# PREDICTION
# -------------------------

with col2:

    st.subheader("Prediction Result")

    if predict_button:

        model = load_model()  # always latest

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

# -------------------------
# SIMULATOR
# -------------------------

st.markdown("---")
st.subheader("🚀 Placement Chance Simulator")

sim_cgpa = st.slider("Simulate CGPA Improvement", 0.0, 10.0, cgpa)
sim_projects = st.slider("Simulate Projects Count", 0, 10, projects)
sim_problem = st.slider("Simulate Problem Solving Skill", 1, 10, problem)

simulate = st.button("Run Simulation")

if simulate:

    model = load_model()

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
# FEATURE IMPORTANCE
# -------------------------

st.markdown("---")
st.subheader("🧠 Feature Importance")

features=[
"CGPA",
"Backlogs",
"Internship",
"Projects",
"Project Depth",
"Communication",
"Problem Solving"
]

importance = abs(model.coef_[0])

fig2, ax2 = plt.subplots()
ax2.barh(features,importance)
ax2.set_title("Feature Importance")

st.pyplot(fig2)

# -------------------------
# DOWNLOAD DATASET
# -------------------------

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
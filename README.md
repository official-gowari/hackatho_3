
<img width="1142" height="958" alt="Screenshot 2026-02-08 230926" src="https://github.com/user-attachments/assets/8466b8c6-4d29-4db2-b38d-04bb3ce2ff71" />

## Project Overview
This project is a machine learning-based system designed to predict whether a student is likely to be placed based on academic and skill-related parameters.
The system uses multiple classification models and provides real-time predictions through an interactive dashboard.

The project follows an end-to-end pipeline including data generation, preprocessing, model training, evaluation, and deployment.
<img width="1169" height="1293" alt="Screenshot 2026-02-08 230802" src="https://github.com/user-attachments/assets/e61c2d5f-d401-408a-acd7-d458887be916" />

---

## Objectives
- To analyze student performance data
- To predict placement status using machine learning
- To compare multiple classification models
- To provide an interactive dashboard for prediction
- To implement an ETL-based data pipeline

---

## Dataset Description
The dataset is generated synthetically using Python and stored in an SQLite database.

## Features Used:
- CGPA
- Number of Internships
- Number of Projects
- Communication Skills
- Placement Status (Target Variable)

The data is generated continuously and structured to simulate real-world student performance patterns.
<img width="763" height="452" alt="Screenshot 2026-02-08 231607" src="https://github.com/user-attachments/assets/84f458d4-066e-4d4b-b407-895f356db513" />

---

## Project Workflow (ETL Pipeline)
<img width="685" height="459" alt="Screenshot 2026-02-08 233737" src="https://github.com/user-attachments/assets/5dc930c1-b194-4d73-b216-9990e31513f0" />

## Extract
- Synthetic data is generated using Python
- Data is stored in SQLite database (`placement.db`)
- Data is loaded into Pandas for analysis

## Transform
- Data cleaning and validation
- Feature selection and engineering
- Exploratory Data Analysis (EDA)
- Train-test split

## Load
- Processed data stored in database
- Trained models saved using Pickle
- Models deployed in dashboard

---

## Machine Learning Models Used
The following classification models are implemented:

- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM with Pipeline)

Each model is evaluated using accuracy score and confusion matrix.

---

## Model Evaluation
Models are evaluated using:
- Train-Test Split
- Accuracy Score
- <img width="275" height="203" alt="Screenshot 2026-02-08 234026" src="https://github.com/user-attachments/assets/4aa68b30-5dfc-43e2-886e-73eddda172c4" />

- Confusion Matrix

Performance comparison helps in selecting the best model.

---

## Technologies Used
- Programming Language: Python
- Database: SQLite
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib
- Dashboard: Streamlit
- Model Serialization: Pickle
- Version Control: GitHub

---

## Dashboard Features
- Interactive student input panel
- <img width="325" height="866" alt="Screenshot 2026-02-08 230851" src="https://github.com/user-attachments/assets/8275c945-a56f-4c5d-a988-c52597017f4d" />

- Multiple model selection
- <img width="1140" height="881" alt="Screenshot 2026-02-08 234432" src="https://github.com/user-attachments/assets/65c13d39-31f7-460c-b49c-83328b8df946" />

- Real-time placement prediction
- <img width="1164" height="960" alt="Screenshot 2026-02-08 231001" src="https://github.com/user-attachments/assets/5a75c03e-5491-42b5-8885-d71cd721b6a1" />

- Dataset overview and statistics
- <img width="1146" height="1506" alt="Screenshot 2026-02-08 234850" src="https://github.com/user-attachments/assets/1257fe42-b565-49cb-8358-d7f5fc68c802" />

- Visualization of placement distribution
- <img width="1163" height="951" alt="Screenshot 2026-02-08 230941" src="https://github.com/user-attachments/assets/8b608b1c-0664-41e7-a99e-0a4d8d1065e0" />

- Correlation analysis
<img width="888" height="676" alt="Screenshot 2026-02-08 235427" src="https://github.com/user-attachments/assets/50c00f82-ce1f-462e-b1ab-d4c85ce2c7c6" />

---

## How to Run the Project

1. Install Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```
<img width="1114" height="85" alt="Screenshot 2026-02-08 231241" src="https://github.com/user-attachments/assets/ca20cdd5-9a59-4710-91a8-5b650e03d391" />

----------------------------------------------------------------------------------------

2. Generate the Dataset
Run the data generator script to create and store student data in the SQLite database.
```bash
python data_generator.py
```
This will generate synthetic student records and store them in `placement.db.`

----------------------------------------------------------------------------------------

3. Train the Machine Learning Models
Run the training script to train all models and save them using Pickle.
```bash
python train_models.py
```
(Replace `train_models.py` with your actual training file name if different.)
>This script will:

>Load data from the database

>Perform data preprocessing

>Train Logistic Regression, Decision Tree, and SVM models

>Evaluate model performance

>Save trained models as .pkl files

4. Run the Dashboard
Start the Streamlit dashboard for real-time predictions.
```bash
python -m streamlit run dashboard.py
```
<img width="1124" height="164" alt="Screenshot 2026-02-08 231454" src="https://github.com/user-attachments/assets/262b4210-a26e-4fb3-b012-2b7c4325ca34" />

The dashboard will open in your browser 

5. Make Predictions

>Enter student details using the sliders
<img width="325" height="866" alt="Screenshot 2026-02-08 230851" src="https://github.com/user-attachments/assets/1d6f3545-9429-498a-88ad-3327e82a9c78" />

>Select the desired machine learning model
<img width="1140" height="881" alt="Screenshot 2026-02-08 234432" src="https://github.com/user-attachments/assets/13fe43e3-7f42-4c05-8631-1a1e2b9dd731" />

>Click on the Predict button
<img width="1164" height="960" alt="Screenshot 2026-02-08 231001" src="https://github.com/user-attachments/assets/4f35d6cf-6048-4cf9-82bf-0ca09998a4be" />

>View the placement prediction result

## Future Improvements

- Integration with real-time student data from the college database
- Full automation of the ETL pipeline using schedulers
- Implementation of advanced machine learning models
- Personalized student dashboards for performance tracking
- Resume analysis and interview performance evaluation
- Recommendation system for skill development and training
- Mobile application development
- Automatic model retraining with new data


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split  
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Define features (X) and target (y)
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Identify categorical and numerical columns
catcols = [cname for cname in X.columns if X[cname].dtype == "object"]
numcols = [cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]]

# Define transformers for preprocessing
numerical_transformer = SimpleImputer(strategy="mean")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Create a preprocessor for columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numcols),
        ("cat", categorical_transformer, catcols)
    ]
)

# Define the model
model = RandomForestClassifier(random_state=42)

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Train the pipeline using the entire dataset
pipeline.fit(X, y)

# Streamlit Frontend
st.title("Prediksi Terkena Stroke")

st.write("""
Aplikasi ini melakukan prediksi apakah seseorang akan mengalami stroke berdasarkan data yang telah disediakan.
Masukkan data yang diminta dan tekan 'Predict Stroke' untuk mendapatkan hasilnya.
""")

# Create input fields using Streamlit
gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
age = st.number_input("Usia", min_value=0, max_value=100, step=1)
hypertension = st.selectbox("Riwayat Hipertensi (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Riwayat Penyakit Jantung (0 = No, 1 = Yes)", [0, 1])
ever_married = st.selectbox("Pernah Menikah ? (Yes/No)", ["Yes", "No"])
work_type = st.selectbox("Tipe Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Tempat Tinggal", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Level Glukosa Rata - Rata", min_value=0.0, step=0.1)
bmi = st.number_input("Indeks Massa Tubuh", min_value=0.0, step=0.1)
smoking_status = st.selectbox("Apakah Merokok ?", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Create a button for prediction
if st.button("Predict Stroke"):
    # Collect input values into a DataFrame
    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    input_df = pd.DataFrame([input_data])

    # Predict the target (stroke) using the trained model
    prediction = pipeline.predict(input_df)[0]

    # Show the result
    if prediction == 1:
        st.error("Prediksi menghasilkan: **Terkena Stroke**")
    else:
        st.success("Prediksi menghasilkan: **Tidak terkena Stroke**")

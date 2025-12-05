import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Performance App", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4B0082;'>🎓 Student Performance Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("StudentsPerformance.csv")
    data['average_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3
    return data

data = load_data()

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Data Overview", "Predict Score"])

# -------------------- PAGE 1 - DATA -----------------------
if page == "Data Overview":
    st.header("📄 Dataset Preview")
    st.dataframe(data.head())

# -------------------- PAGE 2 - PREDICTION -----------------------
elif page == "Predict Score":
    st.header("🎯 Predict Student Final Score")

    # Encode categorical data
    data_encoded = pd.get_dummies(data, drop_first=True)
    X = data_encoded.drop("average_score", axis=1)
    y = data_encoded["average_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Input section
    st.subheader("Enter Student Details")
    gender = st.selectbox("Gender", ["female", "male"])
    race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parent_edu = st.selectbox(
        "Parental Education",
        ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    )
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    prep = st.selectbox("Test Preparation Course", ["none", "completed"])

    math = st.number_input("Math Score", 0, 100, 50)
    reading = st.number_input("Reading Score", 0, 100, 50)
    writing = st.number_input("Writing Score", 0, 100, 50)

    # Create input dataframe
    input_data = pd.DataFrame({
        "math score": [math],
        "reading score": [reading],
        "writing score": [writing],
        "gender_male": [1 if gender == "male" else 0],
        "race/ethnicity_group B": [1 if race == "group B" else 0],
        "race/ethnicity_group C": [1 if race == "group C" else 0],
        "race/ethnicity_group D": [1 if race == "group D" else 0],
        "race/ethnicity_group E": [1 if race == "group E" else 0],
        "parental level of education_high school": [1 if parent_edu == "high school" else 0],
        "parental level of education_some college": [1 if parent_edu == "some college" else 0],
        "parental level of education_associate's degree": [1 if parent_edu == "associate's degree" else 0],
        "parental level of education_bachelor's degree": [1 if parent_edu == "bachelor's degree" else 0],
        "parental level of education_master's degree": [1 if parent_edu == "master's degree" else 0],
        "lunch_standard": [1 if lunch == "standard" else 0],
        "test preparation course_completed": [1 if prep == "completed" else 0],
    })
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"📘 Predicted Final Score: {prediction:.2f} / 100")

        # Plot actual vs predicted scores
        st.subheader("📊 Actual vs Predicted Scores")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred, color="#4B0082", alpha=0.7)
        ax.plot([0, 100], [0, 100], '--', color='gray')  # diagonal line
        ax.set_xlabel("Actual Average Score")
        ax.set_ylabel("Predicted Average Score")
        ax.set_title("Actual vs Predicted Scores")
        st.pyplot(fig)

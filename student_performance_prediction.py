# 🎓 Student Performance Prediction Project
# -----------------------------------------
# Objective: Predict a student’s final exam score using machine learning

# 📦 Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 📥 Step 2: Load the dataset
# Download from: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
data = pd.read_csv("StudentsPerformance.csv")

# 🧾 Step 3: Display basic info
print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())
print("\nColumn names:\n", data.columns)

# 🧹 Step 4: Data preprocessing
# Create a new column for average score
data['average_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3

# Convert categorical columns into numeric (dummy encoding)
data_encoded = pd.get_dummies(data, drop_first=True)

# 🔍 Step 5: Define features and target
X = data_encoded.drop(['average_score'], axis=1)
y = data_encoded['average_score']

# 🔀 Step 6: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🤖 Step 7: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 📈 Step 8: Make predictions
y_pred = model.predict(X_test)

# 📊 Step 9: Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Evaluation:")
print("R² Score (Accuracy):", round(r2, 3))
print("Mean Squared Error:", round(mse, 3))

# 🖼️ Step 10: Visualize Actual vs Predicted Scores
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("🎓 Actual vs Predicted Student Performance")
plt.grid(True)
plt.show()

# 📋 Step 11: Predict for a new student (example)
# Change the values below as needed:
new_student = {
    'math score': [75],
    'reading score': [80],
    'writing score': [78],
    'gender_male': [1],
    'race/ethnicity_group B': [0],
    'race/ethnicity_group C': [1],
    'race/ethnicity_group D': [0],
    'race/ethnicity_group E': [0],
    'parental level of education_bachelor\'s degree': [1],
    'parental level of education_high school': [0],
    'parental level of education_master\'s degree': [0],
    'parental level of education_some college': [0],
    'parental level of education_some high school': [0],
    'lunch_standard': [1],
    'test preparation course_completed': [1]
}

# Convert to DataFrame
new_student_df = pd.DataFrame(new_student)

# Ensure same columns as training data
missing_cols = set(X_train.columns) - set(new_student_df.columns)
for col in missing_cols:
    new_student_df[col] = 0

new_student_df = new_student_df[X_train.columns]

# Predict
predicted_score = model.predict(new_student_df)
print("\nPredicted Final Score for New Student:", round(predicted_score[0], 2))

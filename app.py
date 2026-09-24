import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data
# 0 = Fail, 1 = Pass
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Create and train model
model = LogisticRegression()
model.fit(X, y)

# Streamlit interface
st.title("🎓 Student Pass/Fail Predictor")
st.write("Predict whether a student will pass based on study hours.")

# User input
hours = st.number_input(
    "Enter study hours:",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

# Prediction button
if st.button("Predict Result"):

    prediction = model.predict([[hours]])
    probability = model.predict_proba([[hours]])

    if prediction[0] == 1:
        st.success("✅ Student is predicted to PASS!")
    else:
        st.error("❌ Student is predicted to FAIL!")

    st.write(
        f"Probability of Passing: "
        f"{probability[0][1] * 100:.2f}%"
    )

import streamlit as st
import joblib

# Load the trained spam detection model
model = joblib.load("spam_model.joblib")

# Title
st.title("Spam Message Classifier")

# Text input
message = st.text_area("Enter the message:")

# Predict button
if st.button("Check if Spam"):
    result = model.predict([message])
    if result[0] == "spam":
        st.error("🚨 This message is SPAM!")
    else:
        st.success("✅ This message is NOT spam.")

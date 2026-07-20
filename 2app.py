import streamlit as st
import joblib

# Load Model and Vectorizer
model = joblib.load("LR_NLP model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Page Configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

# Title
st.title("😊 AI Sentiment Analysis")
st.write("Enter a sentence below to predict its sentiment.")

# Text Input
user_input = st.text_area(
    "Enter your text",
    placeholder="Example: I really enjoyed this movie!"
)

# Predict Button
if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Transform input
        vector = tfidf.transform([user_input])

        # Prediction
        prediction = model.predict(vector)[0]

        # Display Result
        if str(prediction).lower() in ["positive", "1"]:
            st.success("😊 Positive Sentiment")

        elif str(prediction).lower() in ["negative", "0"]:
            st.error("😞 Negative Sentiment")

        else:
            st.info(f"Prediction: {prediction}")

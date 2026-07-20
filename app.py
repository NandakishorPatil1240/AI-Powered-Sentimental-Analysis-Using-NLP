import string
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import streamlit as st

# Ensure NLTK resources are downloaded
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


# -----------------------------------------------------------------------------
# 1. Text Preprocessing Functions (From Notebook)
# -----------------------------------------------------------------------------
def remove_punc(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text):
    return "".join([i for i in text if not i.isdigit()])


def remove_emojis(text):
    return "".join([i for i in text if i.isascii()])


def remove_stopwords(text):
    words = text.split()
    cleaned = [word for word in words if word.lower() not in stop_words]
    return " ".join(cleaned)


def full_preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_punc(text)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text


# -----------------------------------------------------------------------------
# 2. Streamlit Dashboard Layout
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Emotion Classifier Workspace", layout="wide")
st.title("🧠 Emotion Classification & Model Workspace")
st.write(
    "This app mirrors your notebook workflow, allowing you to train models on `train.txt` and predict text emotions live."
)

# Sidebar - File Upload & Mapping Setup
st.sidebar.header("Data Configuration")
uploaded_file = st.sidebar.file_uploader("Upload 'train.txt'", type=["txt"])

# Hardcoded mapping configuration based on your notebook's logic
emotion_mapping = {
    "sadness": 0,
    "anger": 1,
    "love": 2,
    "surprise": 3,
    "fear": 4,
    "joy": 5,
}

# THE FIX: Create a reverse mapping dictionary to turn numbers back to words
reverse_mapping = {v: k for k, v in emotion_mapping.items()}

# Main Workflow Control
if uploaded_file is not None:
    # Load original dataset
    df = pd.read_csv(
        uploaded_file, sep=";", header=None, names=["text", "emotion"]
    )

    tabs = st.tabs(["📊 Dataset Explorer", "🏋️ Train Models", "🔮 Live Inference"])

    # -------------------------------------------------------------------------
    # TAB 1: DATA EXPLORATION
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Missing Values Count")
        st.write(df.isnull().sum())

        # Map labels
        df["emotion_id"] = df["emotion"].map(emotion_mapping)

        # Preprocess text
        with st.spinner(
            "Processing text (lowering, removing punctuation/numbers/emojis/stopwords)..."
        ):
            df["cleaned_text"] = df["text"].apply(full_preprocess)

        st.subheader("Processed Data Preview")
        st.dataframe(
            df[["text", "cleaned_text", "emotion", "emotion_id"]].head(),
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # TAB 2: MODEL TRAINING
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Train-Test Split Evaluation")
        st.write(
            "Splitting data into **80% Training** and **20% Testing** configurations."
        )

        df["emotion_encoded"] = df["emotion"].map(emotion_mapping)

        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_text"], df["emotion_encoded"], test_size=0.20, random_state=40
        )

        if st.button("🚀 Run Notebook Training Routine"):
            progress_bar = st.progress(0)

            # --- Model 1: BoW + MultinomialNB ---
            bow_vectorizer = CountVectorizer()
            X_train_bow = bow_vectorizer.fit_transform(X_train)
            X_test_bow = bow_vectorizer.transform(X_test)

            nb_bow_model = MultinomialNB()
            nb_bow_model.fit(X_train_bow, y_train)
            pred_bow = nb_bow_model.predict(X_test_bow)
            acc_bow = accuracy_score(y_test, pred_bow)
            progress_bar.progress(33)

            # --- Model 2: TF-IDF + MultinomialNB ---
            tfidf_vectorizer = TfidfVectorizer()
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)

            nb_tfidf_model = MultinomialNB()
            nb_tfidf_model.fit(X_train_tfidf, y_train)
            pred_tfidf = nb_tfidf_model.predict(X_test_tfidf)
            acc_tfidf = accuracy_score(y_test, pred_tfidf)
            progress_bar.progress(66)

            # --- Model 3: TF-IDF + Logistic Regression ---
            logistic_model = LogisticRegression(max_iter=1000)
            logistic_model.fit(X_train_tfidf, y_train)
            pred_log = logistic_model.predict(X_test_tfidf)
            acc_log = accuracy_score(y_test, pred_log)
            progress_bar.progress(100)

            # Display Summary
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Naive Bayes (Bag of Words)", value=f"{acc_bow:.4f}")
            col2.metric(label="Naive Bayes (TF-IDF)", value=f"{acc_tfidf:.4f}")
            col3.metric(label="Logistic Regression (TF-IDF)", value=f"{acc_log:.4f}")

            # Export champions to disk state
            joblib.dump(logistic_model, "LR_NLP model.pkl")
            joblib.dump(tfidf_vectorizer, "tfidf.pkl")
            st.success(
                "Best artifacts ('LR_NLP.pkl' & 'tfidf.pkl') saved locally successfully!"
            )

    # -------------------------------------------------------------------------
    # TAB 3: LIVE INFERENCE
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Predict Custom Text Elements")
        user_input = st.text_area(
            "Enter a sentence to analyze its emotion:",
            "i feel strong and good overall",
        )

        if st.button("🔮 Classify Sentiment"):
            try:
                # Load models dynamically
                model = joblib.load("LR_NLP.pkl")
                vectorizer = joblib.load("tfidf.pkl")

                # Transform raw strings
                cleaned_input = full_preprocess(user_input)
                vectorized_input = vectorizer.transform([cleaned_input])

                # 1. Get the numeric prediction from the model (e.g., 5)
                numeric_prediction = model.predict(vectorized_input)[0]

                # 2. Convert that number back to the text string (e.g., "joy")
                named_emotion = reverse_mapping.get(
                    numeric_prediction, "unknown"
                )

                # 3. Choose a fun emoji based on the text string
                emojis = {
                    "sadness": "😢",
                    "anger": "😠",
                    "love": "❤️",
                    "surprise": "😮",
                    "fear": "😨",
                    "joy": "😄",
                }
                emoji = emojis.get(named_emotion, "")

                # Output visual formatting components - SAFE FROM ERROR!
                st.markdown("---")
                st.markdown(f"**Processed Clean Text Tokens:** `{cleaned_input}`")
                st.success(
                    f"Predicted Emotion: **{named_emotion.upper()}** {emoji}"
                )

            except FileNotFoundError:
                st.error(
                    "Error: Model binaries not found. Please navigate to the 'Train Models' tab and execute the training workflow first."
                )
else:
    st.info(
        "💡 Please upload your structural dataset pipeline source text document (`train.txt`) in the sidebar dashboard panel to load operational views."
    )

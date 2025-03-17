import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Page config
st.set_page_config(
    page_title="NLP Challenge | News Classifier", page_icon="📰", layout="wide"
)


# Download necessary NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download("stopwords")


@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model(path):
    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
        return model
    except:
        st.error(f"Failed to load model from {path}")
        return None


model = load_model("classical_model.pkl")
vectorizer = load_model("vectorizer.pkl")

download_nltk_resources()


# App title and description
st.title("📰 NLP News Classifier Model")
st.markdown(
    """
This application uses Classical Natural Language Processing to classify news as real or fake.
Upload your trained model or use the demo to test news articles.
"""
)


# Text preprocessing function
@st.cache_data
def preprocess_text(text):
    # Convert to lowercase and remove non-alphanumeric
    text = re.sub("[^a-zA-Z]", " ", text.lower())

    # Tokenize
    words = text.split()

    # Remove stopwords and stem
    ps = PorterStemmer()
    stopwords_set = set(stopwords.words("english"))
    words = [ps.stem(word) for word in words if word not in stopwords_set]

    # Join back to string
    return " ".join(words)


# Create tabs
testing_tab, performance_tab = st.tabs(["Test The Model", "Model Performance"])

with testing_tab:
    st.header("Test Your Model")

    # Text input
    st.subheader("Enter News Text")
    user_input = st.text_area("Paste news article here:", height=200)

    # Prediction
    if st.button("Predict") and user_input and model and vectorizer:
        # Preprocess the text
        processed_text = preprocess_text(user_input)

        # Vectorize
        text_vector = vectorizer.transform([processed_text])

        # Predict
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]

        # Display result
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 0:
                st.error("📢 Prediction: FAKE NEWS")
            else:
                st.success("✅ Prediction: REAL NEWS")

        with col2:
            # Confidence score
            confidence = probability[1] if prediction == 1 else probability[0]
            st.metric("Confidence", f"{confidence*100:.2f}%")

with performance_tab:
    st.header("Model Performance")

    # Placeholder for model metrics
    st.subheader("Model Evaluation Metrics")

    metrics = {"Accuracy": 0.99, "Precision": 0.99, "Recall": 0.99, "F1 Score": 0.99}

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
    col2.metric("Precision", f"{metrics['Precision']:.2f}")
    col3.metric("Recall", f"{metrics['Recall']:.2f}")
    col4.metric("F1 Score", f"{metrics['F1 Score']:.2f}")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    confusion_matrix = pd.DataFrame(
        [[3948, 31], [51, 3919]],
        columns=["Predicted Real", "Predicted Fake"],
        index=["Actual Real", "Actual Fake"],
    )
    st.table(confusion_matrix)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        """
    ### Classical NLP News Classifier Model
    This application demonstrates a trained machine learning model that can distinguish between real and fake news based on word patterns and content analysis.
    """
    )

    # Made by section for your presentation
    st.markdown("---")
    st.markdown(
        """
        ### Created by Team NeuFalcons:
        - Mohammed Almatrafi
        - Yazeed Alghamdi
        - Yasser Alshehri
        
        ### Class:
        Ironhack X SDA
        """
    )
    st.text("Date: March 2025")

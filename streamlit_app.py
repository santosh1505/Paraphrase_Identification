import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow.keras.backend as K
import nltk
import os
from PyPDF2 import PdfReader

max_len = 30

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

@st.cache_resource
def load_model_and_tokenizer():
    try:
        with open("models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        model = load_model('models/malstm_model.h5', custom_objects={'manhattan_distance': manhattan_distance})
        return model, tokenizer
    except FileNotFoundError as e:
        st.error(f"Error: Model files not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered = [w for w in tokens if w.isalnum() and w not in stop_words]
        return " ".join(filtered)
    return ""

def manhattan_distance(vects):
    x, y = vects
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

st.title("Paraphrase Identification")

model, tokenizer = load_model_and_tokenizer()

if model and tokenizer:
    st.subheader("Enter Questions Manually:")
    question1_manual = st.text_area("First Question:")
    question2_manual = st.text_area("Second Question:")

    st.subheader("Or Upload PDF Files:")
    uploaded_file1 = st.file_uploader("Upload PDF for First Question", type=["pdf"])
    uploaded_file2 = st.file_uploader("Upload PDF for Second Question", type=["pdf"])

    question1 = ""
    question2 = ""

    if uploaded_file1:
        question1 = extract_text_from_pdf(uploaded_file1)
        st.info("Text extracted from the first PDF.")
    else:
        question1 = question1_manual

    if uploaded_file2:
        question2 = extract_text_from_pdf(uploaded_file2)
        st.info("Text extracted from the second PDF.")
    else:
        question2 = question2_manual

    if st.button("Check Paraphrase"):
        if question1 and question2:
            q1_processed = preprocess(question1)
            q2_processed = preprocess(question2)

            q1_seq = pad_sequences(tokenizer.texts_to_sequences([q1_processed]), maxlen=max_len)
            q2_seq = pad_sequences(tokenizer.texts_to_sequences([q2_processed]), maxlen=max_len)

            prediction = model.predict([q1_seq, q2_seq])[0][0]
            similarity_score = f"{prediction:.4f}"

            st.subheader("Prediction:")
            st.write(f"Similarity Score: **{similarity_score}**")

            if prediction > 0.5:
                st.success("The questions are likely paraphrases.")
            else:
                st.warning("The questions are likely not paraphrases.")
        else:
            st.warning("Please enter or upload text for both questions.")
elif not model or not tokenizer:
    st.stop()
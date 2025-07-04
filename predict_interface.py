import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow.keras.backend as K
import nltk

max_len = 130

# Ensure necessary NLTK resources are downloaded (in case this script is run independently)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def preprocess(text):
    if isinstance(text, str): # Ensure text is a string
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered = [w for w in tokens if w.isalnum() and w not in stop_words]
        return " ".join(filtered)
    return "" # Handle potential non-string inputs

def manhattan_distance(vects):
    x, y = vects
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))

try:
    # Load tokenizer
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load model
    model = load_model('models/malstm_model.h5', custom_objects={'manhattan_distance': manhattan_distance})

    # Get user input
    q1 = input("Enter first question: ")
    q2 = input("Enter second question: ")

    q1_processed = preprocess(q1)
    q2_processed = preprocess(q2)

    q1_seq = pad_sequences(tokenizer.texts_to_sequences([q1_processed]), maxlen=max_len)
    q2_seq = pad_sequences(tokenizer.texts_to_sequences([q2_processed]), maxlen=max_len)

    pred = model.predict([q1_seq, q2_seq])[0][0]
    print(f"\nSimilarity score: {pred:.4f}")
    if pred > 0.5:
        print("The questions are likely paraphrases.")
    else:
        print("The questions are likely not paraphrases.")

except FileNotFoundError as e:
    print(f"Error: One of the model files ('models/tokenizer.pkl' or 'models/malstm_model.h5') was not found: {e}")
except Exception as e:
    print(f"An error occurred during prediction: {e}")
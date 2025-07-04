import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import tensorflow.keras.backend as K

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Preprocess function to clean the text
def preprocess(text):
    if isinstance(text, str):  # Ensure text is a string
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered = [w for w in tokens if w.isalnum() and w not in stop_words]
        return " ".join(filtered)
    return ""  # Handle potential non-string inputs

# Manhattan distance function for the LSTM
def manhattan_distance(vects):
    x, y = vects
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))

# Build the model
def build_model(vocab_size, embedding_dim, max_seq_length):
    input_1 = Input(shape=(max_seq_length,))
    input_2 = Input(shape=(max_seq_length,))

    embedding_layer = Embedding(vocab_size, embedding_dim, trainable=True)
    lstm_layer = LSTM(50)

    encoded_1 = lstm_layer(embedding_layer(input_1))
    encoded_2 = lstm_layer(embedding_layer(input_2))

    distance = Lambda(manhattan_distance, output_shape=lambda x: (x[0][0], 1))([encoded_1, encoded_2])

    model = Model(inputs=[input_1, input_2], outputs=[distance])
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])
    return model

# Load and preprocess data
try:
    df = pd.read_csv(r"C:\Users\Santo\Desktop\major_para\data\questions.csv")
    print("Columns in the dataset:", df.columns)

    # Ensure the correct columns are present
    if 'question1' in df.columns and 'question2' in df.columns and 'is_duplicate' in df.columns:
        df = df[['question1', 'question2', 'is_duplicate']].dropna()
    else:
        print("Error: Missing required columns in the dataset!")
        exit()

    # Apply preprocessing
    df['question1'] = df['question1'].apply(preprocess)
    df['question2'] = df['question2'].apply(preprocess)

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(pd.concat([df['question1'], df['question2']]))
    os.makedirs("models", exist_ok=True)
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer.word_index, f)

    q1 = tokenizer.texts_to_sequences(df['question1'])
    q2 = tokenizer.texts_to_sequences(df['question2'])

    max_len = 30
    q1 = pad_sequences(q1, maxlen=max_len)
    q2 = pad_sequences(q2, maxlen=max_len)

    labels = df['is_duplicate'].values

    # Train-test split
    X_train1, X_test1, X_train2, X_test2, y_train, y_test = train_test_split(q1, q2, labels, test_size=0.2, random_state=42)

    # Model training
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(vocab_size, 50, max_len)

    checkpoint = ModelCheckpoint('models/malstm_model.h5', save_best_only=True, verbose=1)
    model.fit([X_train1, X_train2], y_train, batch_size=64, epochs=10, validation_split=0.1, callbacks=[checkpoint])

except FileNotFoundError:
    print(f"Error: The file 'r\"C:\\Users\\Santo\\Desktop\\major_para\\data\\questions.csv\"' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()
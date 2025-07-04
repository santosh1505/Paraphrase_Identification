# Paraphrase Identification using Siamese MaLSTM

This project identifies if two questions are paraphrases using a Siamese Manhattan LSTM (MaLSTM) model trained on the Quora Question Pairs dataset. It includes tools for training, command-line prediction, and an interactive Streamlit web app with PDF input support.

## Core Functionality
* Determines semantic similarity between two input texts (questions).
* Predicts if they are paraphrases (same meaning).
* Supports text input directly or via PDF file uploads in the Streamlit app.

## Key Technologies
* **Model**: Siamese Manhattan LSTM (MaLSTM)
* **Text Preprocessing**: NLTK (tokenization, stop-word removal)
* **Frontend**: Streamlit
* **Backend**: Python, TensorFlow/Keras

## Dataset
* **Name**: Quora Question Pairs
* **Source**: [Kaggle - Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
* **Setup**:
    1.  Download `train.csv` from Kaggle, rename to `questions.csv`.
    2.  Create a `data/` folder in your project root and place `questions.csv` inside it.
    3.  **Crucial**: Modify the hardcoded path in `train_model.py` from `r"C:\Users\Santo\Desktop\major_para\data\questions.csv"` to `"data/questions.csv"` (or your actual path).

## Essential Setup & Installation

1.  **Python**: Ensure Python 3.7+ is installed. Create a virtual environment (recommended).
2.  **Dependencies**: Create a `requirements.txt` file with the content below and install using `pip install -r requirements.txt`:
    ```txt
    numpy
    pandas
    tensorflow
    scikit-learn
    nltk
    streamlit
    PyPDF2
    ```
3.  **NLTK Resources**: Run `python download_nltk_data.py` to download 'punkt' and 'stopwords'. The other scripts also attempt to download these if missing.

## How to Run

1.  **Train the Model (First time & if dataset changes):**
    * Ensure `questions.csv` is in `data/` and the path in `train_model.py` is correct.
    * Execute: `python train_model.py`
    * This saves `malstm_model.h5` and `tokenizer.pkl` to a `models/` directory.

2.  **Command-Line Prediction:**
    * Execute: `python predict_interface.py`
    * Enter two questions when prompted.

3.  **Streamlit Web Application:**
    * Execute: `streamlit run streamlit_app.py`
    * Open the URL shown in your browser. Input text manually or upload PDFs.

## Model Overview
The system uses a Siamese network with two identical LSTM branches that share weights. Each branch processes an input question. The Manhattan distance between the LSTM outputs is calculated and transformed into a similarity score (0 to 1) to predict if the questions are paraphrases.

## Files at a Glance
* `train_model.py`: Loads data, preprocesses, trains the MaLSTM, and saves the model/tokenizer.
* `predict_interface.py`: CLI tool for paraphrase prediction using the trained model.
* `streamlit_app.py`: Interactive web UI for paraphrase prediction with PDF support.
* `download_nltk_data.py`: Utility to download NLTK 'punkt' and 'stopwords' resources.
* `models/`: Stores the trained `malstm_model.h5` and `tokenizer.pkl`.
* `data/questions.csv`: Your Quora dataset (you need to add this).
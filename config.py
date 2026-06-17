import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")

# Create directories if they do not exist
for directory in [DATA_DIR, MODEL_DIR, CHARTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# File Paths
DATA_PATH = os.path.join(DATA_DIR, "fake_reviews.csv")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Data & Feature Extraction Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000  # Vocabulary size for TF-IDF

# Dataset Source URL
DATASET_URL = "https://raw.githubusercontent.com/SayamAlt/Fake-Reviews-Detection/master/fake%20reviews%20dataset.csv"

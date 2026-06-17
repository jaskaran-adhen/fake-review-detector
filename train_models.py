import os
import urllib.request
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config
from preprocess import clean_text

def download_dataset():
    """Download dataset from repository if it doesn't already exist."""
    if os.path.exists(config.DATA_PATH):
        print(f"Dataset already exists at {config.DATA_PATH}")
        return

    print(f"Dataset not found. Downloading from {config.DATASET_URL}...")
    try:
        # Create data directory if not exists
        os.makedirs(config.DATA_DIR, exist_ok=True)
        
        # User-Agent header to avoid potential bot blocks
        req = urllib.request.Request(
            config.DATASET_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(config.DATA_PATH, 'wb') as out_file:
            out_file.write(response.read())
        print("Download completed successfully.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def load_and_preprocess_data():
    """Load dataset, clean text columns, and map labels."""
    print("Loading dataset...")
    df = pd.read_csv(config.DATA_PATH)
    
    # Handle missing values
    df = df.dropna(subset=['text_'])
    df = df.drop_duplicates(subset=['text_'])
    
    print(f"Dataset loaded. Total reviews: {len(df)}")
    label_counts = df['label'].value_counts()
    print("Class distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Map labels: CG (Computer Generated) -> 1 (FAKE), OR (Original) -> 0 (REAL)
    df['label_num'] = df['label'].map({'CG': 1, 'OR': 0})
    
    # Check for unmapped labels
    if df['label_num'].isnull().any():
        print("[WARNING] Found unmapped labels. Dropping rows with invalid labels.")
        df = df.dropna(subset=['label_num'])
        df['label_num'] = df['label_num'].astype(int)
        
    print("Preprocessing text (this may take a few moments)...")
    total_rows = len(df)
    cleaned_texts = []
    for idx, text in enumerate(df['text_']):
        cleaned_texts.append(clean_text(text))
        if (idx + 1) % 5000 == 0 or (idx + 1) == total_rows:
            print(f"  Processed {idx + 1}/{total_rows} reviews...")
            
    df['cleaned_text'] = cleaned_texts
    
    # Drop rows that became empty after cleaning
    df = df[df['cleaned_text'].str.strip() != ""]
    print(f"Preprocessing completed. Remaining reviews: {len(df)}")
    
    return df

def train_and_save():
    # 1. Download & Preprocess
    download_dataset()
    df = load_and_preprocess_data()
    
    # 2. Split into Train & Test
    print("Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['label_num'], 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=df['label_num']
    )
    
    # 3. Vectorization (TF-IDF)
    print(f"Extracting TF-IDF features (max_features={config.MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=config.MAX_FEATURES)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save Vectorizer
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    print(f"TF-IDF Vectorizer saved to {config.VECTORIZER_PATH}")
    
    # 4. Train SVM (Calibrated for probability prediction)
    print("Training Support Vector Machine model...")
    base_svm = LinearSVC(random_state=config.RANDOM_STATE, dual=False, max_iter=2000)
    svm_model = CalibratedClassifierCV(estimator=base_svm, cv=5)
    svm_model.fit(X_train_vec, y_train)
    
    # Save SVM
    joblib.dump(svm_model, config.SVM_MODEL_PATH)
    print(f"SVM model saved to {config.SVM_MODEL_PATH}")
    
    # 5. Train Random Forest (Optimized hyperparameters for speed and size)
    print("Training Random Forest model (this may take a minute)...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train_vec, y_train)
    
    # Save Random Forest
    joblib.dump(rf_model, config.RF_MODEL_PATH)
    print(f"Random Forest model saved to {config.RF_MODEL_PATH}")
    
    # 6. Evaluate Models
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    
    models = {
        'SVM': svm_model,
        'Random Forest': rf_model
    }
    
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
        cm = confusion_matrix(y_test, preds)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'CM': cm
        }
        
        print(f"\n{name} Model Performance:")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print("  Confusion Matrix:")
        print(f"    Predicted REAL  Predicted FAKE")
        print(f"    Actual REAL  [{cm[0][0]}]             [{cm[0][1]}]")
        print(f"    Actual FAKE  [{cm[1][0]}]             [{cm[1][1]}]")

    # Save test set predictions summary for visualizer usage
    test_eval_summary = {
        'results': {
            name: {
                'Accuracy': results[name]['Accuracy'],
                'Precision': results[name]['Precision'],
                'Recall': results[name]['Recall'],
                'F1-Score': results[name]['F1-Score'],
                'CM': results[name]['CM'].tolist()
            } for name in results
        }
    }
    joblib.dump(test_eval_summary, os.path.join(config.MODEL_DIR, "eval_summary.pkl"))
    print("\nTraining and evaluation completed. Models are ready.")

if __name__ == "__main__":
    train_and_save()

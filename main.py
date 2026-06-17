import os
import sys
import pandas as pd
import numpy as np
import joblib

import config
from preprocess import clean_text
import visualize

def load_ml_assets():
    """Load trained models and vectorizer. Returns None if missing."""
    if not (os.path.exists(config.SVM_MODEL_PATH) and 
            os.path.exists(config.RF_MODEL_PATH) and 
            os.path.exists(config.VECTORIZER_PATH)):
        return None
        
    try:
        svm_model = joblib.load(config.SVM_MODEL_PATH)
        rf_model = joblib.load(config.RF_MODEL_PATH)
        vectorizer = joblib.load(config.VECTORIZER_PATH)
        
        # Load evaluation summary if available
        eval_summary = None
        eval_path = os.path.join(config.MODEL_DIR, "eval_summary.pkl")
        if os.path.exists(eval_path):
            eval_summary = joblib.load(eval_path)
            
        return svm_model, rf_model, vectorizer, eval_summary
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return None

def run_model_training():
    """Run model training by importing train_models module."""
    print("\nTraining models... Please wait.")
    try:
        import train_models
        train_models.train_and_save()
        print("Model training complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        return False

def get_word_influences(cleaned_text, vectorizer, svm_model):
    """
    Compute local word influences for a review based on SVM coefficients.
    Positive values indicate influence towards FAKE review prediction.
    Negative values indicate influence towards REAL review prediction.
    """
    if not cleaned_text.strip():
        return {}
        
    # Average the coefficients of the base classifiers across folds
    feature_names = vectorizer.get_feature_names_out()
    coefs = np.zeros(len(feature_names))
    
    try:
        for clf in svm_model.calibrated_classifiers_:
            base_est = getattr(clf, 'estimator', getattr(clf, 'base_estimator', None))
            if base_est is not None:
                coefs += base_est.coef_[0]
        coefs /= len(svm_model.calibrated_classifiers_)
    except Exception:
        # Fallback in case of shape/access issues
        return {}
        
    # Vectorize the individual text
    x_vec = vectorizer.transform([cleaned_text])
    
    # Get active features and multiply values by coefficients
    word_influences = {}
    row_indices, col_indices = x_vec.nonzero()
    for col in col_indices:
        word = feature_names[col]
        val = x_vec[0, col]
        influence = val * coefs[col]
        word_influences[word] = float(influence)
        
    return word_influences

def detect_suspicious_patterns(review_text, word_influences):
    """Detect red flags and list suspicious words."""
    red_flags = []
    
    # 1. Check for excessive exclamation marks or capital letters
    excl_count = review_text.count('!')
    if excl_count >= 3:
        red_flags.append(f"Excessive use of exclamation marks ({excl_count} found)")
        
    words = review_text.split()
    if len(words) > 0:
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        caps_ratio = len(caps_words) / len(words)
        if caps_ratio > 0.2:
            red_flags.append(f"Excessive capitalization ({caps_ratio*100:.1f}% of words in ALL CAPS)")
            
    # 2. Check for superlative words
    superlatives = ['amazing', 'best', 'ever', 'perfect', 'incredible', 'awesome', 'greatest', 'life-changing', 'revolutionary']
    found_sups = [w.lower() for w in words if w.lower() in superlatives]
    if len(found_sups) >= 3:
        red_flags.append(f"Excessive use of superlatives ({', '.join(set(found_sups))})")
        
    # 3. Check emotional language density (no details)
    # If the review is short but contains multiple strong sentiment indicators
    strong_words = ['love', 'hate', 'disappointed', 'terrible', 'wonderful', 'perfect', 'waste', 'worst', 'best']
    found_strong = [w.lower() for w in words if w.lower() in strong_words]
    if len(words) < 15 and len(found_strong) >= 2:
        red_flags.append("Emotional language without specific product details")

    # Filter word influences for positive scores (indicates fake)
    suspicious_words = sorted(
        [item for item in word_influences.items() if item[1] > 0.02],
        key=lambda x: x[1],
        reverse=True
    )
    
    return red_flags, suspicious_words[:5]

def analyze_single_review(svm_model, rf_model, vectorizer):
    """Handle choice 1: analyze a single review entered by the user."""
    print("\n" + "="*40)
    print("ANALYZE A SINGLE REVIEW")
    print("="*40)
    
    review = input("Enter your review: ").strip()
    if not review:
        print("[WARNING] Empty input. Returning to menu.")
        return
        
    # Clean and vectorize
    cleaned = clean_text(review)
    x_vec = vectorizer.transform([cleaned])
    
    # Predict probabilities (index 1 is FAKE probability)
    svm_prob = svm_model.predict_proba(x_vec)[0][1]
    rf_prob = rf_model.predict_proba(x_vec)[0][1]
    
    combined_prob = (svm_prob + rf_prob) / 2
    verdict = "FAKE" if combined_prob > 0.5 else "REAL"
    
    # Get word influences
    word_influences = get_word_influences(cleaned, vectorizer, svm_model)
    
    # Detect patterns
    red_flags, suspicious_words = detect_suspicious_patterns(review, word_influences)
    
    # Display terminal results
    print("\n" + "="*40)
    print("FAKE REVIEW DETECTION RESULTS")
    print("="*40)
    
    print("\nREVIEW ANALYZED:")
    print(f'"{review}"')
    
    print("\n" + "-"*40)
    print("MODEL PREDICTIONS")
    print("-"*40)
    print(f"SVM Model:           { 'FAKE' if svm_prob > 0.5 else 'REAL' } ({svm_prob*100:.1f}% FAKE confidence)")
    print(f"Random Forest:       { 'FAKE' if rf_prob > 0.5 else 'REAL' } ({rf_prob*100:.1f}% FAKE confidence)")
    
    print("\n" + "-"*40)
    print("COMBINED VERDICT")
    print("-"*40)
    print(f"Combined Verdict:    {verdict} ({combined_prob*100:.1f}% confidence)")
    if verdict == "FAKE":
        print("  [WARNING] High confidence in FAKE prediction")
    else:
        print("  [INFO] Review appears to be authentic")
        
    print("\n" + "-"*40)
    print("SUSPICIOUS PATTERNS & RED FLAGS")
    print("-"*40)
    
    if suspicious_words:
        print("Suspicious words found (influence scores):")
        for word, score in suspicious_words:
            print(f"  - \"{word}\" (weight: {score:.3f})")
    else:
        print("No heavily suspicious words identified.")
        
    if red_flags:
        print("\nRed flags detected:")
        for flag in red_flags:
            print(f"  - {flag}")
    else:
        print("\nNo obvious behavioral red flags found.")
        
    print("\n" + "-"*40)
    print("VISUALIZATION CHARTS")
    print("-"*40)
    print("[INFO] Launching interactive visualization windows...")
    print("  - Confidence Bar Chart")
    print("  - Feature Importance Chart")
    print("  - Word Cloud")
    print("  - Decision Gauge")
    
    # Run visualizations
    visualize.plot_confidence_comparison(svm_prob, rf_prob, verdict)
    visualize.plot_decision_gauge(combined_prob, verdict)
    visualize.plot_feature_importance(word_influences)
    visualize.plot_word_cloud(review, word_influences)
    
    print("\n" + "="*40)
    print(f"VERDICT: THIS REVIEW IS LIKELY {verdict}")
    print("="*40)

def analyze_multiple_reviews(svm_model, rf_model, vectorizer):
    """Handle choice 2: bulk analyze reviews from a file."""
    print("\n" + "="*40)
    print("BULK REVIEW ANALYSIS")
    print("="*40)
    
    file_path = input("Enter path to CSV or TXT file containing reviews: ").strip()
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found at: {file_path}")
        return
        
    reviews = []
    original_lines = []
    
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
            # Find a column containing reviews
            possible_cols = ['text_', 'text', 'review', 'review_text', 'body']
            review_col = None
            for col in df.columns:
                if col.lower() in possible_cols:
                    review_col = col
                    break
                    
            if review_col is None:
                # If no matching column found, use the first column
                review_col = df.columns[0]
                print(f"[INFO] No standard review column found. Using first column: '{review_col}'")
                
            original_lines = df[review_col].astype(str).tolist()
        else:
            # Assume plain text line-by-line file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return
        
    if not original_lines:
        print("[WARNING] No reviews found in the file.")
        return
        
    print(f"Processing {len(original_lines)} reviews... Please wait.")
    
    results = []
    for review in original_lines:
        cleaned = clean_text(review)
        if not cleaned:
            # Handle empty/missing reviews
            results.append({
                'Review': review,
                'SVM_Prob': 0.0,
                'RF_Prob': 0.0,
                'Combined_Prob': 0.0,
                'Verdict': 'UNRESOLVED'
            })
            continue
            
        x_vec = vectorizer.transform([cleaned])
        svm_prob = float(svm_model.predict_proba(x_vec)[0][1])
        rf_prob = float(rf_model.predict_proba(x_vec)[0][1])
        combined_prob = (svm_prob + rf_prob) / 2
        verdict = "FAKE" if combined_prob > 0.5 else "REAL"
        
        results.append({
            'Review': review,
            'SVM_Prob': svm_prob,
            'RF_Prob': rf_prob,
            'Combined_Prob': combined_prob,
            'Verdict': verdict
        })
        
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    total = len(results_df)
    fake_count = sum(results_df['Verdict'] == 'FAKE')
    real_count = sum(results_df['Verdict'] == 'REAL')
    
    print("\n" + "-"*40)
    print("BULK ANALYSIS SUMMARY")
    print("-"*40)
    print(f"Total Reviews Analyzed:      {total}")
    print(f"Fake Reviews Detected:       {fake_count} ({fake_count/total*100:.1f}%)")
    print(f"Real Reviews Detected:       {real_count} ({real_count/total*100:.1f}%)")
    print(f"Average SVM FAKE Confidence: {results_df['SVM_Prob'].mean()*100:.1f}%")
    print(f"Average RF FAKE Confidence:  {results_df['RF_Prob'].mean()*100:.1f}%")
    
    export_opt = input("\nWould you like to export results to a CSV file? (y/n): ").strip().lower()
    if export_opt == 'y':
        export_dir = os.path.join(config.BASE_DIR, "output")
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, "bulk_analysis_results.csv")
        results_df.to_csv(export_path, index=False)
        print(f"Results successfully saved to: {export_path}")

def display_performance(eval_summary):
    """Display model metrics and display evaluation heatmaps."""
    if eval_summary is None:
        print("\n[WARNING] Evaluation data summary not found. Retrain models to generate metrics.")
        return
        
    results = eval_summary.get('results', {})
    
    print("\n" + "="*40)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*40)
    
    print(f"{'Metric':<15} | {'SVM':<12} | {'Random Forest':<15}")
    print("-" * 50)
    for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        svm_score = results.get('SVM', {}).get(m, 0.0) * 100
        rf_score = results.get('Random Forest', {}).get(m, 0.0) * 100
        print(f"{m:<15} | {svm_score:>10.2f}% | {rf_score:>13.2f}%")
    print("-" * 50)
    
    # Print Confusion Matrices side-by-side
    print("\nCONFUSION MATRIX COMPARISON")
    for name in ['SVM', 'Random Forest']:
        cm = results.get(name, {}).get('CM', [[0, 0], [0, 0]])
        print(f"\n{name} Confusion Matrix:")
        print(f"               Predicted REAL   Predicted FAKE")
        print(f"  Actual REAL      [{cm[0][0]}]            [{cm[0][1]}]")
        print(f"  Actual FAKE      [{cm[1][0]}]            [{cm[1][1]}]")
        
    print("\n[INFO] Launching performance charts...")
    visualize.plot_model_comparison(eval_summary)
    visualize.plot_confusion_matrix(eval_summary)

def main():
    print("========================================")
    print("FAKE REVIEW DETECTION SYSTEM")
    print("========================================")
    
    assets = load_ml_assets()
    if assets is None:
        print("[INFO] Model files or vectorizer are missing.")
        train_opt = input("Would you like to train the models now? (y/n): ").strip().lower()
        if train_opt == 'y':
            success = run_model_training()
            if success:
                assets = load_ml_assets()
            else:
                print("[ERROR] Failed to build models. Exiting.")
                sys.exit(1)
        else:
            print("[INFO] Cannot run predictions without models. Exiting.")
            sys.exit(0)
            
    svm_model, rf_model, vectorizer, eval_summary = assets
    
    while True:
        print("\n" + "="*40)
        print("MAIN MENU")
        print("="*40)
        print("1. Analyze a single review")
        print("2. Analyze multiple reviews (from file)")
        print("3. Compare model performance")
        print("4. Retrain models")
        print("5. Exit")
        print("="*40)
        
        choice = input("Select an option (1-5): ").strip()
        
        if choice == '1':
            analyze_single_review(svm_model, rf_model, vectorizer)
        elif choice == '2':
            analyze_multiple_reviews(svm_model, rf_model, vectorizer)
        elif choice == '3':
            display_performance(eval_summary)
        elif choice == '4':
            confirm = input("Are you sure you want to retrain the models? (y/n): ").strip().lower()
            if confirm == 'y':
                success = run_model_training()
                if success:
                    # Reload assets
                    new_assets = load_ml_assets()
                    if new_assets:
                        svm_model, rf_model, vectorizer, eval_summary = new_assets
                        print("Models reloaded successfully.")
        elif choice == '5':
            print("\nThank you for using the Fake Review Detection System. Goodbye.")
            break
        else:
            print("[WARNING] Invalid choice. Please select 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib

# Set matplotlib backend to Agg to prevent GUI popups
matplotlib.use('Agg')

import config
from preprocess import clean_text
import main
import visualize

# Page configuration
st.set_page_config(
    page_title="Fake Review Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS with imported typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.025em;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Premium Grid Cards */
    .metric-card-custom {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .metric-card-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.5rem;
    }
    
    /* Verdict Status Panels */
    .verdict-panel-fake {
        background-color: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 6px solid #f43f5e;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .verdict-panel-real {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-left: 6px solid #10b981;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    
    .verdict-title-fake {
        font-size: 1.4rem;
        font-weight: 700;
        color: #9f1239;
        margin: 0 0 0.5rem 0;
    }
    .verdict-title-real {
        font-size: 1.4rem;
        font-weight: 700;
        color: #065f46;
        margin: 0 0 0.5rem 0;
    }
    
    .verdict-desc-fake {
        font-size: 0.95rem;
        color: #e11d48;
        font-weight: 500;
        margin: 0;
    }
    .verdict-desc-real {
        font-size: 0.95rem;
        color: #059669;
        font-weight: 500;
        margin: 0;
    }
    
    .verdict-details-text {
        font-size: 0.95rem;
        color: #334155;
        margin-top: 0.8rem;
        line-height: 1.5;
    }

    /* Sidebar visual adjustments */
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    """Load model assets once and cache them."""
    assets = main.load_ml_assets()
    return assets

# Load models
assets = load_assets()

if assets is None:
    st.error("Model files or vectorizer are missing. Please train the models first.")
    if st.button("Train Models Now"):
        with st.spinner("Training models (this might take a minute)..."):
            success = main.run_model_training()
            if success:
                st.success("Models trained successfully! Please reload the page.")
                st.rerun()
            else:
                st.error("Failed to train models. Check terminal logs.")
    st.stop()

svm_model, rf_model, vectorizer, eval_summary = assets

# Sidebar Navigation
st.sidebar.markdown("<h2 style='font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;'>Review Detector</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 0.9rem; color: #64748b;'>Model prediction interface and explainability toolbox.</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

menu_choice = st.sidebar.radio(
    "Navigation Menu",
    ["Single Review Analysis", "Bulk File Analysis", "Model Performance Metrics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size: 0.85rem; color: #64748b; font-weight: 600;'>System Management</p>", unsafe_allow_html=True)
if st.sidebar.button("Retrain Models", use_container_width=True):
    with st.spinner("Retraining models..."):
        success = main.run_model_training()
        if success:
            st.sidebar.success("Models retrained successfully!")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.sidebar.error("Failed to retrain models.")

# Main Page Header
st.markdown('<div class="main-title">Fake Review Detection Dashboard</div>', unsafe_allow_html=True)

if menu_choice == "Single Review Analysis":
    st.markdown('<div class="main-subtitle">Analyze a customer review in real-time using calibrated classification models</div>', unsafe_allow_html=True)
    
    # Input Area Container
    st.markdown("<h4 style='font-weight: 700; color: #0f172a;'>Customer Review Input</h4>", unsafe_allow_html=True)
    review_input = st.text_area(
        label="Review Text",
        label_visibility="collapsed",
        height=140, 
        placeholder="Paste your review text here (e.g. 'I bought this product. Although it works...')"
    )
    
    if st.button("Analyze Input Review", type="primary", use_container_width=True):
        if not review_input.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            # Run prediction
            cleaned = clean_text(review_input)
            x_vec = vectorizer.transform([cleaned])
            
            svm_prob = svm_model.predict_proba(x_vec)[0][1]
            rf_prob = rf_model.predict_proba(x_vec)[0][1]
            combined_prob = (svm_prob + rf_prob) / 2
            verdict = "FAKE" if combined_prob > 0.5 else "REAL"
            
            word_influences = main.get_word_influences(cleaned, vectorizer, svm_model)
            red_flags, suspicious_words = main.detect_suspicious_patterns(review_input, word_influences)
            
            # Create charts files (saved in output/charts)
            visualize.plot_confidence_comparison(svm_prob, rf_prob, verdict, "web_confidence.png")
            visualize.plot_decision_gauge(combined_prob, verdict, "web_gauge.png")
            visualize.plot_feature_importance(word_influences, "web_importance.png")
            visualize.plot_word_cloud(review_input, word_influences, "web_wordcloud.png")
            
            # Layout Columns
            col1, col2 = st.columns([1.1, 0.9], gap="large")
            
            with col1:
                st.markdown("<h4 style='font-weight: 700; color: #0f172a; margin-bottom: 1rem;'>Analysis Verdict</h4>", unsafe_allow_html=True)
                
                # Combined verdict card
                if verdict == "FAKE":
                    st.markdown(f"""
                    <div class="verdict-panel-fake">
                        <div class="verdict-title-fake">Verdict: LIKELY FAKE</div>
                        <div class="verdict-desc-fake">Combined confidence score: {combined_prob*100:.1f}% FAKE probability</div>
                        <div class="verdict-details-text">
                            <strong>Warning:</strong> The review text patterns closely align with automated, computer-generated review templates in the database. Positive indicators are heavily generic without specific product nouns.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-panel-real">
                        <div class="verdict-title-real">Verdict: LIKELY REAL</div>
                        <div class="verdict-desc-real">Combined confidence score: {(1-combined_prob)*100:.1f}% REAL probability</div>
                        <div class="verdict-details-text">
                            <strong>Info:</strong> This review features a natural human writing style. The text contains structural variance and logical transition patterns typical of human reviews.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model predictions breakdown
                st.markdown("<h5 style='font-weight: 700; color: #0f172a; margin-top: 1.5rem;'>Model Confidence Scores</h5>", unsafe_allow_html=True)
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.markdown(f"""
                    <div class="metric-card-custom" style="padding: 1rem;">
                        <div class="metric-card-label" style="font-size: 0.8rem;">SVM Model</div>
                        <div style="font-size: 1.4rem; font-weight: 800; margin-top: 0.3rem; color: #4f46e5;">{svm_prob*100:.1f}% FAKE</div>
                    </div>
                    """, unsafe_allow_html=True)
                with mcol2:
                    st.markdown(f"""
                    <div class="metric-card-custom" style="padding: 1rem;">
                        <div class="metric-card-label" style="font-size: 0.8rem;">Random Forest</div>
                        <div style="font-size: 1.4rem; font-weight: 800; margin-top: 0.3rem; color: #64748b;">{rf_prob*100:.1f}% FAKE</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Red flags
                st.markdown("<h5 style='font-weight: 700; color: #0f172a; margin-top: 1rem;'>Linguistic Red Flags</h5>", unsafe_allow_html=True)
                if red_flags:
                    for flag in red_flags:
                        st.markdown(f"<div style='padding: 0.5rem; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; margin-bottom: 0.5rem; font-size: 0.9rem; color: #475569;'><strong>Flag:</strong> {flag}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='font-size: 0.9rem; color: #64748b;'>No obvious linguistic red flags detected in the review structure.</div>", unsafe_allow_html=True)
                    
                # Suspicious Words Table
                st.markdown("<h5 style='font-weight: 700; color: #0f172a; margin-top: 1.5rem;'>Influential Word Weights</h5>", unsafe_allow_html=True)
                if suspicious_words:
                    word_df = pd.DataFrame(suspicious_words, columns=["Word Token", "SVM Attribution Weight"])
                    st.dataframe(word_df, use_container_width=True, hide_index=True)
                else:
                    st.markdown("<div style='font-size: 0.9rem; color: #64748b;'>No heavily weighted tokens identified.</div>", unsafe_allow_html=True)
                    
            with col2:
                st.markdown("<h4 style='font-weight: 700; color: #0f172a; margin-bottom: 1rem;'>Model Explainability Charts</h4>", unsafe_allow_html=True)
                
                # Display charts generated
                confidence_path = os.path.join(config.CHARTS_DIR, "web_confidence.png")
                gauge_path = os.path.join(config.CHARTS_DIR, "web_gauge.png")
                importance_path = os.path.join(config.CHARTS_DIR, "web_importance.png")
                wc_path = os.path.join(config.CHARTS_DIR, "web_wordcloud.png")
                
                tab1, tab2 = st.tabs(["Confidence Meters", "Word Influences"])
                
                with tab1:
                    if os.path.exists(confidence_path):
                        st.image(confidence_path, use_container_width=True)
                    if os.path.exists(gauge_path):
                        st.image(gauge_path, use_container_width=True)
                        
                with tab2:
                    if os.path.exists(importance_path):
                        st.image(importance_path, use_container_width=True)
                    if os.path.exists(wc_path):
                        st.image(wc_path, use_container_width=True)

elif menu_choice == "Bulk File Analysis":
    st.markdown('<div class="main-subtitle">Upload multiple reviews from a dataset to identify computer-generated reviews in bulk</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a CSV (containing review column) or TXT file (one review per line)", 
        type=["csv", "txt"]
    )
    
    if uploaded_file is not None:
        try:
            reviews = []
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                possible_cols = ['text_', 'text', 'review', 'review_text', 'body']
                review_col = None
                for col in df.columns:
                    if col.lower() in possible_cols:
                        review_col = col
                        break
                if review_col is None:
                    review_col = df.columns[0]
                    st.info(f"Using column '{review_col}' as the review text column.")
                reviews = df[review_col].astype(str).tolist()
            else:
                reviews = [line.decode("utf-8").strip() for line in uploaded_file if line.strip()]
                
            if not reviews:
                st.warning("No reviews found in the uploaded file.")
            else:
                st.write(f"Loaded {len(reviews)} reviews. Performing classifications...")
                
                results = []
                for idx, review in enumerate(reviews):
                    cleaned = clean_text(review)
                    if not cleaned:
                        results.append({
                            'Original Review': review,
                            'SVM Confidence': 0.0,
                            'RF Confidence': 0.0,
                            'Combined Confidence': 0.0,
                            'Verdict': 'UNRESOLVED'
                        })
                        continue
                        
                    x_vec = vectorizer.transform([cleaned])
                    svm_prob = float(svm_model.predict_proba(x_vec)[0][1])
                    rf_prob = float(rf_model.predict_proba(x_vec)[0][1])
                    combined_prob = (svm_prob + rf_prob) / 2
                    verdict = "FAKE" if combined_prob > 0.5 else "REAL"
                    
                    results.append({
                        'Original Review': review,
                        'SVM Confidence': svm_prob,
                        'RF Confidence': rf_prob,
                        'Combined Confidence': combined_prob,
                        'Verdict': verdict
                    })
                    
                res_df = pd.DataFrame(results)
                
                # Metrics Summary Cards
                total = len(res_df)
                fake_count = sum(res_df['Verdict'] == 'FAKE')
                real_count = sum(res_df['Verdict'] == 'REAL')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card-custom">
                        <div class="metric-card-label">Total Reviews</div>
                        <div class="metric-card-value">{total}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card-custom" style="border-bottom: 4px solid #f43f5e;">
                        <div class="metric-card-label">Likely Fake Reviews</div>
                        <div class="metric-card-value" style="color: #f43f5e;">{fake_count} <span style="font-size: 1.1rem; color: #64748b; font-weight: 500;">({fake_count/total*100:.1f}%)</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card-custom" style="border-bottom: 4px solid #10b981;">
                        <div class="metric-card-label">Likely Real Reviews</div>
                        <div class="metric-card-value" style="color: #10b981;">{real_count} <span style="font-size: 1.1rem; color: #64748b; font-weight: 500;">({real_count/total*100:.1f}%)</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display dataframe results
                st.markdown("<h4 style='font-weight: 700; color: #0f172a; margin-top: 1.5rem;'>Bulk Prediction Table</h4>", unsafe_allow_html=True)
                st.dataframe(res_df, use_container_width=True)
                
                # Export results
                csv_buffer = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Prediction Results CSV",
                    data=csv_buffer,
                    file_name="bulk_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Failed to process file: {e}")

elif menu_choice == "Model Performance Metrics":
    st.markdown('<div class="main-subtitle">Inspect holdout evaluation performance data and visual model metrics</div>', unsafe_allow_html=True)
    
    if eval_summary is None:
        st.warning("Evaluation data summary not found. Retrain the models to generate performance metrics.")
    else:
        # Tables comparison
        results = eval_summary.get('results', {})
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        comp_data = []
        for m in metrics:
            comp_data.append({
                'Performance Metric': m,
                'SVM Model (Calibrated)': f"{results.get('SVM', {}).get(m, 0.0)*100:.2f}%",
                'Random Forest Classifier': f"{results.get('Random Forest', {}).get(m, 0.0)*100:.2f}%"
            })
            
        comp_df = pd.DataFrame(comp_data)
        
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.markdown("<h4 style='font-weight: 700; color: #0f172a; margin-bottom: 1rem;'>Evaluation Performance Report</h4>", unsafe_allow_html=True)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # Confusion matrices print
            st.markdown("<h4 style='font-weight: 700; color: #0f172a; margin-top: 2rem; margin-bottom: 1rem;'>Confusion Matrices (Test Set)</h4>", unsafe_allow_html=True)
            for name in ['SVM', 'Random Forest']:
                cm = results.get(name, {}).get('CM', [[0, 0], [0, 0]])
                st.markdown(f"**{name} Classifier**:")
                cm_df = pd.DataFrame(
                    cm, 
                    index=["Actual REAL", "Actual FAKE"], 
                    columns=["Predicted REAL", "Predicted FAKE"]
                )
                st.dataframe(cm_df, use_container_width=True)
                
        with col2:
            st.markdown("<h4 style='font-weight: 700; color: #0f172a; margin-bottom: 1rem;'>Evaluation Visualizations</h4>", unsafe_allow_html=True)
            
            # Create files
            visualize.plot_model_comparison(eval_summary, "web_model_comparison.png")
            visualize.plot_confusion_matrix(eval_summary, "web_confusion_matrix.png")
            
            comp_path = os.path.join(config.CHARTS_DIR, "web_model_comparison.png")
            cm_path = os.path.join(config.CHARTS_DIR, "web_confusion_matrix.png")
            
            tab1, tab2 = st.tabs(["Metric Comparison", "Confusion Matrix Heatmap"])
            with tab1:
                if os.path.exists(comp_path):
                    st.image(comp_path, use_container_width=True)
            with tab2:
                if os.path.exists(cm_path):
                    st.image(cm_path, use_container_width=True)

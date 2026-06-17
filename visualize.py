import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import config

# Global Matplotlib Configurations for Minimalist Design
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Plus Jakarta Sans', 'Arial', 'Helvetica']
plt.rcParams['text.color'] = '#1e293b'
plt.rcParams['axes.labelcolor'] = '#475569'
plt.rcParams['xtick.color'] = '#64748b'
plt.rcParams['ytick.color'] = '#64748b'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Simple, Clean Color Palette
COLOR_PRIMARY = '#3b82f6'    # Clean Blue
COLOR_SECONDARY = '#94a3b8'  # Muted Slate
COLOR_FAKE = '#ef4444'       # Soft Red
COLOR_REAL = '#22c55e'       # Soft Green
COLOR_BG = '#f1f5f9'         # Light Gray Track

def save_and_show_plot(fig, filename):
    """Save the figure to output/charts and show it if GUI is available."""
    os.makedirs(config.CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(config.CHARTS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    
    try:
        plt.show()
    except Exception:
        plt.close(fig)

def plot_confidence_comparison(svm_prob, rf_prob, verdict, filename="confidence_comparison.png"):
    """
    Chart 1: Model Confidence Bar Chart
    Clean, simple horizontal bars.
    """
    fig, ax = plt.subplots(figsize=(6, 2))
    models = ['Random Forest', 'SVM']
    probabilities = [rf_prob * 100, svm_prob * 100]
    
    # Draw background tracks first (light gray)
    ax.barh(models, [100, 100], color=COLOR_BG, height=0.35, edgecolor=None)
    # Draw actual filled values
    bars = ax.barh(models, probabilities, color=[COLOR_SECONDARY, COLOR_PRIMARY], height=0.35, edgecolor=None)
    
    # Customise chart
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fake Probability (%)", fontsize=8, fontweight='medium')
    ax.set_title(f"Model Predictions (Verdict: {verdict})", fontsize=10, fontweight='bold', pad=10, loc='left')
    
    # Add labels on the right
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                va='center', ha='left', fontweight='bold', fontsize=8, color='#0f172a')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Remove tick lines
    ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    save_and_show_plot(fig, filename)

def plot_decision_gauge(combined_prob, verdict, filename="decision_gauge.png"):
    """
    Chart 2: Decision Gauge
    A clean, minimalist progress slider instead of a heavy color gradient.
    """
    fig, ax = plt.subplots(figsize=(7, 1.4))
    
    # Draw background track
    ax.plot([0, 100], [5, 5], color=COLOR_BG, linewidth=10, solid_capstyle='round', zorder=1)
    
    # Fill the track up to the score
    val = combined_prob * 100
    fill_color = COLOR_FAKE if verdict == "FAKE" else COLOR_REAL
    
    if val > 0:
        ax.plot([0, val], [5, 5], color=fill_color, linewidth=10, solid_capstyle='round', zorder=2)
    
    # Draw indicator knob
    ax.plot(val, 5, marker='o', markersize=14, color='#0f172a', markeredgecolor='white', markeredgewidth=2, zorder=3)
    
    # Label above knob
    ax.text(val, 7.5, f"{val:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9, color='#0f172a')
    
    # Customise chart
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 12)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['REAL', 'Likely Real', 'Uncertain', 'Likely Fake', 'FAKE'], fontsize=8)
    ax.get_yaxis().set_visible(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', length=0, pad=8)
    
    ax.set_title(f"Verdict: Review is {verdict}", fontsize=10, fontweight='bold', pad=15, loc='left')
    
    plt.tight_layout()
    save_and_show_plot(fig, filename)

def plot_feature_importance(word_influences, filename="feature_importance.png"):
    """
    Chart 3: Feature Importance (Top Words)
    Clean vertical bar chart of word impacts.
    """
    if not word_influences:
        return
        
    # Sort and take top 6 words for cleaner display
    sorted_words = sorted(word_influences.items(), key=lambda item: abs(item[1]), reverse=True)
    top_words = sorted_words[:6]
    
    words = [item[0] for item in top_words][::-1]
    values = [item[1] for item in top_words][::-1]
    
    colors = [COLOR_FAKE if v > 0 else COLOR_REAL for v in values]
    
    fig, ax = plt.subplots(figsize=(6, 2.8))
    
    # Draw bars
    bars = ax.barh(words, values, color=colors, height=0.45, edgecolor=None)
    
    ax.axvline(0, color='#64748b', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Attribution Score", fontsize=8, fontweight='medium')
    ax.set_title("Word Influence Analysis", fontsize=10, fontweight='bold', pad=10, loc='left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    save_and_show_plot(fig, filename)

def plot_word_cloud(review_text, word_influences, filename="wordcloud.png"):
    """
    Chart 4: Word Cloud
    Simple, flat color palette.
    """
    if not review_text.strip():
        return

    words = review_text.lower().split()
    word_freq = {}
    for w in words:
        if len(w) > 2:
            word_freq[w] = word_freq.get(w, 0) + 1
            
    wordcloud = WordCloud(
        width=600, 
        height=240, 
        background_color='white',
        max_words=30,
        random_state=42
    ).generate_from_frequencies(word_freq)
    
    # Coloring function using our clean red and green accents
    def color_func(word, **kwargs):
        val = word_influences.get(word, 0)
        if val > 0.01:
            return COLOR_FAKE
        elif val < -0.01:
            return COLOR_REAL
        else:
            return COLOR_SECONDARY

    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.imshow(wordcloud.recolor(color_func=color_func), interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Frequency & Influence", fontsize=10, fontweight='bold', pad=10, loc='left')
    
    plt.tight_layout()
    save_and_show_plot(fig, filename)

def plot_model_comparison(eval_summary, filename="model_comparison.png"):
    """
    Chart 5: Model Performance Comparison
    Clean grouped bar chart.
    """
    results = eval_summary.get('results', {})
    if not results:
        return
        
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    svm_scores = [results['SVM'][m] * 100 for m in metrics]
    rf_scores = [results['Random Forest'][m] * 100 for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.28
    
    fig, ax = plt.subplots(figsize=(7, 3))
    
    rects1 = ax.bar(x - width/2, svm_scores, width, label='SVM', color=COLOR_PRIMARY, edgecolor=None)
    rects2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color=COLOR_SECONDARY, edgecolor=None)
    
    ax.set_ylabel('Percentage (%)', fontsize=8, fontweight='medium')
    ax.set_title('Overall Performance Comparison', fontsize=10, fontweight='bold', pad=10, loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    save_and_show_plot(fig, filename)

def plot_confusion_matrix(eval_summary, filename="confusion_matrix_heatmap.png"):
    """
    Chart 6: Confusion Matrix Heatmap
    Minimalist heatmap.
    """
    results = eval_summary.get('results', {})
    if not results or 'SVM' not in results:
        return
        
    cm = np.array(results['SVM']['CM'])
    
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    
    # Simple, clean blue heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'], ax=ax,
                linewidths=1, linecolor='#f8fafc', 
                annot_kws={'fontsize': 9, 'weight': 'bold', 'color': '#0f172a'})
    
    ax.set_xlabel('Predicted Label', fontsize=8, fontweight='semibold')
    ax.set_ylabel('Actual Label', fontsize=8, fontweight='semibold')
    ax.set_title('SVM Confusion Matrix', fontsize=10, fontweight='bold', pad=10, loc='left')
    
    # Simple borders
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('#e2e8f0')
        
    ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    save_and_show_plot(fig, filename)

# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="SecNect - Failed Login Detector",
    page_icon="🔐",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #3d3b37 0%, #5c5952 50%, #8b7355 100%);
    }
    
    /* Top bar styling */
    .top-bar {
        background: linear-gradient(90deg, #2d2b27 0%, #3d3b37 50%, #5c5952 100%);
        padding: 1rem 2rem;
        border-bottom: 3px solid #d4af37;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo {
        width: 50px;
        height: 50px;
        background: linear-gradient(45deg, #d4af37, #f0e68c);
        clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: bold;
        color: #2d2b27;
    }
    
    .company-name {
        font-size: 2rem;
        font-weight: 700;
        color: #f0e68c;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* Card styling */
    .stContainer > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 0;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f0e68c !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(240, 230, 140, 0.1) 100%);
        border: 1px solid rgba(212, 175, 55, 0.3);
        padding: 1rem;
        border-radius: 0;
        clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px));
    }
    
    [data-testid="metric-container"] > div {
        color: #f0e68c !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #d4af37, #f0e68c);
        color: #2d2b27;
        border: none;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        clip-path: polygon(0 0, calc(100% - 15px) 0, 100% 15px, 100% 100%, 15px 100%, 0 calc(100% - 15px));
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(212, 175, 55, 0.3);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(212, 175, 55, 0.5);
        border-radius: 0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #d4af37, #f0e68c);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d2b27 0%, #3d3b37 100%);
    }
    
    /* Text color */
    p, .stText {
        color: #e0e0e0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(212, 175, 55, 0.1);
        color: #f0e68c !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #8b7355, #a68b5b);
    }
    
    /* Divider */
    hr {
        border-color: rgba(212, 175, 55, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Text normalization function
def normalize_text(text):
    """
    Normalize log text by removing timestamps, IPs, and numbers
    to focus on the semantic content
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove timestamps (various formats)
    text = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    
    # Remove IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '', text)
    
    # Remove ports
    text = re.sub(r':\d{1,5}\b', '', text)
    
    # Remove standalone numbers (but keep numbers that are part of words)
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Load positive examples
@st.cache_data
def load_positive_examples():
    try:
        df = pd.read_csv('failed_login_logs.csv')
        df['normalized_log'] = df['Log'].apply(normalize_text)
        return df
    except FileNotFoundError:
        st.error("failed_login_logs.csv not found. Please ensure it's in the app directory.")
        return None

# Main app
def main():
    # Top bar with logo and company name
    st.markdown("""
    <div class="top-bar">
        <div class="logo-container">
            <div class="logo">S</div>
            <div class="company-name">SecNect</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("🔐 Failed Login Detector")
    st.markdown("""
    <p style='color: #e0e0e0; font-size: 1.1rem; margin-bottom: 2rem;'>
    Advanced semantic analysis to identify potential failed login events in log files using state-of-the-art AI technology.
    </p>
    """, unsafe_allow_html=True)
    
    # Load positive examples
    positive_examples_df = load_positive_examples()
    if positive_examples_df is None:
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <div class="logo" style='width: 80px; height: 80px; margin: 0 auto; font-size: 40px;'>S</div>
            <h2 style='color: #f0e68c; margin-top: 1rem;'>Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6,
            step=0.05,
            help="Log lines with similarity scores above this threshold will be highlighted"
        )
        
        top_n = st.number_input(
            "Number of top results to display",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
    
    # File upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <h2 style='color: #f0e68c;'>📁 Upload Log File</h2>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a log file", 
            type=['log', 'txt', 'csv'],
            help="Upload a log file in .log, .txt, or .csv format"
        )
    
    if uploaded_file is not None:
        # Read the uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if len(df.columns) > 1:
                    log_lines = df.iloc[:, -1].tolist()
                else:
                    log_lines = df.iloc[:, 0].tolist()
            else:
                content = uploaded_file.read().decode('utf-8')
                log_lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.success(f"✅ Successfully loaded {len(log_lines)} log lines")
            
            # Show sample of uploaded file
            with st.expander("📋 View first 5 log lines"):
                for i, line in enumerate(log_lines[:5]):
                    st.text(f"{i+1}: {line}")
            
            # Process button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔍 Analyze Log File", type="primary", use_container_width=True):
                    with st.spinner("🧠 Analyzing log file..."):
                        # Load model
                        model = load_model()
                        
                        # Normalize texts
                        positive_texts = positive_examples_df['normalized_log'].tolist()
                        normalized_log_lines = [normalize_text(line) for line in log_lines]
                        
                        # Compute embeddings
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Computing embeddings for positive examples...")
                        positive_embeddings = model.encode(positive_texts)
                        progress_bar.progress(33)
                        
                        status_text.text("Computing embeddings for log lines...")
                        target_embeddings = model.encode(normalized_log_lines)
                        progress_bar.progress(66)
                        
                        status_text.text("Computing similarities...")
                        similarities = cosine_similarity(target_embeddings, positive_embeddings)
                        max_similarities = np.max(similarities, axis=1)
                        most_similar_positive_idx = np.argmax(similarities, axis=1)
                        progress_bar.progress(100)
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame({
                            'original_log_line': log_lines,
                            'normalized_log_line': normalized_log_lines,
                            'max_similarity_score': max_similarities,
                            'most_similar_positive_idx': most_similar_positive_idx,
                            'most_similar_positive_example': [positive_examples_df.iloc[idx]['Log'] for idx in most_similar_positive_idx]
                        })
                        
                        # Sort by similarity score
                        results_df = results_df.sort_values('max_similarity_score', ascending=False)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        st.markdown("""
                        <div style='text-align: center; margin: 2rem 0;'>
                            <h2 style='color: #f0e68c;'>📊 Analysis Results</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Log Lines", len(results_df))
                        with col2:
                            st.metric("High Confidence", len(results_df[results_df['max_similarity_score'] >= confidence_threshold]))
                        with col3:
                            st.metric("Mean Similarity", f"{results_df['max_similarity_score'].mean():.3f}")
                        with col4:
                            st.metric("Max Similarity", f"{results_df['max_similarity_score'].max():.3f}")
                        
                        # Top results
                        st.subheader(f"🎯 Top {top_n} Most Likely Failed Login Events")
                        
                        for idx, row in results_df.head(top_n).iterrows():
                            score = row['max_similarity_score']
                            
                            # Create angular container with color coding
                            if score >= confidence_threshold:
                                container_style = "background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(240, 230, 140, 0.1) 100%); border: 1px solid rgba(212, 175, 55, 0.5);"
                                score_color = "#d4af37"
                            else:
                                container_style = "background: rgba(139, 115, 85, 0.1); border: 1px solid rgba(139, 115, 85, 0.3);"
                                score_color = "#a68b5b"
                            
                            st.markdown(f"""
                            <div style='{container_style} padding: 1rem; margin: 0.5rem 0; clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 20px 100%, 0 calc(100% - 20px));'>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h4 style='color: {score_color}; margin: 0;'>Score: {score:.4f}</h4>
                                </div>
                                <p style='color: #e0e0e0; margin: 0.5rem 0 0 0; word-break: break-all;'>{row['original_log_line']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("📝 Details"):
                                st.write("**Normalized version:**")
                                st.text(row['normalized_log_line'])
                                st.write("**Most similar to:**")
                                st.text(row['most_similar_positive_example'])
                        
                        # Visualization with custom styling
                        st.subheader("📈 Similarity Score Distribution")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        fig.patch.set_facecolor('#2d2b27')
                        ax.set_facecolor('#3d3b37')
                        
                        # Create histogram with angular style
                        n, bins, patches = ax.hist(results_df['max_similarity_score'], bins=30, 
                                                  edgecolor='#d4af37', linewidth=1, alpha=0.8)
                        
                        # Color gradient for bars
                        cm = plt.cm.get_cmap('YlOrBr')
                        bin_centers = 0.5 * (bins[:-1] + bins[1:])
                        col = bin_centers - min(bin_centers)
                        col /= max(col)
                        
                        for c, p in zip(col, patches):
                            plt.setp(p, 'facecolor', cm(c))
                        
                        ax.axvline(x=confidence_threshold, color='#d4af37', linestyle='--', linewidth=2,
                                  label=f'Threshold: {confidence_threshold}')
                        ax.axvline(x=results_df['max_similarity_score'].mean(), color='#f0e68c', linestyle='--', linewidth=2,
                                  label=f'Mean: {results_df["max_similarity_score"].mean():.3f}')
                        
                        ax.set_xlabel('Similarity Score', color='#f0e68c', fontsize=12)
                        ax.set_ylabel('Frequency', color='#f0e68c', fontsize=12)
                        ax.set_title('Distribution of Similarity Scores', color='#f0e68c', fontsize=16, pad=20)
                        ax.tick_params(colors='#e0e0e0')
                        ax.spines['bottom'].set_color('#d4af37')
                        ax.spines['left'].set_color('#d4af37')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.legend(facecolor='#3d3b37', edgecolor='#d4af37', labelcolor='#f0e68c')
                        
                        st.pyplot(fig)
                        
                        # Threshold analysis
                        st.subheader("🎯 Threshold Analysis")
                        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        threshold_data = []
                        
                        for threshold in thresholds:
                            count = len(results_df[results_df['max_similarity_score'] >= threshold])
                            percentage = (count / len(results_df)) * 100
                            threshold_data.append({
                                'Threshold': threshold,
                                'Count': count,
                                'Percentage': f"{percentage:.1f}%"
                            })
                        
                        threshold_df = pd.DataFrame(threshold_data)
                        
                        # Style the dataframe
                        st.markdown("""
                        <style>
                        .dataframe {
                            background: rgba(255, 255, 255, 0.05);
                            color: #e0e0e0;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        st.dataframe(threshold_df, use_container_width=True)
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Full results CSV
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full Results (CSV)",
                                data=csv,
                                file_name="failed_login_analysis_results.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # High confidence results only
                            high_confidence_df = results_df[results_df['max_similarity_score'] >= confidence_threshold]
                            csv_high = high_confidence_df.to_csv(index=False)
                            st.download_button(
                                label=f"⭐ Download High Confidence Results (Score >= {confidence_threshold})",
                                data=csv_high,
                                file_name=f"high_confidence_failed_logins_{confidence_threshold}.csv",
                                mime="text/csv"
                            )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #a68b5b; padding: 2rem 0;'>
        <h3 style='color: #f0e68c;'>How It Works</h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;'>
            <div style='background: rgba(212, 175, 55, 0.1); padding: 1rem; clip-path: polygon(0 0, calc(100% - 15px) 0, 100% 15px, 100% 100%, 0% 100%);'>
                <h4 style='color: #d4af37;'>1. Upload</h4>
                <p>Upload your log file in any common format</p>
            </div>
            <div style='background: rgba(240, 230, 140, 0.1); padding: 1rem; clip-path: polygon(0 0, 100% 0, 100% 100%, 15px 100%, 0 calc(100% - 15px));'>
                <h4 style='color: #f0e68c;'>2. Process</h4>
                <p>AI normalizes and analyzes text patterns</p>
            </div>
            <div style='background: rgba(139, 115, 85, 0.1); padding: 1rem; clip-path: polygon(15px 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0% 100%, 0% 15px);'>
                <h4 style='color: #a68b5b;'>3. Analyze</h4>
                <p>Semantic similarity with known patterns</p>
            </div>
            <div style='background: rgba(166, 139, 91, 0.1); padding: 1rem; clip-path: polygon(0 15px, calc(100% - 15px) 0, 100% 0, 100% 100%, 0 100%);'>
                <h4 style='color: #8b7355;'>4. Results</h4>
                <p>Ranked list of potential failed logins</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

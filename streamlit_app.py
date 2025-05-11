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
    page_title="Failed Login Detector",
    page_icon="🔐",
    layout="wide"
)

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

# Load positive examples (you'll need to have this file in your app directory)
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
    st.title("🔐 Failed Login Detector")
    st.markdown("""
    This app uses semantic similarity to identify potential failed login events in log files.
    Upload your log file and we'll rank each line by its similarity to known failed login patterns.
    """)
    
    # Load positive examples
    positive_examples_df = load_positive_examples()
    if positive_examples_df is None:
        return
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6,
        step=0.05,
        help="Log lines with similarity scores above this threshold will be highlighted"
    )
    
    top_n = st.sidebar.number_input(
        "Number of top results to display",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    
    # File upload
    st.header("📁 Upload Log File")
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
            
            st.success(f"Successfully loaded {len(log_lines)} log lines")
            
            # Show sample of uploaded file
            with st.expander("View first 5 log lines"):
                for i, line in enumerate(log_lines[:5]):
                    st.text(f"{i+1}: {line}")
            
            # Process button
            if st.button("🔍 Analyze Log File", type="primary"):
                with st.spinner("Analyzing log file..."):
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
                    st.header("📊 Analysis Results")
                    
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
                    st.subheader(f"Top {top_n} Most Likely Failed Login Events")
                    
                    for idx, row in results_df.head(top_n).iterrows():
                        score = row['max_similarity_score']
                        color = "red" if score >= confidence_threshold else "orange"
                        
                        with st.container():
                            col1, col2 = st.columns([1, 5])
                            with col1:
                                st.markdown(f"**Score:** :{color}[{score:.4f}]")
                            with col2:
                                st.text(row['original_log_line'])
                                with st.expander("Details"):
                                    st.write("**Normalized version:**")
                                    st.text(row['normalized_log_line'])
                                    st.write("**Most similar to:**")
                                    st.text(row['most_similar_positive_example'])
                            st.divider()
                    
                    # Visualization
                    st.subheader("📈 Similarity Score Distribution")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(results_df['max_similarity_score'], bins=50, edgecolor='black', alpha=0.7)
                    ax.axvline(x=confidence_threshold, color='red', linestyle='--', 
                              label=f'Threshold: {confidence_threshold}')
                    ax.axvline(x=results_df['max_similarity_score'].mean(), color='green', linestyle='--', 
                              label=f'Mean: {results_df["max_similarity_score"].mean():.3f}')
                    ax.set_xlabel('Similarity Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Similarity Scores')
                    ax.legend()
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
                    st.dataframe(threshold_df, use_container_width=True)
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Full results CSV
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Full Results (CSV)",
                            data=csv,
                            file_name="failed_login_analysis_results.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # High confidence results only
                        high_confidence_df = results_df[results_df['max_similarity_score'] >= confidence_threshold]
                        csv_high = high_confidence_df.to_csv(index=False)
                        st.download_button(
                            label=f"Download High Confidence Results (Score >= {confidence_threshold})",
                            data=csv_high,
                            file_name=f"high_confidence_failed_logins_{confidence_threshold}.csv",
                            mime="text/csv"
                        )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Upload your log file
    2. The app normalizes text by removing timestamps, IPs, and numbers
    3. Uses Sentence-BERT to compute semantic embeddings
    4. Calculates cosine similarity with known failed login patterns
    5. Ranks log lines by similarity score
    """)

if __name__ == "__main__":
    main()

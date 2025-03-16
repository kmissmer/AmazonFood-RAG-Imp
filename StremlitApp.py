import os
import sys
import time
import logging
import json
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Import your existing RAG system
from RAG import AmazonReviewsRAG, DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Amazon Reviews RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with golden theme
st.markdown("""
<style>
    /* Global theme colors */
    :root {
        --primary-gold: #D4AF37;
        --dark-gold: #8B6914;
        --light-gold: #F5D98B;
        --gold-gradient: linear-gradient(to right, #8B6914, #D4AF37, #FFD700);
        --text-on-gold: #1E1E1E;
        --background-color: #15131A;
        --text-color: #E6D299;
        --border-color: #D4AF37;
    }
    
    /* Main header with gold gradient */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-gold);
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Rating colors */
    .rating-positive {
        color: var(--dark-gold);
        font-weight: bold;
    }
    .rating-neutral {
        color: #A89968;
        font-weight: bold;
    }
    .rating-negative {
        color: #8B6914;
        font-weight: bold;
    }
    
    /* Headers */
    .source-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 0.5rem;
        color: var(--dark-gold);
    }
    
    /* Containers */
    .metadata {
        background-color: rgba(212, 175, 55, 0.1);
        border: 1px solid var(--border-color);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    .review-content {
        border-left: 4px solid var(--primary-gold);
        padding-left: 10px;
        margin-top: 10px;
        color: var(--text-color);
    }
    
    /* Ensure all text in review content is visible */
    .review-content p, .review-content span, .review-content div {
        color: var(--text-color);
    }
    
    /* Metadata text */
    .metadata p, .metadata span, .metadata strong {
        color: var(--text-color);
    }
    
    /* Streamlit element customization */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stButton button {
        background-color: var(--primary-gold);
        color: var(--background-color);
        border: none;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background-color: var(--light-gold);
        color: var(--background-color);
    }
    
    .stTextInput > div > div > input {
        border: 1px solid var(--border-color);
        background-color: rgba(255, 255, 255, 0.1);
        color: var(--text-color);
    }
    
    .stTextInput > div > div > input:focus {
        border: 1px solid var(--primary-gold);
        box-shadow: 0 0 5px var(--light-gold);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1A1821;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5 {
        color: var(--primary-gold);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1F1C25;
        border: 1px solid var(--border-color);
        color: var(--primary-gold);
    }
    
    div[data-testid="stExpander"] div[role="button"] > div {
        color: var(--primary-gold);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1A1821;
        border-bottom: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--dark-gold);
        color: var(--background-color);
    }
    
    /* General text */
    p, span, div {
        color: var(--text-color);
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: var(--primary-gold) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }
    
    /* Info blocks */
    .stAlert {
        background-color: rgba(212, 175, 55, 0.1);
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system(csv_path, model_id, limit):
    """Initialize the RAG system with caching"""
    try:
        rag_system = AmazonReviewsRAG(
            csv_path=csv_path,
            model_id=model_id,
            limit=limit
        ).initialize()
        return rag_system
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

def format_rating(rating):
    """Format rating with stars and sentiment"""
    if isinstance(rating, (int, float)):
        stars = "⭐" * int(rating) + ("✨" if rating % 1 > 0 else "")
        if rating >= 4.0:
            return f'<span class="rating-positive">{rating} {stars} (Positive)</span>'
        elif rating <= 2.0:
            return f'<span class="rating-negative">{rating} {stars} (Negative)</span>'
        else:
            return f'<span class="rating-neutral">{rating} {stars} (Neutral)</span>'
    return f"{rating}"

def display_document(doc, index):
    """Display a document with formatted metadata and content"""
    with st.expander(f"Source Review {index+1}", expanded=(index==0)):
        # Extract metadata
        rating = doc.metadata.get('rating', 'N/A')
        product_id = doc.metadata.get('product_id', 'N/A')
        date = doc.metadata.get('date', 'N/A')
        user_id = doc.metadata.get('user_id', 'N/A')
        
        # Create metadata section
        st.markdown('<div class="metadata">', unsafe_allow_html=True)
        st.markdown(f"**Rating:** {format_rating(rating)}", unsafe_allow_html=True)
        st.markdown(f"**Product ID:** {product_id}")
        st.markdown(f"**Date:** {date}")
        st.markdown(f"**User ID:** {user_id}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display review content
        st.markdown('<div class="review-content">', unsafe_allow_html=True)
        st.markdown(doc.page_content)
        st.markdown('</div>', unsafe_allow_html=True)

def display_dashboard(rag_system):
    """Display dataset summary and visualizations"""
    summary = rag_system.get_summary()
    
    # Summary metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{summary['total_reviews']:,}")
    col2.metric("Average Rating", f"{summary['average_rating']}/5 ⭐")
    col3.metric("Unique Products", f"{summary['total_unique_products']:,}")
    col4.metric("Unique Users", f"{summary['total_unique_users']:,}")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    rating_data = {
        "Rating Type": ["Positive (4-5★)", "Neutral (3★)", "Negative (1-2★)"],
        "Count": [
            summary['positive_reviews'],
            summary['neutral_reviews'],
            summary['negative_reviews']
        ]
    }
    rating_df = pd.DataFrame(rating_data)
    st.bar_chart(rating_df.set_index("Rating Type"))
    
    # Date info
    st.subheader("Date Range")
    st.info(f"Reviews from {summary['date_range']}")

def main():
    st.markdown('<h1 class="main-header">Amazon Reviews RAG System</h1>', unsafe_allow_html=True)
    st.markdown("""
    Ask questions about Amazon reviews to get insights powered by a Retrieval Augmented Generation system.
    This application uses vector search to find relevant reviews and a language model to generate answers.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        csv_path = st.text_input("Reviews CSV Path", "Reviews.csv")
        model_options = {
            "Flan T5 Base": "google/flan-t5-base",
            "Flan T5 Large": "google/flan-t5-large",
            "Flan T5 Small": "google/flan-t5-small"
        }
        model_selection = st.selectbox("Language Model", list(model_options.keys()))
        model_id = model_options[model_selection]
        limit = st.number_input("Number of Reviews to Load", min_value=100, max_value=100000, value=5000, step=1000)
        
        st.markdown("---")
        
        # No filters section
        st.markdown("---")
    
    # Initialize the RAG system
    with st.spinner("Initializing the RAG system... This may take a minute."):
        rag_system = initialize_rag_system(csv_path, model_id, limit)
    
    if rag_system is None:
        st.error("Failed to initialize the RAG system. Check logs for details.")
        return
    
    # Dashboard tab system
    tab1, tab2 = st.tabs(["Query Interface", "Dashboard"])
    
    with tab1:
        # Query interface
        query = st.text_input("Ask a question about Amazon reviews:", placeholder="e.g., What do people like most about this product?")
        
        if query:
            with st.spinner("Searching for relevant reviews and generating answer..."):
                # No filters applied
                result = rag_system.query(query)
                
                # Display answer
                st.markdown("### Answer")
                st.write(result['result'])
                
                # Display processing time
                st.caption(f"Processing time: {result['processing_time']}")
                
                # Source documents
                if result['source_documents']:
                    st.markdown("### Source Reviews")
                    st.markdown("The answer was generated based on these reviews:")
                    
                    for i, doc in enumerate(result['source_documents']):
                        display_document(doc, i)
                
                # Feedback section
                st.markdown("### Feedback")
                col1, col2, col3, col4, col5 = st.columns(5)
                feedback = None
                
                if col1.button("😟 1"):
                    feedback = 1
                elif col2.button("🙁 2"):
                    feedback = 2
                elif col3.button("😐 3"):
                    feedback = 3
                elif col4.button("🙂 4"):
                    feedback = 4
                elif col5.button("😃 5"):
                    feedback = 5
                
                if feedback:
                    rag_system.collect_feedback(query, result, feedback)
                    st.success(f"Thank you for your feedback! You rated this answer {feedback}/5.")
    
    with tab2:
        # Dashboard visualizations
        display_dashboard(rag_system)

if __name__ == "__main__":
    main()
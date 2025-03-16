# Amazon Reviews RAG System

## Overview

The Amazon Reviews RAG (Retrieval Augmented Generation) System is a tool for analyzing and extracting insights from Amazon food product reviews using state-of-the-art natural language processing techniques. It combines vector search with language models to provide intelligent answers to questions about food product reviews.

> **Note:** This project is currently under active development. New features and improvements will be added regularly.

## Features

- **Smart Review Analysis:** Ask natural language questions about food product reviews and get AI-powered answers
- **Vector Search:** Efficiently find the most relevant reviews using semantic search
- **Interactive Web Interface:** User-friendly Streamlit app with a sleek dark gold theme
- **Review Visualization:** See review data summarized with intuitive charts and metrics
- **Feedback Collection:** Rate answers to help improve the system over time

## Getting Started

### Prerequisites

- Python 3.8 or higher
- An Amazon food product reviews dataset in CSV format (see [Data Format]([#data-format](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)) below)

### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/amazon-reviews-rag.git
cd amazon-reviews-rag
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

3. Run the Streamlit web interface
```bash
streamlit run app.py
```

### Data Format

The system expects an Amazon food product reviews dataset with the following columns:
- `ProductId`: Unique identifier for the food product
- `UserId`: Unique identifier for the reviewer
- `Score`: Rating (1-5 stars)
- `Summary`: Brief review summary/title
- `Text`: Full review text
- `Time`: Review timestamp
- `HelpfulnessNumerator`: Number of users who found the review helpful
- `HelpfulnessDenominator`: Number of users who voted on helpfulness

## Architecture

The system uses a modular architecture with the following components:

- **DataProcessor:** Handles data loading and preprocessing
- **TextChunker:** Splits reviews into manageable chunks
- **VectorStoreManager:** Creates and manages the vector database
- **RetrievalSystem:** Retrieves relevant reviews using vector search
- **ModelManager:** Manages language model operations
- **FeedbackCollector:** Collects and stores user feedback

## Acknowledgments

- This project uses [LangChain](https://github.com/langchain-ai/langchain) for the RAG pipeline
- The web interface is built with [Streamlit](https://streamlit.io/)
- Language models are powered by [Hugging Face Transformers](https://huggingface.co/transformers/)

---

*This project is not affiliated with, endorsed by, or sponsored by Amazon.com, Inc.*

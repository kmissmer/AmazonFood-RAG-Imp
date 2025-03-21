# Amazon Food Reviews RAG System

## Overview

The Amazon Food Reviews RAG (Retrieval Augmented Generation) System is a tool for analyzing and extracting insights from Amazon food product reviews using state-of-the-art natural language processing techniques. It combines vector search with language models to provide intelligent answers to questions about food product reviews.

> **Note:** This project is currently under active development. New features and improvements will be added regularly.

## Features

- **Smart Review Analysis:** Ask questions about food product reviews and get AI-powered answers
- **Vector Search:** Efficiently find the most relevant reviews using semantic search
- **Interactive Web Interface:** User-friendly Streamlit app with a sleek dark gold theme
- **Review Visualization:** See review data summarized with intuitive charts and metrics
- **Feedback Collection:** Rate answers to help improve the system over time

## Getting Started

### Prerequisites

- Python 3.8 or higher
- The Amazon food product reviews dataset in CSV format (see [Link](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews))

### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/AmazonFood-RAG-Imp.git
cd AmazonFood-RAG-Imp
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

3. Run the Streamlit web interface
```bash
streamlit run StreamlitApp.py
```

### Known Issues and Configuration
If you encounter the PyTorch error `RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!` when running the Streamlit app, you'll need to disable Streamlit's file watcher. 

Create a `.streamlit/config.toml` file with:
```toml
[server]
fileWatcherType = "none"
```

Alternatively, you can run the app with:
```bash
streamlit run StreamlitApp.py --server.fileWatcherType=none
```

From my experience and testing, this error will likely not mess up the functionality of the program, but it is slightly annoying

The error is due to an interaction between PyTorch and Streamlit's file watcher functionality.




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

import os
import pandas as pd
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
from datetime import datetime

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 1. Load and prepare Amazon reviews dataset with preprocessing
def load_amazon_reviews(csv_path, limit=5000):
    """Load Amazon Fine Food Reviews dataset with preprocessing."""
    # Read CSV file - adjust dtypes for efficiency
    df = pd.read_csv(
        csv_path, 
        nrows=limit,
        parse_dates=['Time'],
        dtype={
            'ProductId': str,
            'UserId': str,
            'Score': float,
            'Summary': str,
            'Text': str
        }
    )
    
    # Basic cleaning
    df = df.fillna('')
    
    # Convert timestamp to readable date
    df['Date'] = df['Time'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    
    # Remove duplicate reviews (same user, same product, same text)
    df = df.drop_duplicates(subset=['UserId', 'ProductId', 'Text'])
    
    # Create helpfulness ratio
    df['HelpfulnessRatio'] = df.apply(
        lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] 
        if row['HelpfulnessDenominator'] > 0 else 0,
        axis=1
    )
    
    print(f"Loaded {len(df)} Amazon reviews after preprocessing")
    return df

# 2. Create feature-rich documents with enhanced metadata
def prepare_review_documents(df):
    """Convert DataFrame to documents with enhanced metadata."""
    documents = []
    
    for _, row in df.iterrows():
        # Create rich metadata for each review
        metadata = {
            "product_id": row["ProductId"],
            "user_id": row["UserId"],
            "rating": float(row["Score"]),
            "date": row["Date"],
            "helpful_ratio": float(row["HelpfulnessRatio"]),
            "summary": row["Summary"],
            "is_positive": float(row["Score"]) >= 4.0,
            "is_negative": float(row["Score"]) <= 2.0,
            "is_neutral": 2.0 < float(row["Score"]) < 4.0
        }
        
        # Create enhanced content with clear structure
        content = f"Review: {row['Summary']}\n\n"
        content += f"Rating: {row['Score']} out of 5 stars\n"
        content += f"Date: {row['Date']}\n"
        content += f"Product ID: {row['ProductId']}\n\n"
        content += f"Full Review:\n{row['Text']}"
        
        # Create document with text and metadata
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        
        documents.append(doc)
    
    return documents

# 3. Additional analysis for dataset insights
def analyze_dataset(df):
    """Analyze the dataset to extract insights for better RAG."""
    # 1. Rating distribution
    rating_counts = df['Score'].value_counts().sort_index()
    
    # 2. Top products by review count
    top_products = df['ProductId'].value_counts().head(10)
    
    # 3. Average rating by date (monthly)
    df['YearMonth'] = df['Date'].str[:7]  # Get YYYY-MM
    monthly_ratings = df.groupby('YearMonth')['Score'].mean()
    
    # 4. Helpful vs unhelpful reviews
    helpful_avg = df[df['HelpfulnessRatio'] > 0.7]['Score'].mean()
    unhelpful_avg = df[df['HelpfulnessRatio'] < 0.3]['Score'].mean()
    
    # 5. Review length analysis
    df['ReviewLength'] = df['Text'].str.len()
    length_correlation = np.corrcoef(df['ReviewLength'], df['Score'])[0,1]
    
    analysis_text = f"""
    Dataset Analysis:
    - Total reviews: {len(df)}
    - Rating distribution: {dict(rating_counts)}
    - Average rating: {df['Score'].mean():.2f}
    - Reviews with 5-stars: {len(df[df['Score'] == 5])} ({len(df[df['Score'] == 5])/len(df)*100:.1f}%)
    - Reviews with 1-star: {len(df[df['Score'] == 1])} ({len(df[df['Score'] == 1])/len(df)*100:.1f}%)
    - Most reviewed product has {top_products.iloc[0]} reviews
    - Average rating of helpful reviews (>70% helpful): {helpful_avg:.2f}
    - Average rating of unhelpful reviews (<30% helpful): {unhelpful_avg:.2f}
    - Correlation between review length and rating: {length_correlation:.2f}
    """
    
    print(analysis_text)
    return {
        "rating_counts": rating_counts,
        "top_products": top_products,
        "monthly_ratings": monthly_ratings,
        "helpful_avg": helpful_avg,
        "unhelpful_avg": unhelpful_avg,
        "length_correlation": length_correlation
    }

# 4. Create analysis documents for the knowledge base
def create_analysis_documents(df, analysis_results):
    """Create analysis documents to enhance the knowledge base."""
    analysis_docs = []
    
    # 1. General dataset overview
    overview = f"""
    Amazon Fine Food Reviews Dataset Overview:
    
    This dataset consists of {len(df)} food reviews from Amazon spanning multiple years.
    The average rating across all products is {df['Score'].mean():.2f} out of 5 stars.
    Most reviews ({len(df[df['Score'] >= 4])}/{len(df)}) are positive (4 or 5 stars).
    Negative reviews (1 or 2 stars) account for {len(df[df['Score'] <= 2])}/{len(df)} of the dataset.
    
    Users tend to leave more positive reviews than negative ones, with 5-star reviews being the most common ({len(df[df['Score'] == 5])}/{len(df)}).
    """
    
    doc = Document(
        page_content=overview,
        metadata={"doc_type": "analysis", "analysis_type": "overview"}
    )
    analysis_docs.append(doc)
    
    # 2. Rating patterns
    rating_patterns = f"""
    Rating Patterns in Amazon Food Reviews:
    
    The distribution of ratings shows a "J-shaped" pattern typical of online reviews:
    - 5-star ratings: {len(df[df['Score'] == 5])} reviews ({len(df[df['Score'] == 5])/len(df)*100:.1f}%)
    - 4-star ratings: {len(df[df['Score'] == 4])} reviews ({len(df[df['Score'] == 4])/len(df)*100:.1f}%)
    - 3-star ratings: {len(df[df['Score'] == 3])} reviews ({len(df[df['Score'] == 3])/len(df)*100:.1f}%)
    - 2-star ratings: {len(df[df['Score'] == 2])} reviews ({len(df[df['Score'] == 2])/len(df)*100:.1f}%)
    - 1-star ratings: {len(df[df['Score'] == 1])} reviews ({len(df[df['Score'] == 1])/len(df)*100:.1f}%)
    
    This indicates that users are more likely to leave reviews for products they either really loved or really hated, with fewer reviews in the middle range.
    """
    
    doc = Document(
        page_content=rating_patterns,
        metadata={"doc_type": "analysis", "analysis_type": "ratings"}
    )
    analysis_docs.append(doc)
    
    # 3. Helpfulness analysis
    helpfulness = f"""
    Review Helpfulness Analysis:
    
    Reviews marked as helpful by other users tend to have different characteristics than unhelpful reviews:
    
    - Helpful reviews (>70% helpful ratio) have an average rating of {analysis_results['helpful_avg']:.2f}
    - Unhelpful reviews (<30% helpful ratio) have an average rating of {analysis_results['unhelpful_avg']:.2f}
    
    This suggests that users find balanced, detailed reviews more helpful than extremely positive or negative ones.
    Helpful reviews tend to be more detailed, providing specific information about the product's taste, quality, value, and use cases.
    """
    
    doc = Document(
        page_content=helpfulness,
        metadata={"doc_type": "analysis", "analysis_type": "helpfulness"}
    )
    analysis_docs.append(doc)
    
    # 4. Popular products
    top_products = df['ProductId'].value_counts().head(5)
    top_products_text = "Most Reviewed Food Products:\n\n"
    
    for product_id, count in top_products.items():
        # Get average rating for this product
        avg_rating = df[df['ProductId'] == product_id]['Score'].mean()
        product_reviews = df[df['ProductId'] == product_id]
        
        # Get the most common summary words
        summaries = ' '.join(product_reviews['Summary'].tolist())
        
        top_products_text += f"Product ID: {product_id}\n"
        top_products_text += f"Number of reviews: {count}\n"
        top_products_text += f"Average rating: {avg_rating:.2f}/5\n"
        top_products_text += f"Positive reviews: {len(product_reviews[product_reviews['Score'] >= 4])}/{count} ({len(product_reviews[product_reviews['Score'] >= 4])/count*100:.1f}%)\n"
        top_products_text += f"Sample review summary: \"{product_reviews['Summary'].iloc[0]}\"\n\n"
    
    doc = Document(
        page_content=top_products_text,
        metadata={"doc_type": "analysis", "analysis_type": "popular_products"}
    )
    analysis_docs.append(doc)
    
    return analysis_docs

# 5. Process documents into chunks
def process_documents(review_docs, analysis_docs):
    """Process documents into smaller chunks if needed."""
    # Combine all documents
    all_docs = review_docs + analysis_docs
    
    # Create text splitter with different settings for reviews vs analysis
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_documents(all_docs)
    
    print(f"Split {len(all_docs)} documents into {len(chunks)} chunks")
    return chunks

# 6. Create Vector Store
def create_vector_store(chunks):
    """Create a vector store from document chunks."""
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./amazon_reviews_db"
    )
    
    # Persist to disk for future use
    vectorstore.persist()
    
    return vectorstore

# 7. Set up RAG Chain with custom prompt
def setup_rag_chain(vectorstore):
    """Set up the RAG retrieval and generation chain with a specialized prompt."""
    # Custom prompt for product reviews
    prompt_template = """You are a product review analyst specializing in food products. 
    Use the following review information to answer the question. 
    
    If you don't know the answer based on the provided reviews, say so clearly - don't make up information.
    
    {context}
    
    Question: {question}
    
    When discussing reviews, always mention:
    - The overall sentiment (positive/negative)
    - The specific aspects mentioned (taste, quality, etc.)
    - Any patterns across multiple reviews when available
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Create retriever with metadata filtering capabilities
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,  # Retrieve more documents for better context
            "fetch_k": 20  # Fetch more documents before filtering
        }
    )
    
    # Create LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# 8. Enhanced query function with filtering capabilities
def query_review_rag(qa_chain, query, filters=None):
    """
    Query the Amazon reviews RAG system with filtering options.
    
    Parameters:
    - query: The user's question
    - filters: Optional dict with metadata filters like:
      - rating_min: Minimum rating (1-5)
      - rating_max: Maximum rating (1-5)
      - is_positive: Boolean to filter positive reviews
      - is_negative: Boolean to filter negative reviews
    """
    # Apply metadata filters if provided
    metadata = {}
    if filters:
        # Add metadata filters
        if 'is_positive' in filters and filters['is_positive']:
            metadata['is_positive'] = True
        if 'is_negative' in filters and filters['is_negative']:
            metadata['is_negative'] = True
    
    # Process the query
    if metadata:
        print(f"Applying filters: {metadata}")
        result = qa_chain({"query": query, "metadata": metadata})
    else:
        result = qa_chain({"query": query})
    
    print(f"\nQuestion: {query}")
    print(f"\nAnswer: {result['result']}")
    
    # Display sources with product and review info
    print("\nSources:")
    for i, doc in enumerate(result["source_documents"]):
        meta = doc.metadata
        
        # Check if it's an analysis document or a review
        if 'doc_type' in meta and meta['doc_type'] == 'analysis':
            print(f"\nSource {i+1} (Analysis Document):")
            print(f"Analysis Type: {meta['analysis_type']}")
        else:
            # Regular review
            print(f"\nSource {i+1} (Review):")
            if 'product_id' in meta:
                print(f"Product ID: {meta['product_id']}")
            if 'rating' in meta:
                print(f"Rating: {meta['rating']} stars")
            if 'date' in meta:
                print(f"Date: {meta['date']}")
            if 'summary' in meta:
                print(f"Summary: {meta['summary']}")
        
        # Preview content
        content_preview = doc.page_content.replace('\n', ' ')
        print(f"Content: {content_preview[:150]}...")
    
    return result

# 9. Main function with enhanced options
def main():
    # Path to the CSV file
    csv_path = "Reviews.csv"  # Update this to your file path
    
    # Load and process data
    print("Loading Amazon Fine Food Reviews dataset...")
    df = load_amazon_reviews(csv_path, limit=5000)  # Adjust limit based on your system
    
    # Analyze dataset
    print("\nAnalyzing dataset...")
    analysis_results = analyze_dataset(df)
    
    # Prepare documents
    print("\nPreparing review documents...")
    review_docs = prepare_review_documents(df)
    
    print("\nCreating analysis documents...")
    analysis_docs = create_analysis_documents(df, analysis_results)
    
    print("\nProcessing all documents...")
    chunks = process_documents(review_docs, analysis_docs)
    
    print("\nCreating vector store...")
    vectorstore = create_vector_store(chunks)
    
    # Set up RAG chain
    print("\nSetting up RAG chain...")
    qa_chain = setup_rag_chain(vectorstore)
    
    # Interactive querying with filter options
    print("\n==== Amazon Fine Food Reviews RAG System ====")
    print("Type 'exit' to quit.")
    print("Type 'help' for filter options.")
    
    while True:
        query = input("\nEnter your question about the reviews: ")
        
        if query.lower() == 'exit':
            break
            
        if query.lower() == 'help':
            print("\nFilter Options:")
            print("- To see only positive reviews: Add 'filter:positive' to your query")
            print("- To see only negative reviews: Add 'filter:negative' to your query")
            print("- Examples:")
            print("  'What do people say about taste? filter:positive'")
            print("  'What are common complaints? filter:negative'")
            continue
        
        # Check for filter commands
        filters = {}
        if 'filter:positive' in query.lower():
            filters['is_positive'] = True
            query = query.lower().replace('filter:positive', '').strip()
        elif 'filter:negative' in query.lower():
            filters['is_negative'] = True
            query = query.lower().replace('filter:negative', '').strip()
        
        # Execute query
        query_review_rag(qa_chain, query, filters)

if __name__ == "__main__":
    main()
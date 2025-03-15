import os
import sys
import time
import random
import logging
import json
from typing import Dict, List, Optional, Any
from functools import lru_cache
from datetime import datetime

import pandas as pd
import numpy as np

# Langchain and HuggingFace imports
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AmazonReviewsRAG:
    def __init__(
        self, 
        csv_path: str, 
        model_id: str = "google/flan-t5-base", 
        limit: int = 5000,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cache_size: int = 100
    ):
        """
        Initialize the Amazon Reviews RAG system.
        
        Args:
            csv_path (str): Path to the Amazon reviews CSV file
            model_id (str): HuggingFace model ID
            limit (int): Maximum number of reviews to process
            chunk_size (int): Size of text chunks for embedding
            chunk_overlap (int): Overlap between text chunks
            cache_size (int): Size of query cache
        """
        # Load environment variables
        load_dotenv()
        
        # Configuration
        self.csv_path = csv_path
        self.model_id = model_id
        self.limit = limit
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_size = cache_size
        
        # Initialize components
        self.df = None
        self.review_docs = None
        self.chunks = None
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.llm = None
        self.qa_chain = None
        
        # Default prompt template
        self.default_prompt_template = """Analyze the following reviews to answer the question comprehensively (more than just yes or no).

Context from reviews:
{context}

Question: {question}

Provide a detailed, insightful answer based on the review context."""
        
        # Feedback storage
        self.feedback_file = "feedback_log.jsonl"
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input parameters and file existence."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Validate model and configuration
        if not self.model_id:
            raise ValueError("A valid HuggingFace model ID must be provided")
    
    def load_reviews(self) -> pd.DataFrame:
        """
        Load and preprocess Amazon reviews.
        
        Returns:
            pd.DataFrame: Preprocessed reviews DataFrame
        """
        logger.info(f"Loading reviews from {self.csv_path}")
        
        try:
            df = pd.read_csv(
                self.csv_path, 
                nrows=self.limit,
                dtype={
                    'ProductId': str,
                    'UserId': str,
                    'Score': float,
                    'Summary': str,
                    'Text': str,
                    'Time': str
                }
            )
            
            # Basic cleaning and preprocessing
            df = df.fillna('')
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            df['Date'] = df['Time'].apply(
                lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d') if pd.notna(x) else 'Unknown'
            )
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['UserId', 'ProductId', 'Text'])
            
            # Helpfulness ratio
            df['HelpfulnessRatio'] = df.apply(
                lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] 
                if row['HelpfulnessDenominator'] > 0 else 0,
                axis=1
            )
            
            logger.info(f"Loaded {len(df)} reviews after preprocessing")
            return df
        
        except Exception as e:
            logger.error(f"Error loading reviews: {e}")
            raise
    
    def prepare_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Convert DataFrame to LangChain documents.
        
        Args:
            df (pd.DataFrame): Preprocessed reviews DataFrame
        
        Returns:
            List[Document]: List of LangChain documents
        """
        logger.info("Preparing review documents")
        
        documents = []
        for _, row in df.iterrows():
            # Rich metadata
            metadata = {
                "product_id": row["ProductId"],
                "user_id": row["UserId"],
                "rating": float(row["Score"]),
                "date": row["Date"],
                "helpful_ratio": float(row["HelpfulnessRatio"]),
                "summary": row["Summary"],
                "is_positive": float(row["Score"]) >= 4.0,
                "is_negative": float(row["Score"]) <= 2.0,
                "is_neutral": 2.0 < float(row["Score"]) < 4.0,
                "review_length": len(row["Text"]),
                "product_category": self._extract_category(row["ProductId"]),
            }
            
            # Enhanced content structure
            content = f"""Review: {row['Summary']}

Rating: {row['Score']} out of 5 stars
Date: {row['Date']}
Product ID: {row['ProductId']}

Full Review:
{row['Text']}"""
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        logger.info(f"Created {len(documents)} document objects")
        return documents
    
    def _extract_category(self, product_id):
        """Extract product category from ID if possible."""
        # This is a placeholder - in a real implementation, you might 
        # have a mapping of product IDs to categories
        return "Unknown"
    
    def hierarchical_chunking(self, documents: List[Document]) -> List[Document]:
        """
        Implement hierarchical chunking strategy.
        
        Args:
            documents (List[Document]): Original documents
        
        Returns:
            List[Document]: Hierarchically chunked documents
        """
        logger.info("Performing hierarchical chunking")
        
        # Parent chunks (larger context)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        parent_chunks = parent_splitter.split_documents(documents)
        
        # Child chunks (more granular)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        all_chunks = []
        for i, parent in enumerate(parent_chunks):
            children = child_splitter.split_documents([parent])
            
            # Link children to parent for context preservation
            for child in children:
                # Copy parent metadata
                for key, value in parent.metadata.items():
                    if key not in child.metadata:
                        child.metadata[key] = value
                
                # Add parent relationship
                child.metadata["parent_id"] = i
                child.metadata["parent_content"] = parent.page_content[:200] + "..."
                all_chunks.append(child)
        
        logger.info(f"Created {len(all_chunks)} hierarchical chunks from {len(documents)} documents")
        return all_chunks
    
    def create_vector_store(self, chunks: List[Document]):
        """
        Create vector store with embeddings.
        
        Args:
            chunks (List[Document]): Chunked documents
        
        Returns:
            Chroma: Vector store
        """
        logger.info("Creating vector store with embeddings")
        
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "hkunlp/instructor-base"
        ]
        
        for model_name in embedding_models:
            try:
                logger.info(f"Trying embedding model: {model_name}")
                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./amazon_reviews_db"
                )
                
                logger.info("Vector store created successfully")
                return vectorstore
            
            except Exception as e:
                logger.warning(f"Failed with {model_name}: {e}")
        
        raise RuntimeError("Could not create vector store with any available method")
    
    def setup_hybrid_retrieval(self, chunks: List[Document]):
        """
        Set up hybrid retrieval with vector store and BM25.
        
        Args:
            chunks (List[Document]): Document chunks
        """
        logger.info("Setting up hybrid retrieval system")
        
        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 5
        
        # Vector retriever
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        logger.info("Hybrid retrieval system configured")
    
    def setup_huggingface_model(self, max_retries: int = 3):
        """
        Initialize HuggingFace model with robust error handling.
        """
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        
        api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN not found. "
                "Get your token from https://huggingface.co/settings/tokens"
            )
        
        # Fallback models
        fallback_models = [
            self.model_id,
            "google/flan-t5-small",
            "google/flan-t5-base",
            "facebook/bart-large-cnn"  # More reliable alternative
        ]
        
        for attempt, current_model in enumerate(fallback_models):
            try:
                logger.info(f"Attempt {attempt+1} - Using model: {current_model}")
                
                # Exponential backoff with jitter
                if attempt > 0:
                    backoff_time = (2 ** attempt) + (random.random() * 2)
                    logger.info(f"Backing off for {backoff_time:.2f} seconds")
                    time.sleep(backoff_time)
                
                # Use HuggingFacePipeline instead of HuggingFaceEndpoint
                tokenizer = AutoTokenizer.from_pretrained(current_model)
                model = AutoModelForSeq2SeqLM.from_pretrained(current_model)
                
                # Create the pipeline with controlled parameters
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=120,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.95,
                    top_k=10,
                    repetition_penalty=1.03
                )
                
                # Create LangChain wrapper
                llm = HuggingFacePipeline(pipeline=pipe)
                
                # Test the model
                test_response = llm.invoke("Summarize this review: Great product, very helpful!")
                
                if test_response and len(test_response) > 10:
                    logger.info(f"✅ Model {current_model} successfully initialized")
                    return llm
                
            except Exception as e:
                logger.warning(f"Model {current_model} failed: {e}")
        
        raise RuntimeError("Failed to initialize any HuggingFace model")
    
    def rewrite_query(self, original_query: str) -> str:
        """
        Expand the original query to improve retrieval.
        
        Args:
            original_query (str): User's original query
        
        Returns:
            str: Rewritten query
        """
        logger.info(f"Rewriting query: {original_query}")
        
        # Simple rule-based query expansion
        if len(original_query) < 10:
            # For very short queries, use the model to expand
            prompt = f"Rewrite this search query to improve retrieval from product reviews: '{original_query}'"
            try:
                rewritten = self.llm.invoke(prompt)
                logger.info(f"Query rewritten to: {rewritten}")
                return rewritten
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}")
                return original_query
        
        return original_query
    
    def get_prompt_template(self, query: str) -> str:
        """
        Select appropriate prompt template based on query type.
        
        Args:
            query (str): User query
        
        Returns:
            str: Prompt template
        """
        query_lower = query.lower()
        
        # Comparison queries
        if any(term in query_lower for term in ["compare", "difference", "versus", "vs", "better"]):
            return """Compare the following product reviews to answer the question:
            
Context from reviews:
{context}

Question: {question}

Provide a balanced comparison highlighting the key differences based on these reviews."""
        
        # Summary queries
        elif any(term in query_lower for term in ["summary", "overview", "brief", "summarize"]):
            return """Summarize the key points from these reviews:
            
Context from reviews:
{context}

Question: {question}

Provide a concise summary of the main trends and opinions."""
        
        # Specific feature queries
        elif any(term in query_lower for term in ["feature", "function", "capability", "work", "performs"]):
            return """Analyze how specific features are discussed in these reviews:
            
Context from reviews:
{context}

Question: {question}

Focus on the functionality and performance aspects mentioned in the reviews."""
        
        # Default template
        return self.default_prompt_template
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Document]) -> List[tuple]:
        """
        Score relevance of retrieved documents.
        
        Args:
            query (str): User query
            retrieved_docs (List[Document]): Retrieved documents
        
        Returns:
            List[tuple]: Documents with relevance scores
        """
        logger.info(f"Evaluating relevance of {len(retrieved_docs)} retrieved documents")
        
        # Simple heuristic scoring based on keyword overlap
        query_terms = set(query.lower().split())
        scored_docs = []
        
        for doc in retrieved_docs:
            content = doc.page_content.lower()
            
            # Count query term occurrences
            term_matches = sum(1 for term in query_terms if term in content)
            
            # Calculate relevance score (0-10)
            # More sophisticated in practice, but this is a simple heuristic
            if len(query_terms) > 0:
                match_ratio = term_matches / len(query_terms)
                base_score = match_ratio * 8  # Scale to 0-8
            else:
                base_score = 5
                
            # Bonus for metadata matches
            metadata = doc.metadata
            metadata_bonus = 0
            
            # Rating-specific queries
            if "positive" in query.lower() and metadata.get("is_positive", False):
                metadata_bonus += 1
            elif "negative" in query.lower() and metadata.get("is_negative", False):
                metadata_bonus += 1
                
            # Recency bonus
            if "recent" in query.lower() and metadata.get("date", "Unknown") >= "2020":
                metadata_bonus += 1
                
            final_score = min(base_score + metadata_bonus, 10)
            scored_docs.append((doc, final_score))
        
        # Sort by relevance score
        return sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    def collect_feedback(self, query: str, result: Dict, feedback_score: int):
        """
        Store user feedback for continuous improvement.
        
        Args:
            query (str): User query
            result (Dict): Query result
            feedback_score (int): User feedback score (1-5)
        """
        logger.info(f"Collecting feedback. Score: {feedback_score}/5")
        
        feedback_record = {
            "query": query,
            "result": result["result"],
            "feedback_score": feedback_score,
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to feedback log
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_record) + "\n")
    
    def initialize(self):
        """
        Complete initialization process.
        """
        self.df = self.load_reviews()
        self.review_docs = self.prepare_documents(self.df)
        self.chunks = self.hierarchical_chunking(self.review_docs)
        self.vectorstore = self.create_vector_store(self.chunks)
        self.setup_hybrid_retrieval(self.chunks)
        self.llm = self.setup_huggingface_model()
        
        logger.info("RAG system initialization complete")
        return self
    
    @lru_cache(maxsize=100)
    def cached_query(self, query_str: str, filters_str: Optional[str] = None):
        """
        Cache query results for performance.
        
        Args:
            query_str (str): User query
            filters_str (str): JSON string of filters
        
        Returns:
            Dict: Query results
        """
        # Convert filters_str back to dict if not None
        filters = json.loads(filters_str) if filters_str else None
        return self.multi_stage_query(query_str, filters)
    
    def query(self, query: str, filters: Optional[Dict] = None):
        """
        User-facing query method with caching.
        
        Args:
            query (str): User query
            filters (Optional[Dict]): Optional filters
            
        Returns:
            Dict: Query results
        """
        # For caching to work, convert filters to a string
        filters_str = json.dumps(filters) if filters else None
        return self.cached_query(query, filters_str)
    
    def multi_stage_query(self, query: str, filters: Optional[Dict] = None):
        """
        Multi-stage retrieval and generation pipeline.
        
        Args:
            query (str): User query
            filters (Optional[Dict]): Optional filters
            
        Returns:
            Dict: Query results with answer and source documents
        """
        logger.info(f"Processing query through multi-stage pipeline: '{query}'")
        start_time = time.time()
        
        try:
            # Stage 1: Query rewriting and expansion
            rewritten_query = self.rewrite_query(query)
            
            # Stage 2: Hybrid retrieval
            if filters:
                initial_docs = self.vectorstore.similarity_search(
                    rewritten_query, k=8, filter=filters
                )
            else:
                initial_docs = self.ensemble_retriever.invoke(rewritten_query)

            
            logger.info(f"Retrieved {len(initial_docs)} initial documents")
            
            # Stage 3: Re-ranking based on relevance
            ranked_docs = self.evaluate_retrieval(query, initial_docs)
            
            # Filter to high-relevance documents (score > 5)
            top_docs = [doc for doc, score in ranked_docs if score > 5][:4]
            
            if not top_docs and ranked_docs:
                # Fallback to best available if nothing scores > 5
                top_docs = [ranked_docs[0][0]]
            
            logger.info(f"Selected {len(top_docs)} most relevant documents")
            
            # Stage 4: Generate response with dynamically selected prompt
            context = "\n\n".join([doc.page_content for doc in top_docs])
            
            prompt_template = self.get_prompt_template(query)
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | self.llm
            answer = chain.invoke({"context": context, "question": query})

            
            query_time = time.time() - start_time
            logger.info(f"Query processed in {query_time:.2f}s")
            
            return {
                "result": answer,
                "source_documents": top_docs,
                "processing_time": f"{query_time:.2f}s"
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                "result": f"Error processing query: {e}",
                "source_documents": [],
                "error": str(e)
            }
    
    def get_summary(self):
        """
        Generate a summary of the loaded dataset.
        
        Returns:
            Dict: Dataset summary statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call initialize() first.")
        
        return {
            "total_reviews": len(self.df),
            "average_rating": round(self.df['Score'].mean(), 2),
            "positive_reviews": len(self.df[self.df['Score'] >= 4]),
            "negative_reviews": len(self.df[self.df['Score'] <= 2]),
            "neutral_reviews": len(self.df[(self.df['Score'] > 2) & (self.df['Score'] < 4)]),
            "date_range": f"{self.df['Date'].min()} to {self.df['Date'].max()}",
            "total_unique_products": self.df['ProductId'].nunique(),
            "total_unique_users": self.df['UserId'].nunique()
        }


def main():
    """   
    Main execution function for Amazon Reviews RAG.
    """
    # Default configurations
    csv_path = "Reviews.csv"
    model_id = "google/flan-t5-base"
    limit = 5000
    
    # Allow command-line configuration
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    if len(sys.argv) > 2:
        csv_path = sys.argv[2]
    if len(sys.argv) > 3:
        limit = int(sys.argv[3])
    
    try:
        # Initialize the RAG system
        rag_system = AmazonReviewsRAG(
            csv_path=csv_path, 
            model_id=model_id, 
            limit=limit
        ).initialize()
        
        # Print dataset summary
        summary = rag_system.get_summary()
        print("\n===== Dataset Summary =====")
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("===========================\n")
        
        # Interactive query loop
        while True:
            try:
                query = input("Enter your question (or 'exit' to quit): ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                
                # Process query
                result = rag_system.query(query)
                print("\n=== Answer ===")
                print(result['result'])
                
                print("\n=== Top Relevant Source Review ===")
                if result['source_documents']:
                    doc = result['source_documents'][0]
                    
                    # Extract metadata
                    rating = doc.metadata.get('rating', 'N/A')
                    product_id = doc.metadata.get('product_id', 'N/A')
                    date = doc.metadata.get('date', 'N/A')
                    
                    # Format rating
                    if isinstance(rating, (int, float)):
                        if rating >= 4.0:
                            rating_text = f'{rating} ⭐ (Positive)'
                        elif rating <= 2.0:
                            rating_text = f'{rating} ⭐ (Negative)'
                        else:
                            rating_text = f'{rating} ⭐ (Neutral)'
                    else:
                        rating_text = str(rating)
                        
                    print(f"Rating: {rating_text}")
                    print(f"Product ID: {product_id}")
                    print(f"Date: {date}")
                    
                    # Extract and display review content
                    content = doc.page_content
                    print("\nReview Content:")
                    print(content[:500] + "..." if len(content) > 500 else content)
                    print("-" * 50)
                
                # Collect feedback
                try:
                    feedback = int(input("\nRate this answer (1-5, or press Enter to skip): ") or "0")
                    if 1 <= feedback <= 5:
                        rag_system.collect_feedback(query, result, feedback)
                except ValueError:
                    pass
            
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            except Exception as e:
                print(f"An error occurred during query processing: {e}")
    
    except Exception as e:
        print(f"An error occurred during initialization: {e}")

if __name__ == "__main__":
    main()
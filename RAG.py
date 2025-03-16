import os
import sys
import time
import logging
import json
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

#Langchain Stuff
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading and preprocessing operations"""
    
    @staticmethod
    def load_reviews(csv_path: str, limit: int) -> pd.DataFrame:
        """
        Load and preprocess Amazon reviews from CSV.
        
        Args:
            csv_path: Path to the CSV file
            limit: Maximum number of reviews to load
            
        Returns:
            Preprocessed DataFrame of reviews
        """
        logger.info(f"Loading reviews from {csv_path}")
        
        try:
            df = pd.read_csv(
                csv_path, 
                nrows=limit,
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
    
    @staticmethod
    def convert_to_documents(df: pd.DataFrame) -> List[Document]:
        """
        Convert DataFrame to LangChain documents.
        
        Args:
            df: Preprocessed reviews DataFrame
        
        Returns:
            List of LangChain documents with metadata
        """
        logger.info("Converting DataFrame to LangChain documents")
        
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
                "product_category": DataProcessor._extract_category(row["ProductId"]),
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


class TextChunker:
    """Handles document chunking for efficient retrieval"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunker with configurable parameters.
        
        Args:
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between text chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Simple document chunking with fixed size and overlap.
        
        Args:
            documents: Original documents
        
        Returns:
            Chunked documents
        """
        logger.info("Performing basic document chunking")
        
        # Simple text splitter with standard parameters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        chunked_docs = splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs


class VectorStoreManager:
    """Manages vector embedding and storage operations"""
    
    @staticmethod
    def create_vector_store(chunks: List[Document], persist_dir: str = "./amazon_reviews_db") -> Chroma:
        """
        Create vector store with single embedding model.
        
        Args:
            chunks: Chunked documents
            persist_dir: Directory to persist vector store
            
        Returns:
            Initialized vector store
        """
        logger.info("Creating vector store with embeddings")
        
        # reliable embedding model
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        try:
            logger.info(f"Using embedding model: {embedding_model}")
            
            # Configure embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"}
            )
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            
            logger.info("Vector store created successfully")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise


class RetrievalSystem:
    """Manages vector-only document retrieval operations"""
    
    def __init__(self, vectorstore: Chroma, chunks: List[Document]):
        """
        Initialize vector-only retrieval system.
        
        Args:
            vectorstore: Vector database for semantic search
            chunks: Document chunks
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        logger.info("Vector-only retrieval system initialized")
    
    def retrieve_documents(self, query: str, filters: Optional[Dict] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query using only vector search.
        
        Args:
            query: User query
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        return self.vectorstore.similarity_search(query, k=8, filter=filters)
    
    def evaluate_relevance(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Score relevance of retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Documents with relevance scores
        """
        logger.info(f"Evaluating relevance of {len(documents)} retrieved documents")
        
        query_terms = set(query.lower().split())
        scored_docs = []
        
        for doc in documents:
            content = doc.page_content.lower()
            
            # Count query term occurrences
            term_matches = sum(1 for term in query_terms if term in content)
            
            # Calculate relevance score (0-10)
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


class ModelManager:
    """Manages language model initialization and operations"""
    
    def __init__(self, model_id: str = "google/flan-t5-base"):
        """
        Initialize model manager.
        
        Args:
            model_id: HuggingFace model ID
        """
        self.model_id = model_id
        self.llm = None
        
    def initialize_model(self):
        """
        Simple model initialization without fallbacks.
        
        Returns:
            Initialized language model
        """
        try:
            # HuggingFacePipeline
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            
            # Create the pipeline
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=120                                       
                )
            
            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Model {self.model_id} initialized")
            return self.llm
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def rewrite_query(self, query: str) -> str:
        """
        Expand the original query to improve retrieval.
        
        Args:
            query: User's original query
            
        Returns:
            Rewritten query
        """
        return query
        
    def get_prompt_template(self, query: str) -> str:
        """
        Select appropriate prompt template based on query type.
        
        Args:
            query: User query
            
        Returns:
            Prompt template
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
        return """Analyze the following reviews to answer the question comprehensively (more than just yes or no).

        Context from reviews:
        {context}

        Question: {question}

        Provide a detailed, insightful answer based on the review context."""


class FeedbackCollector:
    """Collects and stores user feedback"""
    
    def __init__(self, feedback_file: str = "feedback_log.jsonl"):
        """
        Initialize feedback collector.
        
        Args:
            feedback_file: Path to feedback log file
        """
        self.feedback_file = feedback_file
    
    def collect_feedback(self, query: str, result: Dict, feedback_score: int):
        """
        Store user feedback for continuous improvement.
        
        Args:
            query: User query
            result: Query result
            feedback_score: User feedback score (1-5)
        """
        logger.info(f"Collecting feedback. Score: {feedback_score}/5")
        
        feedback_record = {
            "query": query,
            "result": result["result"],
            "feedback_score": feedback_score,
            "timestamp": datetime.now().isoformat()
        }
        
        # Ensure directory exists
        Path(self.feedback_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Append to feedback log
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_record) + "\n")


class AmazonReviewsRAG:
    """Main class for the Amazon Reviews RAG system"""
    
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
            csv_path: Path to the Amazon reviews CSV file
            model_id: HuggingFace model ID
            limit: Maximum number of reviews to process
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between text chunks
            cache_size: Size of query cache
        """
        
        # Configuration
        self.csv_path = csv_path
        self.model_id = model_id
        self.limit = limit
        self.cache_size = cache_size
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.vector_store_manager = VectorStoreManager()
        self.model_manager = ModelManager(model_id)
        self.feedback_collector = FeedbackCollector()
        
        # Initialize data structures
        self.df = None
        self.review_docs = None
        self.chunks = None
        self.vectorstore = None
        self.retrieval_system = None
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input parameters and file existence"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Validate model and configuration
        if not self.model_id:
            raise ValueError("A valid HuggingFace model ID must be provided")
    
    def initialize(self):
        """Complete initialization process"""
        # Load and process data
        self.df = self.data_processor.load_reviews(self.csv_path, self.limit)
        self.review_docs = self.data_processor.convert_to_documents(self.df)
        self.chunks = self.chunker.chunk_documents(self.review_docs)
        
        # Initialize vector store and retrieval system
        self.vectorstore = self.vector_store_manager.create_vector_store(self.chunks)
        self.retrieval_system = RetrievalSystem(self.vectorstore, self.chunks)
        
        # Initialize language model
        self.model_manager.initialize_model()
        
        logger.info("RAG system initialization complete")
        return self
    
    @lru_cache(maxsize=100)
    def cached_query(self, query_str: str, filters_str: Optional[str] = None):
        """
        Cache query results for performance.
        
        Args:
            query_str: User query
            filters_str: JSON string of filters
        
        Returns:
            Query results
        """
        # Convert filters_str back to dict if not None
        filters = json.loads(filters_str) if filters_str else None
        return self._process_query(query_str, filters)
    
    def query(self, query: str, filters: Optional[Dict] = None):
        """
        User-facing query method with caching.
        
        Args:
            query: User query
            filters: Optional filters
            
        Returns:
            Query results
        """
        # For caching to work, convert filters to a string
        filters_str = json.dumps(filters) if filters else None
        return self.cached_query(query, filters_str)
    
    def _process_query(self, query: str, filters: Optional[Dict] = None):
        """
        Vector-based retrieval and generation pipeline.
        
        Args:
            query: User query
            filters: Optional filters
            
        Returns:
            Query results with answer and source documents
        """
        logger.info(f"Processing query: '{query}'")
        start_time = time.time()
        
        try:
            # Stage 1: Direct vector retrieval
            initial_docs = self.retrieval_system.retrieve_documents(query, filters)
            logger.info(f"Retrieved {len(initial_docs)} documents using vector search")
            
            # Stage 2: Re-ranking based on relevance
            ranked_docs = self.retrieval_system.evaluate_relevance(query, initial_docs)
            
            # Filter to high-relevance documents
            top_docs = [doc for doc, score in ranked_docs if score > 5][:4]
            
            if not top_docs and ranked_docs:
                # Fallback to best available if nothing scores > 5
                top_docs = [ranked_docs[0][0]]
            
            logger.info(f"Selected {len(top_docs)} most relevant documents")
            
            # Stage 3: Generate response with dynamically selected prompt
            context = "\n\n".join([doc.page_content for doc in top_docs])
            
            prompt_template = self.model_manager.get_prompt_template(query)
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | self.model_manager.llm
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
    
    def collect_feedback(self, query: str, result: Dict, feedback_score: int):
        """Store user feedback"""
        self.feedback_collector.collect_feedback(query, result, feedback_score)
    
    def get_summary(self):
        """
        Generate a summary of the loaded dataset.
        
        Returns:
            Dataset summary statistics
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
    """Main execution function for Amazon Reviews RAG"""
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
                            rating_text = f'{rating} (Positive)'
                        elif rating <= 2.0:
                            rating_text = f'{rating} (Negative)'
                        else:
                            rating_text = f'{rating} (Neutral)'
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

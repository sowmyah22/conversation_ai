import logging
import numpy as np
import ollama
import time
import json
import sqlite3
import faiss
from typing import List, Dict, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
db_file = 'articles.db'
index_file = 'embeddings.index'
EMBEDDING_DIM = 768  # Dimension of nomic-embed-text embeddings

# Conversation history
conversation_history = []
MAX_HISTORY_LENGTH = 5  # Maximum number of recent exchanges to keep

# Initialize SQLite connection
def get_sqlite_connection():
    try:
        conn = sqlite3.connect(db_file)
        logger.info(f"Connected to SQLite database: {db_file}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to SQLite: {e}")
        return None

# Load Faiss index
def load_faiss_index():
    try:
        index = faiss.read_index(index_file)
        logger.info(f"Loaded Faiss index: {index_file}")
        return index
    except Exception as e:
        logger.error(f"Failed to load Faiss index: {e}")
        return None

# Function to generate query embedding using Ollama
def generate_query_embedding(query: str) -> Optional[np.ndarray]:
    """Generate embedding for a query using Ollama"""
    try:
        response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        embedding = np.array(response['embedding'], dtype=np.float32)
        if len(embedding) != EMBEDDING_DIM:
            logger.error(f"Generated embedding has incorrect dimension: {len(embedding)}")
            return None
        return embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return None

# Function to get intent-aware contextual query
def get_contextual_query(current_query: str) -> str:
    """Create an intent-aware contextual query by analyzing conversation history"""
    global conversation_history
    
    if not conversation_history:
        return current_query
    
    try:
        history_context = ""
        for i, (q, a) in enumerate(conversation_history[-MAX_HISTORY_LENGTH:]):
            history_context += f"User: {q}\nAssistant: {a}\n\n"
        
        prompt = f"""Given this conversation history and current query, please:
1. Identify the main intent and key entities in the current query
2. Determine if this query references previous conversation
3. Create an enhanced search query that captures the full intent

Conversation history:
{history_context}

Current query: {current_query}

Output only the enhanced search query that best captures the user's intent with any implicit references resolved.
"""
        
        response = ollama.generate(
            model='llama3:8b',
            prompt=prompt,
            options={
                'temperature': 0.1,
                'top_p': 0.9,
                'max_tokens': 200
            }
        )
        enhanced_query = response['response'].strip()
        
        logger.info(f"Generated intent-aware query: {enhanced_query[:100]}...")
        
        if not enhanced_query or len(enhanced_query) < 5:
            logger.warning("Intent extraction failed, using original query")
            return current_query
            
        return enhanced_query
        
    except Exception as e:
        logger.error(f"Error generating intent-aware query: {e}")
        context_parts = []
        for i, (q, _) in enumerate(conversation_history[-2:]):
            context_parts.append(f"Previous question: {q}")
        context_parts.append(f"Current question: {current_query}")
        return " ".join(context_parts)

# Query function with intent-aware search
def query_embeddings(query: str, use_context: bool = True, top_k: int = 10) -> List[Dict]:
    try:
        ollama.list()
        logger.info("Ollama server is running")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        return []

    start_time = time.time()
    
    # Get intent-aware query
    if use_context:
        contextual_query = get_contextual_query(query)
        logger.info(f"Using intent-aware query: {contextual_query[:100]}...")
    else:
        contextual_query = query
    
    # Generate query embedding
    query_embedding = generate_query_embedding(contextual_query)
    if query_embedding is None:
        logger.error("Failed to generate query embedding")
        return []
    
    # Load Faiss index
    index = load_faiss_index()
    if index is None:
        logger.error("No Faiss index available")
        return []
    
    # Normalize query embedding for cosine similarity
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    # Query Faiss
    try:
        distances, indices = index.search(query_embedding, top_k)
        logger.info(f"Found {len(indices[0])} results from Faiss")
    except Exception as e:
        logger.error(f"Error querying Faiss: {e}")
        return []
    
    # Get article metadata from SQLite
    conn = get_sqlite_connection()
    if conn is None:
        logger.error("No SQLite connection available")
        return []
    
    cursor = conn.cursor()
    articles = []
    
    try:
        for idx, distance in zip(indices[0], distances[0]):
            cursor.execute('SELECT * FROM articles WHERE id = ?', (idx + 1,))  # Faiss indices are 0-based, SQLite IDs are 1-based
            row = cursor.fetchone()
            if row:
                article = {
                    'id': row[0],
                    'title': row[1],
                    'url': row[2],
                    'full_text': row[3],
                    'publish_date': row[4],
                    'keyword': row[5],
                    'author': row[6],
                    'article_keywords': json.loads(row[7]),
                    'similarity': float(distance)  # Inner product (cosine similarity)
                }
                articles.append(article)
            else:
                logger.warning(f"No article found for ID {idx + 1}")
    except Exception as e:
        logger.error(f"Error querying SQLite: {e}")
    finally:
        conn.close()
    
    logger.info(f"Query processing took {time.time() - start_time:.2f} seconds")
    logger.info(f"Found relevant articles for query: {query}")
    return articles

# Enhanced response generation with citations
def generate_response(query: str, articles: List[Dict]) -> Tuple[str, List[Dict]]:
    try:
        start_time = time.time()
        
        context_parts = []
        sources = []
        
        for i, a in enumerate(articles, 1):
            title = a.get('title', 'Unknown Title')
            url = a.get('url', '#')
            text = a.get('full_text', '')[:1500]
            
            sources.append({
                'index': i,
                'title': title,
                'url': url
            })
            
            context_parts.append(f"Article {i}: {title}\nSource: {url}\n{text}")
        
        context = "\n\n".join(context_parts)
        
        history_context = ""
        if conversation_history:
            history_parts = []
            for q, a in conversation_history[-3:]:
                history_parts.append(f"User: {q}\nAssistant: {a}")
            history_context = "Previous conversation:\n" + "\n\n".join(history_parts) + "\n\n"
        
        prompt = f"""You are a helpful assistant that provides information based on news articles. 
When referencing information, include citation numbers [1], [2], etc. that correspond to the source articles.
Always reference your sources when providing facts. Always include URLs for your sources at the end of your response.

{history_context}
User: {query}

Here are relevant articles to help you answer:
{context}

Provide a helpful response with proper citations using [1], [2], etc. and include a "Sources:" section at the end with the article titles and URLs.
"""
        
        response = ollama.generate(
            model='llama3:8b',
            prompt=prompt,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 1500
            }
        )
        response_text = response['response'].strip()
        
        logger.info(f"Response generation took {time.time() - start_time:.2f} seconds")
        return response_text, sources
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        query_words = query.lower().split()
        
        fallback = f"I found some information that might help with your question about {' '.join(query_words[:3])}...\n\n"
        fallback += "Here are some topics I can share information about:\n"
        
        for i, a in enumerate(articles[:5], 1):
            title = a.get('title', 'Unknown Topic')
            url = a.get('url', '#')
            fallback += f"{i}. {title}\n"
        
        fallback += "\nSources:\n"
        for i, a in enumerate(articles[:5], 1):
            title = a.get('title', 'Unknown Topic')
            url = a.get('url', '#')
            fallback += f"[{i}] {title} - {url}\n"
            
        fallback += "\nWould you like to know more about any of these topics specifically?"
        return fallback, articles[:5]

# Ensure response has citations
def ensure_citations(response: str, sources: List[Dict]) -> str:
    """Make sure the response includes citations and a Sources section"""
    if not sources:
        return response
        
    if "Sources:" not in response:
        response += "\n\nSources:\n"
        for src in sources:
            response += f"[{src['index']}] {src['title']} - {src['url']}\n"
    
    return response

def main():
    global conversation_history
    
    print("\nContextual RAG News Search Assistant\n")
    
    # Check SQLite database
    conn = get_sqlite_connection()
    if conn is None:
        print("Failed to initialize SQLite. Exiting.")
        return
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM articles")
    article_count = cursor.fetchone()[0]
    conn.close()
    
    if article_count == 0:
        print("No articles found in the database. Please run import_articles_sqlite_faiss.py first.")
        return
    
    # Check Faiss index
    index = load_faiss_index()
    if index is None:
        print("Failed to load Faiss index. Exiting.")
        return
    
    print(f"Found {article_count} articles in the database.")
    print("Ready for your questions!\n")
    
    while True:
        query = input("\nWhat would you like to know? (type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using Article Search Assistant")
            break
            
        total_start_time = time.time()
        
        print("\nSearching for relevant information...")
        results = query_embeddings(query, use_context=True, top_k=10)
        
        if not results:
            print("I couldn't find any relevant articles for your question.")
            continue

        print("Analyzing results to provide you with the best answer...")
        response, sources = generate_response(query, results)
        
        response = ensure_citations(response, sources)
        
        print("\n" + "─" * 80)
        print(response)
        print("─" * 80)
        
        conversation_history.append((query, response))
        
        if len(conversation_history) > MAX_HISTORY_LENGTH + 2:
            conversation_history = conversation_history[-MAX_HISTORY_LENGTH - 2:]
        
        if logger.level <= logging.DEBUG:
            print(f"\n[Debug: Total processing time: {time.time() - total_start_time:.2f} seconds]")
            
        if query.lower() == "clear history":
            conversation_history = []
            print("Conversation history cleared.")

if __name__ == "__main__":
    main()
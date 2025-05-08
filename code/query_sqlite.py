import logging
import numpy as np
import ollama
import time
import json
import sqlite3
from typing import List, Dict, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
db_file = 'articles.db'
EMBEDDING_DIM = 768  # Dimension of nomic-embed-text embeddings

# In-memory data stores
articles_memory = []  # Stores all article data
embeddings_memory = None  # Stores all embeddings as numpy array

# Conversation history
conversation_history = []
MAX_HISTORY_LENGTH = 5  # Maximum number of recent exchanges to keep

def load_articles_to_memory():
    """Load all articles and embeddings into memory at startup"""
    global articles_memory, embeddings_memory
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Load all articles
        cursor.execute('SELECT id, title, url, full_text, publish_date, keyword, author, article_keywords, embedding FROM articles')
        rows = cursor.fetchall()
        
        # Prepare data structures
        articles_memory = []
        embeddings_list = []
        
        for row in rows:
            article = {
                'id': row[0],
                'title': row[1],
                'url': row[2],
                'full_text': row[3],
                'publish_date': row[4],
                'keyword': row[5],
                'author': row[6],
                'article_keywords': json.loads(row[7]) if row[7] else []
            }
            articles_memory.append(article)
            
            # Convert blob to numpy array
            embedding = np.frombuffer(row[8], dtype=np.float32)
            embeddings_list.append(embedding)
        
        embeddings_memory = np.array(embeddings_list)
        logger.info(f"Loaded {len(articles_memory)} articles into memory")
        
    except Exception as e:
        logger.error(f"Failed to load articles into memory: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

# Initialize in-memory data at startup
load_articles_to_memory()

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

def query_embeddings(query: str, use_context: bool = True, top_k: int = 10) -> List[Dict]:
    try:
        ollama.list()
        logger.info("Ollama server is running")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        return []

    start_time = time.time()
    
    if use_context:
        contextual_query = get_contextual_query(query)
        logger.info(f"Using intent-aware query: {contextual_query[:100]}...")
    else:
        contextual_query = query
    
    query_embedding = generate_query_embedding(contextual_query)
    if query_embedding is None:
        logger.error("Failed to generate query embedding")
        return []
    
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    if embeddings_memory is None or len(embeddings_memory) == 0:
        logger.error("No embeddings available in memory")
        return []
    
    embeddings_norm = embeddings_memory / np.linalg.norm(embeddings_memory, axis=1)[:, np.newaxis]
    similarities = np.dot(embeddings_norm, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []
    
    for idx in top_indices:
        article = articles_memory[idx].copy()
        article['similarity'] = float(similarities[idx])
        results.append(article)
    
    logger.info(f"Query processing took {time.time() - start_time:.2f} seconds")
    logger.info(f"Found {len(results)} relevant articles for query: {query}")
    
    return results

def generate_response(query: str, articles: List[Dict]) -> Tuple[str, List[Dict]]:
    try:
        start_time = time.time()
        context_parts = []
        sources = []
        
        for i, a in enumerate(articles, 1):
            title = a.get('title', 'no Title')
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
            title = a.get('title', 'No Topic')
            url = a.get('url', '#')
            fallback += f"{i}. {title}\n"
        
        fallback += "\nSources:\n"
        for i, a in enumerate(articles[:5], 1):
            title = a.get('title', 'No Topic')
            url = a.get('url', '#')
            fallback += f"[{i}] {title} - {url}\n"
            
        fallback += "\nWould you like to know more about any of these topics specifically?"
        return fallback, articles[:5]

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
    
    if not articles_memory:
        print("No articles loaded in memory. Please check database.")
        return
    
    
    while True:
        query = input("\nWhat would you like to know? (type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using News Search Assistant")
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
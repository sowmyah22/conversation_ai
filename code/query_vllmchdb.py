import logging
import numpy as np
import ollama
import time
import json
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from vllm import LLM, SamplingParams

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chroma settings
COLLECTION_NAME = "articles"
EMBEDDING_DIM = 768  # Dimension of nomic-embed-text embeddings

# Conversation history
conversation_history = []
MAX_HISTORY_LENGTH = 5  # Maximum number of recent exchanges to keep

# Initialize vLLM model
try:
    llm = LLM(model="meta-llama/Llama-3-8b", gpu_memory_utilization=0.9)
    logger.info("Initialized vLLM with Llama-3-8b model")
except Exception as e:
    logger.error(f"Failed to initialize vLLM: {e}")
    llm = None

# Initialize Chroma client
try:
    chroma_client = chromadb.Client(Settings(allow_reset=True))
    logger.info("Initialized Chroma client")
except Exception as e:
    logger.error(f"Failed to initialize Chroma: {e}")
    chroma_client = None

# Get or create Chroma collection
def get_collection():
    """Get the Chroma collection, creating it if necessary"""
    try:
        global chroma_client
        if chroma_client is None:
            raise Exception("Chroma client not initialized")
        
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
        except:
            collection = chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created Chroma collection: {COLLECTION_NAME}")
        return collection
    except Exception as e:
        logger.error(f"Error getting Chroma collection: {e}")
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
        
        if llm is None:
            raise Exception("vLLM model not initialized")
        
        sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=200)
        outputs = llm.generate([prompt], sampling_params)
        enhanced_query = outputs[0].outputs[0].text.strip()
        
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
    
    original_query = query
    if use_context:
        contextual_query = get_contextual_query(query)
        logger.info(f"Using intent-aware query: {contextual_query[:100]}...")
    else:
        contextual_query = query
    
    query_embedding = generate_query_embedding(contextual_query)
    if query_embedding is None:
        logger.error("Failed to generate query embedding")
        return []
    
    collection = get_collection()
    if not collection:
        logger.error("No collection available")
        return []
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        articles = []
        for id, metadata, distance in zip(results["ids"][0], results["metadatas"][0], results["distances"][0]):
            try:
                article = {
                    "id": int(id),
                    "title": metadata.get("title", "Unknown"),
                    "url": metadata.get("url", "#"),
                    "full_text": metadata.get("full_text", ""),
                    "publish_date": metadata.get("publish_date", "Unknown"),
                    "keyword": metadata.get("keyword", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "article_keywords": json.loads(metadata.get("article_keywords", "[]")),
                    "similarity": 1.0 - distance  # Convert distance to similarity
                }
                articles.append(article)
            except Exception as e:
                logger.warning(f"Error processing article {id}: {e}")
                continue
        
        logger.info(f"Query processing took {time.time() - start_time:.2f} seconds")
        logger.info(f"Found {len(articles)} relevant articles for query: {query}")
        return articles
    except Exception as e:
        logger.error(f"Error querying Chroma: {e}")
        return []

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
        
        if llm is None:
            raise Exception("vLLM model not initialized")
        
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1500)
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        logger.info(f"Response generation took {time.time() - start_time:.2f} seconds")
        return response, sources
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
    
    if not get_collection():
        print("Failed to initialize Chroma. Exiting.")
        return
    
    if llm is None:
        print("Failed to initialize vLLM model. Exiting.")
        return
    
    collection = get_collection()
    if collection.count() == 0:
        print("No articles found in the database. Please run import_articles_chroma.py first.")
        return
    
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
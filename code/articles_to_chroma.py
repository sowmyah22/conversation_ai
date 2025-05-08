### Embedds the text and stores in chroma db
import json
import logging
import chromadb
import numpy as np
import ollama
import glob
import os
from typing import Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
db_path = './chroma_db'  # Directory for Chroma persistent storage
collection_name = 'articles'
EMBEDDING_DIM = 768  # Dimension of nomic-embed-text embeddings

# Find the latest NewsAPI output file
def get_latest_articles_file(data_dir: str = 'data') -> Optional[str]:
    try:
        json_files = glob.glob(os.path.join(data_dir, 'articles.json'))
        if not json_files:
            return None
        return max(json_files, key=os.path.getmtime)  # Get the most recent file
    except Exception as e:
        logger.error(f"Error finding latest articles file: {e}")
        return None

# Initialize Chroma database
def init_chroma():
    try:
        # Initialize Chroma client with persistent storage
        client = chromadb.PersistentClient(path=db_path)
        # Delete existing collection if it exists (optional, comment out if not desired)
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection {collection_name}")
        except Exception:
            pass
        # Create new collection with cosine similarity
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for searches
        )
        logger.info(f"Initialized Chroma database at {db_path} with collection {collection_name}")
        return collection
    except Exception as e:
        logger.error(f"Failed to initialize Chroma: {e}")
        exit(1)

# Generate embedding using Ollama
def generate_embedding(text: str) -> Optional[np.ndarray]:
    try:
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        embedding = np.array(response['embedding'], dtype=np.float32)
        if len(embedding) != EMBEDDING_DIM:
            logger.error(f"Generated embedding has incorrect dimension: {len(embedding)}")
            return None
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def main():
    # Find the input file
    input_file = get_latest_articles_file()
    if not input_file:
        logger.error("No articles JSON file found in data directory")
    else:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            logger.info(f"Loaded {len(articles)} articles from {input_file}")
        except Exception as e:
            logger.error(f"Error loading {input_file}: {e}")
            exit(1)

    # Initialize Chroma
    collection = init_chroma()
    seen_urls = set()  # Track URLs to skip duplicates

    # Process articles
    for idx, article in enumerate(articles, 1):
        try:
            # Generate embedding
            logger.info(f"Generating embedding for article {idx}: {article.get('title', 'unknown')}")
            embedding = generate_embedding(article.get('full_text', ''))
            if embedding is None:
                logger.warning(f"Skipping article {idx} due to embedding error")
                continue

            # Prepare metadata
            article_id = str(article.get('id', idx))  # Use provided ID or index
            title = str(article.get('title', 'Unknown'))
            url = str(article.get('url', 'Unknown'))
            full_text = str(article.get('full_text', ''))
            publish_date = str(article.get('publish_date', 'Unknown'))
            keyword = str(article.get('keyword', 'Unknown'))
            author = str(article.get('author', 'Unknown'))
            article_keywords = article.get('article_keywords', [])

            # Check for duplicate URLs
            if url in seen_urls:
                logger.warning(f"Skipping article {idx} due to duplicate URL: {url}")
                continue
            seen_urls.add(url)

            # Store in Chroma
            collection.add(
                ids=[article_id],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    'id': article_id,  # Explicit ID for LangChain compatibility
                    'title': title,
                    'url': url,
                    'publish_date': publish_date,
                    'keyword': keyword,
                    'author': author,
                    'article_keywords': json.dumps(article_keywords)
                }],
                documents=[full_text]
            )

            logger.info(f"Prepared article {idx}: {title}")
        except Exception as e:
            logger.warning(f"Error preparing article {idx}: {e}")
            continue

    # Log final count
    try:
        article_count = collection.count()
        logger.info(f"Saved {article_count} articles to Chroma database")
    except Exception as e:
        logger.error(f"Error querying Chroma: {e}")

    logger.info("Completed processing")

if __name__ == "__main__":
    main()
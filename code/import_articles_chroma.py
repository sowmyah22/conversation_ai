import json
import logging
import chromadb
from chromadb.config import Settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Input file and Chroma configuration
input_file = 'embeddings.json'
collection_name = 'articles'

# Initialize Chroma client
try:
    chroma_client = chromadb.Client(Settings(allow_reset=True))
    logger.info("Initialized Chroma client")
except Exception as e:
    logger.error(f"Failed to initialize Chroma: {e}")
    exit(1)

# Create or connect to collection
try:
    # Delete existing collection if it exists (for clean setup)
    try:
        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted existing Chroma collection: {collection_name}")
    except:
        pass
    
    # Create new collection with cosine similarity
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Created Chroma collection: {collection_name}")
except Exception as e:
    logger.error(f"Error creating collection: {e}")
    exit(1)

# Load articles with embeddings
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles from {input_file}")
except FileNotFoundError:
    logger.error(f"Input file {input_file} not found")
    exit(1)
except Exception as e:
    logger.error(f"Error reading {input_file}: {e}")
    exit(1)

# Prepare data for insertion
ids = []
embeddings = []
metadatas = []
documents = []

for idx, article in enumerate(articles, 1):
    try:
        embedding = article.get('embedding')
        if not embedding or len(embedding) != 768:
            logger.warning(f"Invalid or missing embedding for article {idx}: {article.get('title', 'unknown')}")
            continue

        # Ensure all metadata fields are strings, replacing None with 'Unknown'
        title = str(article.get('title', 'Unknown')) if article.get('title') is not None else 'Unknown'
        url = str(article.get('url', 'Unknown')) if article.get('url') is not None else 'Unknown'
        full_text = str(article.get('full_text', '')) if article.get('full_text') is not None else ''
        publish_date = str(article.get('publish_date', 'Unknown')) if article.get('publish_date') is not None else 'Unknown'
        keyword = str(article.get('keyword', 'Unknown')) if article.get('keyword') is not None else 'Unknown'
        author = str(article.get('author', 'Unknown')) if article.get('author') is not None else 'Unknown'
        article_keywords = json.dumps(article.get('article_keywords', [])) if article.get('article_keywords') is not None else json.dumps([])

        ids.append(str(idx))
        embeddings.append(embedding)
        metadatas.append({
            "title": title,
            "url": url,
            "full_text": full_text,
            "publish_date": publish_date,
            "keyword": keyword,
            "author": author,
            "article_keywords": article_keywords
        })
        documents.append(full_text)

        logger.info(f"Prepared article {idx}: {title}")
    except Exception as e:
        logger.warning(f"Error preparing article {idx}: {e}")
        continue

# Insert data into Chroma
try:
    if ids:  # Only insert if there is data
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        logger.info(f"Saved {len(ids)} articles to Chroma collection {collection_name}")
    else:
        logger.warning("No valid articles to insert")
except Exception as e:
    logger.error(f"Error inserting to Chroma: {e}")
    exit(1)

logger.info("Completed processing")
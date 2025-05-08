import json
import logging
import sqlite3
import numpy as np
import ollama
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
input_file = 'embeddings.json'  # Change to 'articles.json' if no embeddings
db_file = 'articles.db'
EMBEDDING_DIM = 768  # Dimension of nomic-embed-text embeddings

# Initialize SQLite database
def init_sqlite():
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # Drop existing table to clear any constraints
        cursor.execute('DROP TABLE IF EXISTS articles')
        # Create new table with embedding column
        cursor.execute('''
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                url TEXT,
                full_text TEXT,
                publish_date TEXT,
                keyword TEXT,
                author TEXT,
                article_keywords TEXT,
                embedding BLOB
            )
        ''')
        conn.commit()
        logger.info(f"Initialized SQLite database: {db_file}")
        return conn
    except Exception as e:
        logger.error(f"Failed to initialize SQLite: {e}")
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

# Load articles and process
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles from {input_file}")
except FileNotFoundError:
    logger.error(f"Input file {input_file} not found")
    # Create a sample article
    articles = [
        {
            "title": "Test Article",
            "url": "https://example.com",
            "full_text": "This is a test article about AI advancements.",
            "publish_date": "2023-01-01",
            "keyword": "AI",
            "author": "John Doe",
            "article_keywords": ["AI", "tech"]
        }
    ]
    logger.info("Using sample article")

# Initialize SQLite
conn = init_sqlite()
cursor = conn.cursor()
seen_urls = set()  # Track URLs to skip duplicates

# Process articles
for idx, article in enumerate(articles, 1):
    try:
        # Generate embedding if not provided
        if 'embedding' not in article or not article['embedding'] or len(article['embedding']) != EMBEDDING_DIM:
            logger.info(f"Generating embedding for article {idx}: {article.get('title', 'unknown')}")
            embedding = generate_embedding(article.get('full_text', ''))
            if embedding is None:
                logger.warning(f"Skipping article {idx} due to embedding error")
                continue
        else:
            embedding = np.array(article['embedding'], dtype=np.float32)

        # Prepare metadata
        title = str(article.get('title', 'Unknown'))
        url = str(article.get('url', 'Unknown'))
        full_text = str(article.get('full_text', ''))
        publish_date = str(article.get('publish_date', 'Unknown'))
        keyword = str(article.get('keyword', 'Unknown'))
        author = str(article.get('author', 'Unknown'))
        article_keywords = json.dumps(article.get('article_keywords', []))

        # Check for duplicate URLs
        if url in seen_urls:
            logger.warning(f"Skipping article {idx} due to duplicate URL: {url}")
            continue
        seen_urls.add(url)

        # Convert embedding to bytes for SQLite BLOB
        embedding_bytes = embedding.tobytes()

        # Store in SQLite
        cursor.execute('''
            INSERT INTO articles (id, title, url, full_text, publish_date, keyword, author, article_keywords, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (idx, title, url, full_text, publish_date, keyword, author, article_keywords, embedding_bytes))

        logger.info(f"Prepared article {idx}: {title}")
    except Exception as e:
        logger.warning(f"Error preparing article {idx}: {e}")
        continue

# Commit and close
try:
    conn.commit()
    article_count = cursor.execute('SELECT COUNT(*) FROM articles').fetchone()[0]
    logger.info(f"Saved {article_count} articles to SQLite database")
except Exception as e:
    logger.error(f"Error saving to SQLite: {e}")
finally:
    conn.close()

logger.info("Completed processing")
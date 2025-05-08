import json
import logging
import ollama

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Input and output files
input_file = 'toi_articles_by_keyword.json'
output_file = 'embeddings.json'

# Function to generate embedding for text using Ollama
def generate_embedding(text):
    if not text:
        logger.warning("Empty text provided for embedding")
        return None
    try:
        # Truncate text to avoid exceeding model limits (8192 tokens)
        max_length = 5000  # Characters, rough estimate
        text = text[:max_length]
        # Generate embedding using Ollama
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        embedding = response['embedding']
        return embedding  # List of 768 floats
    except Exception as e:
        logger.warning(f"Error generating embedding: {e}")
        return None

# Verify Ollama model availability
try:
    ollama.list()  # Check if Ollama server is running
    logger.info("Ollama server is running, using nomic-embed-text model")
except Exception as e:
    logger.error(f"Failed to connect to Ollama server or model not found: {e}")
    logger.error("Ensure Ollama is running (`ollama serve`) and `nomic-embed-text` is pulled (`ollama pull nomic-embed-text`)")
    exit(1)

# Load scraped articles
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

# Process articles and generate embeddings
articles_with_embeddings = []
for idx, article in enumerate(articles, 1):
    try:
        # Extract full_text for embedding
        text = article.get('full_text', '')
        if not text:
            logger.warning(f"Article {idx} has no full_text: {article.get('url', 'unknown')}")
            article['embedding'] = None
            articles_with_embeddings.append(article)
            continue
        # Optionally include title: text = f"{article.get('title', '')}\n{text}"
        embedding = generate_embedding(text)
        article['embedding'] = embedding
        if embedding is not None:
            logger.info(f"Generated embedding for article {idx}: {article.get('title', 'unknown')}")
        else:
            logger.warning(f"Failed to generate embedding for article {idx}: {article.get('title', 'unknown')}")
        articles_with_embeddings.append(article)
    except Exception as e:
        logger.warning(f"Error processing article {idx}: {e}")
        article['embedding'] = None
        articles_with_embeddings.append(article)

# Save articles with embeddings
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles_with_embeddings, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved {len(articles_with_embeddings)} articles with embeddings to {output_file}")
except Exception as e:
    logger.error(f"Failed to save {output_file}: {e}")
    exit(1)

print(f"Embedding complete. Saved {len(articles_with_embeddings)} articles to {output_file}")
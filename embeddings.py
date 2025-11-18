"""
Embedding generation module for StudyBuddy.
Converts text into vector embeddings using OpenRouter or local models.
"""

import logging
import os
from typing import List, Optional
from dotenv import load_dotenv

# Try to import sentence transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Only OpenRouter embeddings will work.")

# Load environment variables
load_dotenv()

from config import Config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using OpenRouter or local SentenceTransformers."""

    def __init__(self, model_name: Optional[str] = None, use_openrouter: bool = True, use_local: bool = False):
        """Initialize embedding generator.

        Args:
            model_name: model to use for embeddings
            use_openrouter: whether to use OpenRouter API
            use_local: whether to use local sentence-transformers
        """
        self.use_openrouter = use_openrouter
        self.use_local = use_local
        self.model_name = model_name or Config.EMBEDDING_MODEL

        if self.use_openrouter:
            if not Config.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY not set in environment")
            self.openrouter_api_key = Config.OPENROUTER_API_KEY
            self.openrouter_base = Config.OPENROUTER_API_BASE
            logger.info(f"Using OpenRouter for embeddings (base={self.openrouter_base})")

        if self.use_local:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded local SentenceTransformer model: {self.model_name}")

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Embed a single text string."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text for embedding")
                return None

            if self.use_local:
                emb = self.model.encode(text, convert_to_numpy=True)
                return emb.tolist()

            if self.use_openrouter:
                return self._embed_text_openrouter(text)

            logger.error("No embedding backend configured")
            return None
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
        """Embed a batch of texts."""
        try:
            if not texts:
                return []

            valid_texts = [t for t in texts if t and t.strip()]
            if not valid_texts:
                return [None] * len(texts)

            embeddings = []
            if self.use_local:
                embs = self.model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
                embeddings = [e.tolist() for e in embs]
                return embeddings

            if self.use_openrouter:
                # OpenRouter embedding endpoint (OpenAI-compatible payload)
                for i in range(0, len(valid_texts), batch_size):
                    batch = valid_texts[i:i + batch_size]
                    batch_embs = self._embed_batch_openrouter(batch)
                    embeddings.extend(batch_embs)
                return embeddings

            return [None] * len(texts)

        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            return [None] * len(texts)

    def _embed_batch_openrouter(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Call OpenRouter embeddings endpoint for a batch."""
        try:
            import requests
            endpoints = [
                f"{self.openrouter_base.rstrip('/')}/embeddings",
                f"{self.openrouter_base.rstrip('/')}/api/v1/embeddings",
                f"{self.openrouter_base.rstrip('/')}/v1/embeddings",
            ]
            headers = {
                'Authorization': f'Bearer {self.openrouter_api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': self.model_name,
                'input': texts
            }
            resp = None
            data = None
            for url in endpoints:
                try:
                    resp = requests.post(url, json=payload, headers=headers, timeout=60)
                    if resp.status_code >= 200 and resp.status_code < 300:
                        try:
                            data = resp.json()
                            logger.info(f"OpenRouter embeddings success via {url}")
                            break
                        except Exception:
                            logger.warning(f"OpenRouter returned non-JSON at {url}: {resp.text}")
                            continue
                except Exception as e:
                    logger.debug(f"Embedding endpoint {url} failed: {e}")
                    continue
            if data is None:
                # No endpoint succeeded
                raise RuntimeError(f"All OpenRouter embedding endpoints failed; last status: {getattr(resp,'status_code',None)}")
            # Handle typical OpenAI-style response
            if 'data' in data and isinstance(data['data'], list):
                embs = []
                for item in data['data']:
                    if isinstance(item, dict) and 'embedding' in item:
                        embs.append(item['embedding'])
                    else:
                        embs.append(None)
                return embs
            # Handle possible alternative shapes
            if 'embeddings' in data and isinstance(data['embeddings'], list):
                return data['embeddings']
            # Unknown format
            logger.error(f"Unexpected OpenRouter embedding response format: {data}")
            return [None] * len(texts)
        except Exception as e:
            logger.error(f"OpenRouter batch embedding error: {e}")
            # Retry per-item to possibly succeed
            results = []
            for t in texts:
                try:
                    single = self._embed_text_openrouter(t)
                    results.append(single)
                except Exception:
                    results.append(None)
            return results

    def _embed_text_openrouter(self, text: str) -> Optional[List[float]]:
        """Call OpenRouter embedding endpoint for a single text."""
        try:
            import requests
            endpoints = [
                f"{self.openrouter_base.rstrip('/')}/embeddings",
                f"{self.openrouter_base.rstrip('/')}/api/v1/embeddings",
                f"{self.openrouter_base.rstrip('/')}/v1/embeddings",
            ]
            headers = {
                'Authorization': f'Bearer {self.openrouter_api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': self.model_name,
                'input': text
            }
            resp = None
            data = None
            for url in endpoints:
                try:
                    resp = requests.post(url, json=payload, headers=headers, timeout=30)
                    if resp.status_code >= 200 and resp.status_code < 300:
                        try:
                            data = resp.json()
                            logger.info(f"OpenRouter single embedding success via {url}")
                            break
                        except Exception:
                            logger.warning(f"OpenRouter single returned non-JSON at {url}: {resp.text}")
                            continue
                except Exception as e:
                    logger.debug(f"Embedding single endpoint {url} failed: {e}")
                    continue
            if data is None:
                raise RuntimeError(f"All OpenRouter single embedding endpoints failed; last status: {getattr(resp,'status_code',None)}")
            # Accept common shapes
            if 'data' in data and isinstance(data['data'], list) and 'embedding' in data['data'][0]:
                return data['data'][0]['embedding']
            if 'embedding' in data:
                return data['embedding']
            if 'embeddings' in data and isinstance(data['embeddings'], list):
                return data['embeddings'][0]
            logger.error(f"Unexpected OpenRouter single embedding response format: {data}")
            return None
        except Exception as e:
            logger.error(f"OpenRouter embedding error: {e}")
            return None

    def get_embedding_dimension(self) -> int:
        """Return embedding dimension; best-effort guess."""
        if self.use_local:
            return self.model.get_sentence_embedding_dimension()
        # common default for ada-like models
        return 1536


class EmbeddingCache:
    """Cache for embeddings to avoid recomputing."""

    def __init__(self):
        self.cache = {}

    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        return self.cache.get(text)

    def set(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        self.cache[text] = embedding

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def size(self) -> int:
        """Get number of cached embeddings."""
        return len(self.cache)
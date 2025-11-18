"""
Vector store module for StudyBuddy.
Handles storing and retrieving document embeddings using Qdrant.
"""

import logging
import os
import uuid
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

from config import Config

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False
    logger.error("qdrant-client not available. Install with: pip install qdrant-client[http]")


class VectorStore:
    """Manages vector storage and retrieval using Qdrant."""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or Config.QDRANT_COLLECTION
        self.url = Config.QDRANT_URL
        self.api_key = Config.QDRANT_API_KEY

        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed")

        # Initialize client
        # If api_key provided, use it via api_key param
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key, prefer_grpc=False)
        else:
            # If URL points to localhost, no api key needed
            self.client = QdrantClient(url=self.url, prefer_grpc=False)

        # Create collection if not exists or recreate if misconfigured
        try:
            collection_exists = self._collection_exists(self.collection_name)
            needs_recreation = False

            if collection_exists:
                # Check if collection has proper vector configuration
                try:
                    collection_info = self.client.get_collection(self.collection_name)
                    vectors_config = collection_info.config.params.vectors
                    # If vectors config is empty dict or None, we need to recreate
                    if not vectors_config or (isinstance(vectors_config, dict) and len(vectors_config) == 0):
                        logger.warning(f"Collection '{self.collection_name}' has empty vector config, recreating...")
                        needs_recreation = True
                except Exception as e:
                    logger.warning(f"Could not verify collection config: {e}, recreating...")
                    needs_recreation = True

            if not collection_exists or needs_recreation:
                # Delete existing collection if it needs recreation
                if needs_recreation:
                    try:
                        self.client.delete_collection(self.collection_name)
                        logger.info(f"Deleted misconfigured collection '{self.collection_name}'")
                    except Exception:
                        pass

                # Create new collection with proper vector config
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(size=1536, distance=qmodels.Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}' with vector size 1536")

            logger.info(f"Qdrant collection '{self.collection_name}' ready at {self.url}")
        except Exception as e:
            logger.error(f"Error creating/initializing Qdrant collection: {e}")
            raise

    def _collection_exists(self, name: str) -> bool:
        try:
            collections = self.client.get_collections()
            return any(col.name == name for col in collections.collections)
        except Exception:
            return False

    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]], batch_size: int = 100) -> bool:
        """
        Add documents with embeddings to the vector store.

        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors
            batch_size: Number of documents to add at once

        Returns:
            True if successful, False otherwise
        """
        try:
            if not chunks or not embeddings:
                logger.error("No chunks or embeddings to add")
                return False

            if len(chunks) != len(embeddings):
                logger.error("Chunks and embeddings length mismatch")
                return False

            # Prepare points
            point_ids = []
            vectors = []
            payloads = []

            for chunk, emb in zip(chunks, embeddings):
                if emb is None:
                    continue
                source = chunk.get('source', 'unknown')
                chunk_id = chunk.get('chunk_id', 'chunk')
                # Generate UUID v5 (deterministic) from source and chunk_id
                # This ensures same chunk always gets same UUID
                namespace = uuid.NAMESPACE_DNS
                point_id_str = f"{source}_{chunk_id}"
                point_id = str(uuid.uuid5(namespace, point_id_str))
                point_ids.append(point_id)
                vectors.append(emb)

                meta = {
                    'source': source,
                    'chunk_id': chunk_id,
                    'chunk_index': chunk.get('chunk_index', 0),
                    'chunk_size': chunk.get('chunk_size', 0),
                    'total_chunks': chunk.get('total_chunks', 0)
                }
                if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                    file_meta = chunk['metadata']
                    meta['filename'] = file_meta.get('filename')
                # store the text separately as payload
                meta['text'] = chunk.get('text', '')
                payloads.append(meta)

            # Upload in batches
            for i in range(0, len(vectors), batch_size):
                batch_ids = point_ids[i:i+batch_size]
                batch_vectors = vectors[i:i+batch_size]
                batch_payloads = payloads[i:i+batch_size]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=qmodels.Batch(ids=batch_ids, vectors=batch_vectors, payloads=batch_payloads)
                )
                logger.info(f"Added batch of {len(batch_ids)} vectors to Qdrant")

            return True
        except Exception as e:
            logger.error(f"Error adding to Qdrant: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of matching documents with metadata
        """
        try:
            if not query_embedding:
                return []

            # Prepare filter
            qfilter = None
            if filter_dict:
                # simple eq filters
                must = []
                for k, v in filter_dict.items():
                    must.append(qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)))
                qfilter = qmodels.Filter(must=must)

            response = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qfilter
            )

            results = []
            for hit in response:
                meta = hit.payload
                results.append({
                    'id': hit.id,
                    'text': meta.get('text', ''),
                    'metadata': meta,
                    'distance': hit.score
                })
            return results
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []

    def search_by_text(self, query_text: str, embedder, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search using text query (convenience method).

        Args:
            query_text: Text to search for
            embedder: EmbeddingGenerator instance
            top_k: Number of results
            filter_dict: Optional filters

        Returns:
            List of matching documents
        """
        query_embedding = embedder.embed_text(query_text)
        if not query_embedding:
            return []
        return self.search(query_embedding, top_k, filter_dict)

    def delete_by_source(self, source: str) -> bool:
        """Delete all documents from a specific source."""
        try:
            # Filter points by payload
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=[qmodels.FieldCondition(key='source', match=qmodels.MatchValue(value=source))]))
            )
            logger.info(f"Deleted points with source={source}")
            return True
        except Exception as e:
            logger.error(f"Error deleting by source: {e}")
            return False

    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            # recreate empty collection
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=1536, distance=qmodels.Distance.COSINE)
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    def get_all_sources(self) -> List[str]:
        """Get list of all unique sources in the collection."""
        try:
            resp = self.client.scroll(collection_name=self.collection_name, limit=100, with_payload=True)
            sources = set()
            # qdrant-client may return an object with .points or a tuple (points, next_cursor)
            points = None
            if hasattr(resp, 'points'):
                points = resp.points
            elif isinstance(resp, tuple) and len(resp) >= 1:
                points = resp[0]
            elif isinstance(resp, dict) and 'result' in resp and 'points' in resp['result']:
                points = resp['result']['points']

            if not points:
                return []

            for point in points:
                # point may be a simple dict or an object
                payload = None
                if hasattr(point, 'payload'):
                    payload = point.payload
                elif isinstance(point, dict):
                    payload = point.get('payload') or point
                if isinstance(payload, dict):
                    sources.add(payload.get('source', 'unknown'))
            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count_resp = self.client.count(collection_name=self.collection_name)
            # count_resp may be an object or dict
            count = getattr(count_resp, 'count', None) or (count_resp.get('count') if isinstance(count_resp, dict) else 0)
            sources = self.get_all_sources()
            return {
                'total_documents': count,
                'total_sources': len(sources),
                'sources': sources,
                'collection_name': self.collection_name,
                'qdrant_url': self.url
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    if not QDRANT_AVAILABLE:
        print("qdrant-client not installed")
    else:
        vs = VectorStore()
        print(vs.get_collection_stats())

        # Test with sample data
        sample_chunks = [
            {
                'chunk_id': 'chunk_0',
                'text': 'This is a test chunk.',
                'source': 'test.txt',
                'chunk_index': 0,
                'chunk_size': 21,
                'total_chunks': 1
            }
        ]

        # Would need actual embeddings to add
        print(f"VectorStore initialized and ready")

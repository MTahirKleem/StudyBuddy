"""
Handles question answering, document summarization, and flashcard generation.
RAG (Retrieval-Augmented Generation) pipeline for StudyBuddy.
"""

from config import Config
from embeddings import EmbeddingGenerator
import requests
import logging
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)

# Use requests to call OpenRouter


class RAGPipeline:
    """Main RAG pipeline for question answering and content generation using OpenRouter."""

    def __init__(
        self,
        vectorstore,
        embedder: EmbeddingGenerator,
        llm_model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500
    ):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.llm_model = llm_model or Config.LLM_MODEL
        self.temperature = float(os.getenv('LLM_TEMPERATURE', temperature))
        self.max_tokens = max_tokens

        self.openrouter_base = Config.OPENROUTER_API_BASE.rstrip('/')
        self.openrouter_api_key = Config.OPENROUTER_API_KEY or Config.OPENAI_API_KEY
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY must be set for LLM calls")

        logger.info(f"Initialized RAG pipeline with model: {self.llm_model}")

    def _call_openrouter_chat(self, messages: List[Dict], max_tokens: int = 500) -> str:
        url = f"{self.openrouter_base}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': self.llm_model,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': max_tokens
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # OpenRouter returns choices[0].message.content similar to OpenAI
            return data['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"OpenRouter LLM error: {e} - response: {getattr(e, 'response', None)}")
            raise

    def retrieve(self, question: str, top_k: int = 3, filter_dict: Optional[Dict] = None) -> List[Dict]:
        try:
            query_embedding = self.embedder.embed_text(question)
            if not query_embedding:
                logger.error("Failed to embed question")
                return []
            results = self.vectorstore.search(query_embedding=query_embedding, top_k=top_k, filter_dict=filter_dict)
            return results
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return []

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "I don't have enough information to answer that question. Please make sure relevant documents have been uploaded."

        context = self._format_context(context_chunks)
        prompt = f"""Answer the following question directly and concisely using ONLY the information from the context below.

IMPORTANT INSTRUCTIONS:
- Provide a direct answer without phrases like "Based on the context" or "The context shows"
- Do NOT explain what the context does or doesn't contain
- If the context has the answer, state it clearly and completely
- Only say you don't have enough information if the answer is truly not in the context
- Be specific and factual

Context:
{context}

Question: {question}

Direct Answer:"""
        messages = [
            {"role": "system", "content": "You are a helpful study assistant. Answer questions directly and concisely using only the provided context. Never use phrases like 'based on the context' or 'the context shows'. Just provide the factual answer."},
            {"role": "user", "content": prompt}
        ]
        try:
            return self._call_openrouter_chat(messages, max_tokens=self.max_tokens)
        except Exception as e:
            logger.error(f"Error generating answer via OpenRouter: {e}")
            return f"Error generating answer: {e}"

    def ask(self, question: str, top_k: int = 3, filter_dict: Optional[Dict] = None, include_sources: bool = True) -> Dict:
        try:
            context_chunks = self.retrieve(question, top_k, filter_dict)
            answer = self.generate_answer(question, context_chunks)
            result = {
                'question': question,
                'answer': answer,
                'num_sources': len(context_chunks)
            }
            if include_sources:
                result['sources'] = self._extract_sources(context_chunks)
                result['context_chunks'] = context_chunks
            return result
        except Exception as e:
            logger.error(f"Error in ask: {e}")
            return {'question': question, 'answer': f'Error: {e}', 'num_sources': 0, 'sources': []}

    def summarize_document(self, source: str, max_length: int = 200, style: str = "concise") -> str:
        try:
            all_results = self.vectorstore.client.scroll(collection_name=self.vectorstore.collection_name, with_payload=True, limit=100)

            # Extract points from response (handle tuple, object, or dict)
            points = None
            if hasattr(all_results, 'points'):
                points = all_results.points
            elif isinstance(all_results, tuple) and len(all_results) >= 1:
                points = all_results[0]
            elif isinstance(all_results, dict) and 'result' in all_results:
                points = all_results['result'].get('points', [])

            if not points:
                return f"No content found for source: {source}"

            # Filter by source
            documents = []
            for p in points:
                payload = p.payload if hasattr(p, 'payload') else p.get('payload', {})
                if isinstance(payload, dict) and payload.get('source') == source:
                    documents.append(payload.get('text', ''))
            if not documents:
                return f"No content found for source: {source}"
            if len(documents) > 10:
                step = len(documents) // 10
                documents = documents[::step][:10]
            content = "\n\n".join(documents)
            if len(content) > 8000:
                content = content[:8000] + "..."
            style_instruction = {
                'bullet-points': 'Summarize using bullet points (•).',
                'detailed': 'Provide a detailed summary covering all main points.'
            }.get(style, 'Provide a concise summary focusing on the most important concepts.')
            prompt = f"Summarize the following content in approximately {max_length} words. {style_instruction}\n\nContent:\n{content}\n\nSummary:"
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant that creates clear, accurate summaries.'},
                {'role': 'user', 'content': prompt}
            ]
            return self._call_openrouter_chat(messages, max_tokens=max_length*2)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"

    def generate_flashcards(self, source: str, num_cards: int = 10, difficulty: str = 'medium') -> List[Dict]:
        try:
            all_results = self.vectorstore.client.scroll(collection_name=self.vectorstore.collection_name, with_payload=True, limit=200)

            # Extract points from response (handle tuple, object, or dict)
            points = None
            if hasattr(all_results, 'points'):
                points = all_results.points
            elif isinstance(all_results, tuple) and len(all_results) >= 1:
                points = all_results[0]
            elif isinstance(all_results, dict) and 'result' in all_results:
                points = all_results['result'].get('points', [])

            if not points:
                return []

            # Filter by source
            documents = []
            for p in points:
                payload = p.payload if hasattr(p, 'payload') else p.get('payload', {})
                if isinstance(payload, dict) and payload.get('source') == source:
                    documents.append(payload.get('text', ''))
            if not documents:
                return []
            if len(documents) > 15:
                step = len(documents) // 15
                documents = documents[::step][:15]
            content = "\n\n".join(documents)
            if len(content) > 8000:
                content = content[:8000] + "..."
            difficulty_instruction = {
                'easy': 'Create straightforward questions that test basic understanding and recall.',
                'medium': 'Create questions that test comprehension and application of concepts.',
                'hard': 'Create challenging questions that test analysis and deep understanding.'
            }.get(difficulty, 'Create questions that test understanding of the material.')
            prompt = f"Generate exactly {num_cards} flashcard questions and answers based on this content. {difficulty_instruction}\n\nFormat each flashcard exactly as:\nQ: [question]\nA: [answer]\n\nContent:\n{content}\n\nFlashcards:"
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant that creates educational flashcards.'},
                {'role': 'user', 'content': prompt}
            ]
            resp = self._call_openrouter_chat(messages, max_tokens=num_cards*120)
            flashcards = self._parse_flashcards(resp, source)
            return flashcards
        except Exception as e:
            logger.error(f"Error generating flashcards: {e}")
            return []

    def _format_context(self, chunks: List[Dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            source = chunk.get('metadata', {}).get('source', chunk.get('metadata', {}).get('filename', 'unknown'))
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        return "\n\n".join(context_parts)

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        sources = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            sources.append({
                'index': i,
                'source': metadata.get('source', metadata.get('filename', 'unknown')),
                'chunk_id': metadata.get('chunk_id', 'unknown'),
                'text_preview': chunk.get('text', '')[:150] + '...'
            })
        return sources

    def _parse_flashcards(self, text: str, source: str) -> List[Dict]:
        flashcards = []
        parts = text.split('\n\n')
        current_q = None
        current_a = None
        for part in parts:
            part = part.strip()
            if not part:
                continue
            lines = part.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    if current_q and current_a:
                        flashcards.append({'question': current_q, 'answer': current_a, 'source': source, 'tags': source})
                    current_q = line[2:].strip()
                    current_a = None
                elif line.startswith('A:'):
                    current_a = line[2:].strip()
                elif current_q and not current_a:
                    current_q += ' ' + line
                elif current_a:
                    current_a += ' ' + line
        if current_q and current_a:
            flashcards.append({'question': current_q, 'answer': current_a, 'source': source, 'tags': source})
        return flashcards

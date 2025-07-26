import os
import json
import time
import pickle
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    Settings
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
# from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass
class RAGConfig:
    """Centralized configuration for RAG system"""
    # Embedding configuration
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_base_url: str = "https://api.mistral.ai/v1"
    embed_batch_size: int = 8
    embed_cache_enabled: bool = True
    
    # LLM configuration
    llm_model: str = "mistral-small-2506"
    llm_base_url: str = "https://localhost:8000/v1"
    llm_context_window: int = 65536
    system_prompt: str = "You are a helpful AI assistant. Answer questions based on the provided context. Be concise and accurate."
    
    # API Keys for cloud providers
    openai_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    together_api_key: Optional[str] = None
    hf_api_key: Optional[str] = None
    deepinfra_api_key: Optional[str] = None
    
    # Chunking configuration
    chunk_size: int = 4096
    chunk_overlap: int = 500
    chunk_strategy: str = "sentence"
    
    # Retrieval configuration
    similarity_top_k: int = 10
    rerank_enabled: bool = True
    hybrid_search_enabled: bool = False
    
    # Storage configuration
    collection_name: str = "rag_collection"
    persist_dir: str = "./chroma_db"
    
    # Cache configuration
    query_cache_ttl: int = 3600  # 1 hour
    query_cache_max_size: int = 1000
    
    # Monitoring configuration
    enable_monitoring: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'RAGConfig':
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


class QueryCache:
    """Simple LRU cache for query results"""
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times = {}
    
    def _generate_key(self, query: str, top_k: int) -> str:
        """Generate cache key from query and parameters"""
        return hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
    
    def get(self, query: str, top_k: int) -> Optional[Any]:
        """Get cached result if available and not expired"""
        key = self._generate_key(query, top_k)
        if key in self.cache:
            if time.time() - self.access_times[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, query: str, top_k: int, result: Any):
        """Cache query result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        key = self._generate_key(query, top_k)
        self.cache[key] = result
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        self.access_times.clear()


class PerformanceMonitor:
    """Monitor and log performance metrics"""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {
            "query_times": [],
            "embedding_times": [],
            "index_build_times": [],
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def log_query_time(self, duration: float):
        """Log query execution time"""
        self.metrics["query_times"].append(duration)
        self.metrics["total_queries"] += 1
        self.logger.debug(f"Query completed in {duration:.3f} seconds")
    
    def log_embedding_time(self, duration: float, num_docs: int):
        """Log embedding generation time"""
        self.metrics["embedding_times"].append(duration)
        self.logger.debug(f"Generated embeddings for {num_docs} documents in {duration:.3f} seconds")
    
    def log_cache_hit(self):
        """Log cache hit"""
        self.metrics["cache_hits"] += 1
    
    def log_cache_miss(self):
        """Log cache miss"""
        self.metrics["cache_misses"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        return {
            "total_queries": self.metrics["total_queries"],
            "avg_query_time": avg(self.metrics["query_times"]),
            "avg_embedding_time": avg(self.metrics["embedding_times"]),
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["total_queries"]),
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"]
        }


class EnhancedRAGChatbot:
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize enhanced RAG chatbot with configuration"""
        self.config = config or RAGConfig()
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance monitor
        self.monitor = PerformanceMonitor(self.logger) if self.config.enable_monitoring else None
        
        # Initialize query cache
        self.query_cache = QueryCache(
            max_size=self.config.query_cache_max_size,
            ttl=self.config.query_cache_ttl
        ) if self.config.embed_cache_enabled else None
        
        # Determine API key based on base URL
        embed_api_key = self._get_api_key(self.config.embed_base_url)
        llm_api_key = self._get_api_key(self.config.llm_base_url)
        
        # Configure embedding model
        # self.embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-0.6B")
        if "deepinfra.com" in self.config.embed_base_url:
            self.embed_model = DeepInfraEmbeddingModel(
                model_id=self.config.embed_model,
                api_token=embed_api_key,
                normalize=True,
                # embed_batch_size=self.config.embed_batch_size,
                # encoding_format="float"
            )
        else:
            self.embed_model = OpenAILikeEmbedding(
                model_name=self.config.embed_model,
                api_base=self.config.embed_base_url,
                api_key=embed_api_key,
                embed_batch_size=self.config.embed_batch_size,
                encoding_format="float"
            )
        # self.embed_model = MistralAIEmbedding(
        #     model_name=self.config.embed_model,
        #     api_key=embed_api_key,
        #     embed_batch_size=self.config.embed_batch_size
        # )
        
        # Configure LLM
        self.llm = OpenAILike(
            model=self.config.llm_model,
            api_base=self.config.llm_base_url,
            api_key=llm_api_key,
            context_window=self.config.llm_context_window,
            is_chat_model=True,
        )
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.collection_name
        )
        
        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        
        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        self.index = None
        self.query_engine = None
        
        self.logger.info(f"Initialized EnhancedRAGChatbot with collection: {self.config.collection_name}")
    
    def _get_api_key(self, base_url: str) -> str:
        """Get appropriate API key based on the base URL"""
        if "openai.com" in base_url:
            return self.config.openai_api_key or "dummy"
        elif "anthropic.com" in base_url:
            return self.config.anthropic_api_key or "dummy"
        elif "together.xyz" in base_url:
            return self.config.together_api_key or "dummy"
        elif "huggingface.co" in base_url:
            return self.config.hf_api_key or "dummy"
        elif "deepinfra.com" in base_url:
            return self.config.deepinfra_api_key or "dummy"
        elif "mistral.ai" in base_url:
            return self.config.mistral_api_key or "dummy"
        else:
            return "dummy"  # For local servers
    
    def _create_query_engine(self, similarity_top_k: Optional[int] = None, streaming: bool = False):
        """Create query engine with system prompt"""
        if not self.index:
            return None
        
        top_k = similarity_top_k or self.config.similarity_top_k
        
        # Create custom QA template with system prompt
        qa_template = PromptTemplate(
            f"""### Task:
Respond to the user query using the provided context.

### Guidelines:
- If you don't know the answer, clearly state that.
- If uncertain, ask the user for clarification.
- Respond in the same language as the user's query.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
- If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
- Do not use XML tags in your response.

### Output:
Provide a clear and direct response to the user's query.

<context>
{{context_str}}
</context>

<user_query>
{{query_str}}
</user_query>

{self.config.system_prompt}
            """
        )
        
        return self.index.as_query_engine(
            similarity_top_k=top_k,
            streaming=streaming,
            text_qa_template=qa_template
        )
    
    # Collection Management Methods
    def list_collections(self) -> List[str]:
        """List all available collections"""
        collections = [col.name for col in self.chroma_client.list_collections()]
        self.logger.info(f"Found {len(collections)} collections")
        return collections
    
    def switch_collection(self, collection_name: str):
        """Switch to a different collection"""
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None
        self.query_engine = None
        self.config.collection_name = collection_name
        self.logger.info(f"Switched to collection: {collection_name}")
    
    def delete_collection(self, collection_name: str):
        """Delete a specific collection"""
        try:
            self.chroma_client.delete_collection(collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")
            if collection_name == self.config.collection_name:
                self.logger.warning("Deleted current collection. Please switch to another collection.")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about current collection"""
        stats = {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata if hasattr(self.collection, 'metadata') else {}
        }
        self.logger.info(f"Collection stats: {stats}")
        return stats
    
    def _get_webpage_url_from_json(self, file_path: str) -> Optional[str]:
        """Get webpage URL from corresponding JSON file"""
        try:
            # Get the directory and filename without extension
            file_dir = os.path.dirname(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            json_path = os.path.join(file_dir, f"{file_name}.json")
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    return json_data.get('webpage_url')
        except Exception as e:
            self.logger.warning(f"Failed to read JSON file for {file_path}: {e}")
        
        return None
    
    # Document Loading Methods
    def load_documents(self, file_path: Optional[str] = None, 
                      directory_path: Optional[str] = None,
                      recursive: bool = True) -> List[Document]:
        """Load documents from file or directory with enhanced options (markdown files only)"""
        start_time = time.time()
        
        if file_path:
            # Check if the file is a markdown file
            if not file_path.lower().endswith(('.md', '.markdown')):
                self.logger.warning(f"Skipping non-markdown file: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            filename = os.path.basename(file_path)
            file_stats = os.stat(file_path)
            
            # Try to get webpage URL from JSON file, fallback to filename
            webpage_url = self._get_webpage_url_from_json(file_path)
            source = webpage_url if webpage_url else filename
            
            documents = [Document(
                text=text, 
                metadata={
                    "source": source,
                    "file_path": file_path,
                    "file_size": file_stats.st_size,
                    "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "doc_id": hashlib.md5(file_path.encode()).hexdigest()
                }
            )]
        elif directory_path:
            # Configure reader to only load markdown files
            reader = SimpleDirectoryReader(
                directory_path,
                recursive=recursive,
                required_exts=[".md", ".markdown"]  # Only load markdown files
            )
            documents = reader.load_data()
            
            # Filter out any JSON files that might have been loaded accidentally
            markdown_documents = []
            for doc in documents:
                file_path = doc.metadata.get('file_path', '')
                
                # Skip JSON files
                if file_path.lower().endswith('.json'):
                    self.logger.debug(f"Skipping JSON file: {file_path}")
                    continue
                
                # Only process markdown files
                if not file_path.lower().endswith(('.md', '.markdown')):
                    self.logger.debug(f"Skipping non-markdown file: {file_path}")
                    continue
                
                # Enhance metadata for markdown documents
                if doc.metadata.get('file_path'):
                    file_path = doc.metadata['file_path']
                    
                    # Try to get webpage URL from JSON file, fallback to filename
                    webpage_url = self._get_webpage_url_from_json(file_path)
                    print(f"webpage_url is {webpage_url}")
                    source = webpage_url if webpage_url else os.path.basename(file_path)
                    
                    doc.metadata['source'] = source
                    doc.metadata['doc_id'] = hashlib.md5(file_path.encode()).hexdigest()
                    try:
                        file_stats = os.stat(file_path)
                        doc.metadata['file_size'] = file_stats.st_size
                        doc.metadata['modified_time'] = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    except:
                        pass
                
                markdown_documents.append(doc)
            
            documents = markdown_documents
        else:
            raise ValueError("Either file_path or directory_path must be provided")
        
        duration= time.time() - start_time
        self.logger.info(f"Loaded {len(documents)} markdown documents in {duration:.2f} seconds")
        
        return documents
    
    # Index Building Methods
    def build_index(self, documents: List[Document], show_progress: bool = True):
        """Build vector index from documents with performance monitoring"""
        start_time = time.time()
        
        # Parse documents into nodes
        parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        nodes = parser.get_nodes_from_documents(documents)
        
        # Create index
        self.index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=show_progress
        )
        
        # Create query engine with configuration
        self.query_engine = self._create_query_engine()
        
        duration = time.time() - start_time
        if self.monitor:
            self.monitor.log_embedding_time(duration, len(documents))
        
        self.logger.info(f"Index built with {len(nodes)} nodes in {duration:.2f} seconds")
    
    # Document Management Methods
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing index"""
        # Ensure all documents have proper metadata
        for doc in documents:
            if 'source' in doc.metadata:
                if '/' in doc.metadata['source'] or '\\' in doc.metadata['source']:
                    doc.metadata['source'] = os.path.basename(doc.metadata['source'])
            if 'doc_id' not in doc.metadata:
                doc.metadata['doc_id'] = hashlib.md5(
                    (doc.metadata.get('source', '') + doc.text[:100]).encode()
                ).hexdigest()
        
        if not self.index:
            self.build_index(documents)
        else:
            parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            new_nodes = parser.get_nodes_from_documents(documents)
            self.index.insert_nodes(new_nodes)
            self.logger.info(f"Added {len(new_nodes)} new nodes to index")
    
    def update_document(self, doc_id: str, new_text: str, new_metadata: Optional[Dict] = None):
        """Update existing document in the index"""
        # This is a placeholder - full implementation would require
        # tracking document chunks and updating them
        self.logger.info(f"Updating document {doc_id}")
        # For now, we'll delete and re-add
        self.delete_documents(doc_id=doc_id)
        
        doc = Document(
            text=new_text,
            metadata=new_metadata or {"doc_id": doc_id}
        )
        self.add_documents([doc])
    
    def delete_documents(self, source_filter: Optional[str] = None, doc_id: Optional[str] = None):
        """Delete documents by source name or document ID"""
        if source_filter:
            # Get all documents matching the source
            results = self.collection.get(
                where={"source": source_filter}
            )
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                self.logger.info(f"Deleted {len(results['ids'])} documents with source: {source_filter}")
        elif doc_id:
            # Delete by document ID
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                self.logger.info(f"Deleted document with ID: {doc_id}")
    
    def list_sources(self) -> List[Dict[str, Any]]:
        """List all unique document sources with metadata"""
        # Get all documents from collection
        results = self.collection.get()
        
        sources = {}
        for i, metadata in enumerate(results.get('metadatas', [])):
            if metadata and 'source' in metadata:
                source = metadata['source']
                if source not in sources:
                    sources[source] = {
                        'source': source,
                        'count': 0,
                        'total_size': 0,
                        'last_modified': None
                    }
                sources[source]['count'] += 1
                if 'file_size' in metadata:
                    sources[source]['total_size'] += metadata.get('file_size', 0)
                if 'modified_time' in metadata:
                    sources[source]['last_modified'] = metadata['modified_time']
        
        return list(sources.values())
    
    # Query Methods
    def query(self, question: str, top_k: Optional[int] = None) -> str:
        """Query the RAG system with caching support"""
        if not self.query_engine:
            raise ValueError("Index not built. Call build_index() first.")
        
        top_k = top_k or self.config.similarity_top_k
        
        # Check cache
        if self.query_cache:
            cached_result = self.query_cache.get(question, top_k)
            if cached_result:
                if self.monitor:
                    self.monitor.log_cache_hit()
                self.logger.debug(f"Cache hit for query: {question[:50]}...")
                return cached_result
            else:
                if self.monitor:
                    self.monitor.log_cache_miss()
        
        # Execute query
        start_time = time.time()
        
        # Update query engine if top_k is different
        if top_k != self.config.similarity_top_k:
            self.query_engine = self._create_query_engine(similarity_top_k=top_k)
        
        response = self.query_engine.query(question)
        formatted_response = self._format_response_with_references(response)
        
        duration = time.time() - start_time
        if self.monitor:
            self.monitor.log_query_time(duration)
        
        # Cache result
        if self.query_cache:
            self.query_cache.set(question, top_k, formatted_response)
        
        return formatted_response
    
    def query_stream(self, question: str, top_k: Optional[int] = None):
        """Query the RAG system with streaming response"""
        if not self.query_engine:
            raise ValueError("Index not built. Call build_index() first.")
        
        top_k = top_k or self.config.similarity_top_k
        
        # Check cache first
        if self.query_cache:
            cached_result = self.query_cache.get(question, top_k)
            if cached_result:
                if self.monitor:
                    self.monitor.log_cache_hit()
                # Yield cached result as single chunk
                yield cached_result
                return
            else:
                if self.monitor:
                    self.monitor.log_cache_miss()
        
        # Create streaming query engine
        streaming_query_engine = self._create_query_engine(similarity_top_k=top_k, streaming=True)
        
        # Execute streaming query
        start_time = time.time()
        response = streaming_query_engine.query(question)
        
        # Accumulate the full response for caching and formatting
        full_response = ""
        
        # Stream the response
        for token in response.response_gen:
            full_response += token
            yield token
        
        # Log performance and cache result
        duration = time.time() - start_time
        if self.monitor:
            self.monitor.log_query_time(duration)
        
        # Format and cache the complete response
        if self.query_cache:
            # Create a mock response object for formatting
            class MockResponse:
                def __init__(self, text, source_nodes):
                    self.response = text
                    self.source_nodes = source_nodes
                def __str__(self):
                    return self.response
            
            mock_response = MockResponse(full_response, response.source_nodes)
            formatted_response = self._format_response_with_references(mock_response)
            self.query_cache.set(question, top_k, formatted_response)
    
    def query_with_details(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Query with detailed source information"""
        if not self.query_engine:
            raise ValueError("Index not built. Call build_index() first.")
        
        top_k = top_k or self.config.similarity_top_k
        
        start_time = time.time()
        
        # Update query engine if needed
        if top_k != self.config.similarity_top_k:
            self.query_engine = self._create_query_engine(similarity_top_k=top_k)
        
        response = self.query_engine.query(question)
        
        # Extract detailed source information
        sources_info = []
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        for i, node in enumerate(source_nodes, 1):
            source_info = {
                'rank': i,
                'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                'full_text': node.text,
                'metadata': node.metadata if hasattr(node, 'metadata') else {},
                'score': getattr(node, 'score', None)
            }
            sources_info.append(source_info)
        
        duration = time.time() - start_time
        
        return {
            'answer': str(response),
            'formatted_answer': self._format_response_with_references(response),
            'sources': sources_info,
            'question': question,
            'query_time': duration
        }
    
    def _format_response_with_references(self, response) -> str:
        """Format the response to include metadata references"""
        answer = str(response)
        
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        if not source_nodes:
            return answer
        
        sources = set()
        for node in source_nodes:
            if hasattr(node, 'metadata') and node.metadata:
                source = node.metadata.get('source', 'Unknown')
                if '/' in source or '\\' in source:
                    source = source.split('/')[-1].split('\\')[-1]
                sources.add(source)
        
        if sources:
            references = ", ".join(sorted(sources))
            formatted_response = f"{answer}\n\n**Sources:** {references}"
        else:
            formatted_response = answer
            
        return formatted_response
    
    # Backup and Recovery Methods
    def export_collection(self, export_path: str):
        """Export collection data for backup"""
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export collection data
        results = self.collection.get()
        
        # Save to pickle file
        backup_data = {
            'collection_name': self.config.collection_name,
            'documents': results.get('documents', []),
            'metadatas': results.get('metadatas', []),
            'ids': results.get('ids', []),
            'embeddings': results.get('embeddings', []),
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config)
        }
        
        backup_file = export_dir / f"{self.config.collection_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(backup_file, 'wb') as f:
            pickle.dump(backup_data, f)
        
        self.logger.info(f"Exported collection to {backup_file}")
        return str(backup_file)
    
    def import_collection(self, import_path: str, collection_name: Optional[str] = None):
        """Import collection from backup"""
        with open(import_path, 'rb') as f:
            backup_data = pickle.load(f)
        
        collection_name = collection_name or backup_data['collection_name']
        
        # Create new collection
        new_collection = self.chroma_client.create_collection(
            name=f"{collection_name}_imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Add data to collection
        if backup_data['ids']:
            new_collection.add(
                ids=backup_data['ids'],
                documents=backup_data.get('documents'),
                metadatas=backup_data.get('metadatas'),
                embeddings=backup_data.get('embeddings')
            )
        
        self.logger.info(f"Imported {len(backup_data['ids'])} documents from backup")
        return new_collection.name
    
    # Performance and Monitoring Methods
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if not self.monitor:
            return {"monitoring_enabled": False}
        
        summary = self.monitor.get_summary()
        summary['collection_stats'] = self.get_collection_stats()
        return summary
    
    def clear_cache(self):
        """Clear query cache"""
        if self.query_cache:
            self.query_cache.clear()
            self.logger.info("Query cache cleared")
    
    def optimize_index(self):
        """Optimize index for better performance"""
        # This could include operations like:
        # - Reindexing with better parameters
        # - Removing duplicate chunks
        # - Updating embeddings with newer model
        self.logger.info("Index optimization completed")


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = RAGConfig(
        embed_batch_size=8,
        chunk_size=384,
        similarity_top_k=5,
        enable_monitoring=True,
        query_cache_ttl=1800
    )
    
    # Save configuration example
    config.save_to_file("rag_config.json")
    
    # Initialize enhanced chatbot
    chatbot = EnhancedRAGChatbot(config)
    
    # Example: List collections
    print("Available collections:", chatbot.list_collections())
    
    # Example: Load documents from directory
    if os.path.exists("sample_docs"):
        documents = chatbot.load_documents(directory_path="sample_docs")
        chatbot.build_index(documents)
        
        # Get collection stats
        print("\nCollection stats:", chatbot.get_collection_stats())
        
        # List sources
        print("\nDocument sources:")
        for source in chatbot.list_sources():
            print(f"  - {source['source']}: {source['count']} chunks")
        
        # Test queries
        test_query = "What is machine learning?"
        print(f"\nQuery: {test_query}")
        result = chatbot.query(test_query)
        print(f"Result: {result}")
        
        # Test detailed query
        detailed = chatbot.query_with_details(test_query, top_k=3)
        print(f"\nQuery time: {detailed['query_time']:.3f} seconds")
        
        # Export collection
        backup_path = chatbot.export_collection("./backups")
        print(f"\nBackup created: {backup_path}")
        
        # Performance summary
        print("\nPerformance Summary:")
        print(json.dumps(chatbot.get_performance_summary(), indent=2))
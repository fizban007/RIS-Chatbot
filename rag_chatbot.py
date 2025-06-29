import os
from typing import List, Optional
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings


class LocalRAGChatbot:
    def __init__(
        self,
        embed_base_url: str = "http://localhost:8000/v1",
        llm_base_url: str = "http://localhost:8000/v1",
        embed_model: str = "all-minilm-l6-v2-embedding",
        llm_model: str = "mistral-small-3.2-24b",
        collection_name: str = "rag_collection",
        persist_dir: str = "./chroma_db"
    ):
        # Configure embedding model (using OpenAI-compatible endpoint)
        self.embed_model = OpenAILikeEmbedding(
            model_name=embed_model,
            # model_name=
            api_base=embed_base_url,
            api_key="dummy",  # Required but not used for local servers
            embed_batch_size=32
        )
        
        # Configure LLM (using OpenAI-compatible endpoint)
        self.llm = OpenAILike(
            model=llm_model,
            api_base=llm_base_url,
            api_key="dummy",  # Required but not used for local servers
            context_window=32768,
            is_chat_model=True,
        )
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # Initialize Chroma client and collection
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        
        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        
        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        self.index = None
        self.query_engine = None
    
    def load_documents(self, file_path: Optional[str] = None, directory_path: Optional[str] = None) -> List[Document]:
        """Load documents from file or directory"""
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Use filename as the source in metadata
            filename = os.path.basename(file_path)
            documents = [Document(text=text, metadata={"source": filename})]
        elif directory_path:
            reader = SimpleDirectoryReader(directory_path)
            documents = reader.load_data()
            # Update source to be just the filename for each document
            for doc in documents:
                if doc.metadata.get('file_path'):
                    doc.metadata['source'] = os.path.basename(doc.metadata['file_path'])
                elif doc.metadata.get('source'):
                    doc.metadata['source'] = os.path.basename(doc.metadata['source'])
        else:
            raise ValueError("Either file_path or directory_path must be provided")
        
        return documents
    
    def build_index(self, documents: List[Document]):
        """Build vector index from documents"""
        # Parse documents into nodes
        parser = SentenceSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap
        )
        nodes = parser.get_nodes_from_documents(documents)
        
        # Create index
        self.index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=True
        )
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            streaming=False
        )
        
        print(f"Index built with {len(nodes)} nodes")
    
    def query(self, question: str) -> str:
        """Query the RAG system and return response with metadata references"""
        if not self.query_engine:
            raise ValueError("Index not built. Call build_index() first.")
        
        response = self.query_engine.query(question)
        
        # Format response with metadata references
        formatted_response = self._format_response_with_references(response)
        return formatted_response
    
    def _format_response_with_references(self, response) -> str:
        """Format the response to include metadata references"""
        answer = str(response)
        
        # Get source nodes from the response
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        if not source_nodes:
            return answer
        
        # Extract unique sources from metadata
        sources = set()
        for node in source_nodes:
            if hasattr(node, 'metadata') and node.metadata:
                # Get source information from metadata
                source = node.metadata.get('source', 'Unknown')
                # If it's a file path, get just the filename
                if '/' in source or '\\' in source:
                    source = source.split('/')[-1].split('\\')[-1]
                sources.add(source)
        
        # Add references to the response
        if sources:
            references = ", ".join(sorted(sources))
            formatted_response = f"{answer}\n\n**Sources:** {references}"
        else:
            formatted_response = answer
            
        return formatted_response
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing index"""
        # Ensure all documents have just filename as source in metadata
        for doc in documents:
            if 'source' in doc.metadata:
                # If source contains path separators, extract just the filename
                if '/' in doc.metadata['source'] or '\\' in doc.metadata['source']:
                    doc.metadata['source'] = os.path.basename(doc.metadata['source'])
        
        if not self.index:
            self.build_index(documents)
        else:
            # Parse new documents
            parser = SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap
            )
            new_nodes = parser.get_nodes_from_documents(documents)
            
            # Insert into index
            self.index.insert_nodes(new_nodes)
            print(f"Added {len(new_nodes)} new nodes to index")

    def query_with_details(self, question: str) -> dict:
        """Query the RAG system and return detailed response with source information"""
        if not self.query_engine:
            raise ValueError("Index not built. Call build_index() first.")
        
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
        
        return {
            'answer': str(response),
            'formatted_answer': self._format_response_with_references(response),
            'sources': sources_info,
            'question': question
        }


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    chatbot = LocalRAGChatbot()
    
    # Create sample documents with more detailed metadata
    sample_docs = [
        Document(
            text="The capital of France is Paris. Paris is known for the Eiffel Tower, the Louvre Museum, and its beautiful architecture. The city is located on the Seine River and has a population of over 2 million people.",
            metadata={"source": "geography_facts.txt", "topic": "geography", "country": "France"}
        ),
        Document(
            text="Python is a high-level programming language. It emphasizes code readability and has a simple syntax that makes it easy to learn. Python was created by Guido van Rossum and first released in 1991.",
            metadata={"source": "programming_guide.pdf", "topic": "programming", "language": "Python"}
        ),
        Document(
            text="Machine learning is a subset of artificial intelligence. It enables systems to learn and improve from experience without being explicitly programmed. Common algorithms include neural networks, decision trees, and support vector machines.",
            metadata={"source": "ai_handbook.docx", "topic": "artificial intelligence", "category": "machine learning"}
        )
    ]
    
    # Build index
    print("Building index...")
    chatbot.build_index(sample_docs)
    
    # Test queries with the new functionality
    test_queries = [
        "What is the capital of France?",
        "Tell me about Python programming",
        "What is machine learning?"
    ]
    
    print("\n" + "="*60)
    print("TESTING BASIC QUERY WITH REFERENCES")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuestion: {query}")
        answer = chatbot.query(query)
        print(f"Answer: {answer}")
    
    print("\n" + "="*60)
    print("TESTING DETAILED QUERY WITH FULL SOURCE INFO")
    print("="*60)
    
    # Test detailed query
    detailed_result = chatbot.query_with_details("What is the capital of France?")
    print(f"\nQuestion: {detailed_result['question']}")
    print(f"Answer: {detailed_result['answer']}")
    print(f"\nFormatted Answer with References:\n{detailed_result['formatted_answer']}")
    print(f"\nDetailed Source Information:")
    for source in detailed_result['sources']:
        print(f"  Source {source['rank']}:")
        print(f"    Text: {source['text']}")
        print(f"    Metadata: {source['metadata']}")
        if source['score']:
            print(f"    Relevance Score: {source['score']}")
        print()
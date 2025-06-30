from rag_chatbot import EnhancedRAGChatbot, RAGConfig
from llama_index.core import Document
import os

def main():
    # Initialize the chatbot with custom configuration
    config = RAGConfig(
        embed_base_url="http://localhost:8000/v1",     # Embedding server
        llm_base_url="http://localhost:8000/v1",        # LLM server
        embed_model="all-minilm-l6-v2-embedding",      # Your embedding model
        llm_model="mistral-small-3.2-24b",             # Your LLM model
        collection_name="my_knowledge_base",
        persist_dir="./chroma_db",
        chunk_size=512,
        chunk_overlap=50,
        similarity_top_k=3,
        embed_batch_size=64,
        query_cache_ttl=1800,  # 30 minutes
        enable_monitoring=True
    )
    
    chatbot = EnhancedRAGChatbot(config)
    
    # Example 1: Load documents from text
    print("=== Example 1: Loading documents from text ===")
    documents = [
        Document(
            text="""Artificial Intelligence (AI) is the simulation of human intelligence in machines. 
            Key areas include machine learning, natural language processing, computer vision, and robotics. 
            AI systems can perform tasks like speech recognition, decision-making, and visual perception.""",
            metadata={"source": "ai_basics.txt", "category": "technology"}
        ),
        Document(
            text="""Climate change refers to long-term shifts in global temperatures and weather patterns. 
            Human activities, primarily fossil fuel burning, have been the main driver since the 1800s. 
            Effects include rising sea levels, extreme weather events, and ecosystem disruption.""",
            metadata={"source": "climate_info.txt", "category": "environment"}
        ),
        Document(
            text="""The Python programming language was created by Guido van Rossum and released in 1991. 
            Python's design philosophy emphasizes code readability with significant whitespace. 
            It supports multiple programming paradigms including procedural, object-oriented, and functional programming.""",
            metadata={"source": "python_history.txt", "category": "programming"}
        )
    ]
    
    chatbot.build_index(documents)
    
    # Show collection stats
    stats = chatbot.get_collection_stats()
    print(f"\nCollection '{stats['name']}' has {stats['count']} chunks")
    
    # Example 2: Query the knowledge base
    print("\n=== Example 2: Querying the knowledge base ===")
    queries = [
        "What are the main areas of AI?",
        "Who created Python and when?",
        "What causes climate change?",
        "Tell me about the effects of global warming"
    ]
    
    for query in queries:
        print(f"\nQ: {query}")
        response = chatbot.query(query)
        print(f"A: {response}")
    
    # Example 3: Load documents from directory
    print("\n=== Example 3: Loading from directory ===")
    # Create a sample directory with files
    os.makedirs("sample_docs", exist_ok=True)
    
    with open("sample_docs/quantum_computing.txt", "w") as f:
        f.write("""Quantum computing uses quantum mechanical phenomena like superposition and entanglement. 
        Unlike classical bits that are 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously. 
        This allows quantum computers to solve certain problems exponentially faster than classical computers.""")
    
    with open("sample_docs/blockchain.txt", "w") as f:
        f.write("""Blockchain is a distributed ledger technology that maintains a secure and decentralized record. 
        Each block contains a cryptographic hash of the previous block, timestamp, and transaction data. 
        Bitcoin was the first application of blockchain technology, launched in 2009.""")
    
    # Load and add new documents
    new_docs = chatbot.load_documents(directory_path="sample_docs")
    chatbot.add_documents(new_docs)
    
    # List all document sources
    print("\nDocument sources in collection:")
    for source in chatbot.list_sources():
        print(f"  - {source['source']}: {source['count']} chunks")
    
    # Query about new content
    print("\nQ: What are qubits?")
    print(f"A: {chatbot.query('What are qubits?')}")
    
    print("\nQ: When was Bitcoin launched?")
    print(f"A: {chatbot.query('When was Bitcoin launched?')}")
    
    # Example 4: Advanced querying with details
    print("\n=== Example 4: Detailed query results ===")
    detailed_result = chatbot.query_with_details(
        "Compare quantum computing with classical computing",
        top_k=4
    )
    
    print(f"Question: {detailed_result['question']}")
    print(f"Query time: {detailed_result['query_time']:.3f} seconds")
    print(f"\nAnswer: {detailed_result['answer']}")
    print(f"\nSources used:")
    for source in detailed_result['sources']:
        print(f"  {source['rank']}. {source['metadata'].get('source', 'Unknown')}")
        if source.get('score'):
            print(f"     Score: {source['score']:.4f}")
    
    # Example 5: Performance monitoring
    print("\n=== Example 5: Performance Summary ===")
    perf = chatbot.get_performance_summary()
    print(f"Total queries: {perf['total_queries']}")
    print(f"Average query time: {perf['avg_query_time']:.3f} seconds")
    print(f"Cache hit rate: {perf['cache_hit_rate']:.2%}")
    
    # Example 6: Collection management
    print("\n=== Example 6: Collection Management ===")
    
    # List all collections
    collections = chatbot.list_collections()
    print(f"Available collections: {collections}")
    
    # Create a backup
    print("\nCreating backup...")
    backup_path = chatbot.export_collection("./backups")
    print(f"Backup saved to: {backup_path}")
    
    # Example 7: Document management
    print("\n=== Example 7: Document Management ===")
    
    # Delete a specific document source
    print("Deleting blockchain.txt...")
    chatbot.delete_documents(source_filter="blockchain.txt")
    
    # Verify deletion
    remaining_sources = [s['source'] for s in chatbot.list_sources()]
    print(f"Remaining sources: {remaining_sources}")
    
    # Test query after deletion
    print("\nQ: When was Bitcoin launched? (after deleting blockchain.txt)")
    print(f"A: {chatbot.query('When was Bitcoin launched?')}")
    
    # Save configuration for future use
    print("\n=== Saving Configuration ===")
    config.save_to_file("my_rag_config.json")
    print("Configuration saved to my_rag_config.json")
    
    # Clean up
    os.remove("my_rag_config.json")
    import shutil
    if os.path.exists("backups"):
        shutil.rmtree("backups")


if __name__ == "__main__":
    main()
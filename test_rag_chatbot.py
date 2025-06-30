#!/usr/bin/env python3
"""
Test script demonstrating all features of the Enhanced RAG Chatbot
"""

import os
import time
import json
from rag_chatbot import EnhancedRAGChatbot, RAGConfig
from llama_index.core import Document


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}\n")


def test_configuration():
    """Test configuration management"""
    print_section("CONFIGURATION MANAGEMENT")
    
    # Create custom configuration
    config = RAGConfig(
        embed_batch_size=128,
        chunk_size=384,
        chunk_overlap=75,
        similarity_top_k=5,
        enable_monitoring=True,
        query_cache_ttl=1800,
        query_cache_max_size=500
    )
    
    # Save configuration
    config_path = "test_rag_config.json"
    config.save_to_file(config_path)
    print(f"✓ Configuration saved to {config_path}")
    
    # Load configuration
    loaded_config = RAGConfig.from_file(config_path)
    print(f"✓ Configuration loaded successfully")
    print(f"  - Chunk size: {loaded_config.chunk_size}")
    print(f"  - Embed batch size: {loaded_config.embed_batch_size}")
    print(f"  - Cache TTL: {loaded_config.query_cache_ttl} seconds")
    
    # Clean up
    os.remove(config_path)
    
    return loaded_config


def test_collection_management(chatbot):
    """Test collection management features"""
    print_section("COLLECTION MANAGEMENT")
    
    # List all collections
    collections = chatbot.list_collections()
    print(f"✓ Found {len(collections)} collections:")
    for col in collections:
        print(f"  - {col}")
    
    # Get current collection stats
    stats = chatbot.get_collection_stats()
    print(f"\n✓ Current collection stats:")
    print(f"  - Name: {stats['name']}")
    print(f"  - Document count: {stats['count']}")
    
    # Create a test collection
    test_collection_name = "test_collection_temp"
    chatbot.switch_collection(test_collection_name)
    print(f"\n✓ Switched to new collection: {test_collection_name}")
    
    return test_collection_name


def test_document_management(chatbot):
    """Test document loading and management"""
    print_section("DOCUMENT MANAGEMENT")
    
    # Create sample documents
    sample_docs = [
        Document(
            text="Artificial Intelligence (AI) is the simulation of human intelligence in machines. "
                 "These machines are programmed to think and learn like humans. AI systems can perform "
                 "tasks such as visual perception, speech recognition, decision-making, and language translation.",
            metadata={
                "source": "ai_basics.txt",
                "topic": "AI",
                "category": "introduction"
            }
        ),
        Document(
            text="Machine Learning is a subset of AI that enables systems to learn and improve from "
                 "experience without being explicitly programmed. It focuses on developing algorithms "
                 "that can access data and use it to learn for themselves. Common types include "
                 "supervised learning, unsupervised learning, and reinforcement learning.",
            metadata={
                "source": "ml_guide.txt",
                "topic": "Machine Learning",
                "category": "fundamentals"
            }
        ),
        Document(
            text="Deep Learning is a subset of machine learning that uses neural networks with multiple "
                 "layers (deep neural networks). These networks attempt to simulate the behavior of the "
                 "human brain—albeit far from matching its ability—allowing it to 'learn' from large "
                 "amounts of data. Deep learning drives many AI applications.",
            metadata={
                "source": "deep_learning.txt",
                "topic": "Deep Learning",
                "category": "advanced"
            }
        ),
        Document(
            text="Natural Language Processing (NLP) is a branch of AI that helps computers understand, "
                 "interpret and manipulate human language. NLP draws from many disciplines, including "
                 "computer science and computational linguistics. Applications include chatbots, "
                 "translation services, and sentiment analysis.",
            metadata={
                "source": "nlp_overview.txt",
                "topic": "NLP",
                "category": "applications"
            }
        )
    ]
    
    # Build index with documents
    print("✓ Building index with sample documents...")
    start_time = time.time()
    chatbot.build_index(sample_docs)
    print(f"  - Index built in {time.time() - start_time:.2f} seconds")
    
    # List document sources
    sources = chatbot.list_sources()
    print(f"\n✓ Document sources in index:")
    for source in sources:
        print(f"  - {source['source']}: {source['count']} chunks")
    
    # Add additional documents
    additional_doc = Document(
        text="Computer Vision is a field of AI that trains computers to interpret and understand "
             "the visual world. Using digital images from cameras and videos and deep learning models, "
             "machines can accurately identify and classify objects — and then react to what they 'see'.",
        metadata={
            "source": "computer_vision.txt",
            "topic": "Computer Vision",
            "category": "applications"
        }
    )
    
    print(f"\n✓ Adding additional document...")
    chatbot.add_documents([additional_doc])
    
    # Test document deletion
    print(f"\n✓ Testing document deletion...")
    chatbot.delete_documents(source_filter="computer_vision.txt")
    print("  - Document deleted successfully")
    
    return sample_docs


def test_query_functionality(chatbot):
    """Test query functionality with caching"""
    print_section("QUERY FUNCTIONALITY")
    
    test_queries = [
        "What is artificial intelligence?",
        "Explain the difference between machine learning and deep learning",
        "What are the applications of NLP?",
        "How does supervised learning work?",
        "What is artificial intelligence?"  # Duplicate to test cache
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        # Time the query
        start_time = time.time()
        result = chatbot.query(query, top_k=3)
        query_time = time.time() - start_time
        
        print(f"   Response time: {query_time:.3f} seconds")
        print(f"   Answer: {result[:200]}...")
        
        # Small delay between queries
        time.sleep(0.1)
    
    # Test detailed query
    print_section("DETAILED QUERY RESULTS")
    
    detailed_result = chatbot.query_with_details(
        "What is the relationship between AI, ML, and deep learning?",
        top_k=4
    )
    
    print(f"Question: {detailed_result['question']}")
    print(f"Query time: {detailed_result['query_time']:.3f} seconds")
    print(f"\nAnswer: {detailed_result['answer']}")
    print(f"\nSources used:")
    for source in detailed_result['sources']:
        print(f"  {source['rank']}. {source['metadata'].get('source', 'Unknown')}")
        print(f"     Score: {source.get('score', 'N/A')}")
        print(f"     Preview: {source['text'][:100]}...")


def test_backup_restore(chatbot):
    """Test backup and restore functionality"""
    print_section("BACKUP AND RESTORE")
    
    # Create backup directory
    backup_dir = "./test_backups"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Export collection
    print("✓ Exporting collection...")
    backup_path = chatbot.export_collection(backup_dir)
    print(f"  - Backup saved to: {backup_path}")
    
    # Get file size
    backup_size = os.path.getsize(backup_path) / 1024  # KB
    print(f"  - Backup size: {backup_size:.2f} KB")
    
    # Import to new collection
    print("\n✓ Importing from backup...")
    new_collection_name = chatbot.import_collection(backup_path)
    print(f"  - Imported to new collection: {new_collection_name}")
    
    # Clean up
    import shutil
    shutil.rmtree(backup_dir)
    
    return backup_path


def test_performance_monitoring(chatbot):
    """Test performance monitoring and optimization"""
    print_section("PERFORMANCE MONITORING")
    
    # Get performance summary
    performance = chatbot.get_performance_summary()
    
    print("✓ Performance Metrics:")
    print(f"  - Total queries: {performance.get('total_queries', 0)}")
    print(f"  - Average query time: {performance.get('avg_query_time', 0):.3f} seconds")
    print(f"  - Cache hit rate: {performance.get('cache_hit_rate', 0):.2%}")
    print(f"  - Cache hits: {performance.get('cache_hits', 0)}")
    print(f"  - Cache misses: {performance.get('cache_misses', 0)}")
    
    # Clear cache
    print("\n✓ Clearing query cache...")
    chatbot.clear_cache()
    print("  - Cache cleared successfully")
    
    # Optimize index (placeholder)
    print("\n✓ Optimizing index...")
    chatbot.optimize_index()
    print("  - Index optimization completed")


def test_advanced_features(chatbot):
    """Test advanced features"""
    print_section("ADVANCED FEATURES")
    
    # Test different chunk sizes by creating new instance
    print("✓ Testing different configurations:")
    
    configs = [
        {"chunk_size": 256, "chunk_overlap": 50},
        {"chunk_size": 512, "chunk_overlap": 100},
        {"chunk_size": 1024, "chunk_overlap": 200}
    ]
    
    test_doc = Document(
        text="This is a test document. " * 100,  # Long document
        metadata={"source": "test.txt"}
    )
    
    for cfg in configs:
        config = RAGConfig(**cfg)
        test_chatbot = EnhancedRAGChatbot(config)
        test_chatbot.build_index([test_doc])
        stats = test_chatbot.get_collection_stats()
        print(f"  - Chunk size {cfg['chunk_size']}: {stats['count']} chunks created")
    
    # Test batch processing
    print("\n✓ Testing batch document processing:")
    batch_docs = []
    for i in range(10):
        batch_docs.append(Document(
            text=f"Document {i}: This is test content for batch processing. " * 20,
            metadata={"source": f"batch_doc_{i}.txt", "batch_id": "test_batch"}
        ))
    
    start_time = time.time()
    chatbot.add_documents(batch_docs)
    batch_time = time.time() - start_time
    print(f"  - Processed {len(batch_docs)} documents in {batch_time:.2f} seconds")
    print(f"  - Average time per document: {batch_time/len(batch_docs):.3f} seconds")


def cleanup(chatbot, test_collection_name):
    """Clean up test collections"""
    print_section("CLEANUP")
    
    # Delete test collection
    try:
        chatbot.delete_collection(test_collection_name)
        print(f"✓ Deleted test collection: {test_collection_name}")
    except:
        pass
    
    # Delete any imported collections
    collections = chatbot.list_collections()
    for col in collections:
        if "imported" in col or "test" in col:
            try:
                chatbot.delete_collection(col)
                print(f"✓ Deleted collection: {col}")
            except:
                pass


def main():
    """Run all tests"""
    print_section("ENHANCED RAG CHATBOT TEST SUITE")
    
    try:
        # Test configuration
        config = test_configuration()
        
        # Initialize chatbot
        chatbot = EnhancedRAGChatbot(config)
        
        # Test collection management
        test_collection_name = test_collection_management(chatbot)
        
        # Test document management
        test_document_management(chatbot)
        
        # Test query functionality
        test_query_functionality(chatbot)
        
        # Test backup and restore
        test_backup_restore(chatbot)
        
        # Test performance monitoring
        test_performance_monitoring(chatbot)
        
        # Test advanced features
        test_advanced_features(chatbot)
        
        # Cleanup
        cleanup(chatbot, test_collection_name)
        
        print_section("ALL TESTS COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
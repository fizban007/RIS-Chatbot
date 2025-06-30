#!/usr/bin/env python3
"""
Command-line management tool for RAG Chatbot

Usage:
    python manage_rag.py load-docs --dir ./documents
    python manage_rag.py load-docs --file document.txt
    python manage_rag.py list-sources
    python manage_rag.py delete-source --name document.txt
    python manage_rag.py stats
    python manage_rag.py clear-collection
    python manage_rag.py backup --output ./backup.pkl
    python manage_rag.py restore --input ./backup.pkl
"""

import argparse
import os
import sys
from rag_chatbot import EnhancedRAGChatbot, RAGConfig
from llama_index.core import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config():
    """Load configuration from environment or defaults"""
    return RAGConfig(
        embed_base_url=os.getenv("EMBED_BASE_URL", "http://localhost:8001/v1"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        embed_model=os.getenv("EMBED_MODEL", "all-minilm-l6-v2-embedding"),
        llm_model=os.getenv("LLM_MODEL", "mistral-small-3.2-24b"),
        system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant. Answer questions based on the provided context. Be concise and accurate."),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        hf_api_key=os.getenv("HF_API_KEY"),
        collection_name=os.getenv("COLLECTION_NAME", "rag_collection"),
        persist_dir=os.getenv("PERSIST_DIR", "./chroma_db"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        enable_monitoring=True
    )

def load_documents(args):
    """Load documents from file or directory"""
    config = load_config()
    chatbot = EnhancedRAGChatbot(config)
    print(f"Chunk size: {config.chunk_size}")
    print(f"Chunk overlap: {config.chunk_overlap}")
    
    if args.dir:
        print(f"Loading documents from directory: {args.dir}")
        docs = chatbot.load_documents(directory_path=args.dir)
    elif args.file:
        print(f"Loading document from file: {args.file}")
        docs = chatbot.load_documents(file_path=args.file)
    else:
        print("Error: Must specify either --dir or --file")
        return
    
    # Check if index exists
    stats = chatbot.get_collection_stats()
    if stats['count'] == 0:
        print(f"Building new index with {len(docs)} documents...")
        chatbot.build_index(docs)
    else:
        print(f"Adding {len(docs)} documents to existing index...")
        chatbot.add_documents(docs)
    
    print("✅ Documents loaded successfully!")
    
    # Show updated stats
    stats = chatbot.get_collection_stats()
    print(f"Total documents in collection: {stats['count']}")

def list_sources(args):
    """List all document sources"""
    config = load_config()
    chatbot = EnhancedRAGChatbot(config)
    
    sources = chatbot.list_sources()
    if not sources:
        print("No documents found in collection.")
        return
    
    print("\nDocument sources:")
    print("-" * 50)
    for source in sources:
        print(f"📄 {source['source']:<30} ({source['count']} chunks)")
    print("-" * 50)
    print(f"Total: {len(sources)} sources")

def delete_source(args):
    """Delete a document source"""
    config = load_config()
    chatbot = EnhancedRAGChatbot(config)
    
    print(f"Deleting source: {args.name}")
    chatbot.delete_documents(source_filter=args.name)
    print("✅ Source deleted successfully!")
    
    # Show updated stats
    stats = chatbot.get_collection_stats()
    print(f"Remaining documents in collection: {stats['count']}")

def show_stats(args):
    """Show collection statistics"""
    config = load_config()
    chatbot = EnhancedRAGChatbot(config)
    
    stats = chatbot.get_collection_stats()
    print("\nCollection Statistics:")
    print("-" * 30)
    print(f"Collection name: {stats['name']}")
    print(f"Document chunks: {stats['count']}")
    
    # Get performance metrics if available
    perf = chatbot.get_performance_summary()
    if perf.get('monitoring_enabled', True):
        print(f"\nPerformance Metrics:")
        print(f"Total queries: {perf.get('total_queries', 0)}")
        print(f"Avg query time: {perf.get('avg_query_time', 0):.3f}s")
        print(f"Cache hit rate: {perf.get('cache_hit_rate', 0):.1%}")

def clear_collection(args):
    """Clear all documents from collection"""
    config = load_config()
    chatbot = EnhancedRAGChatbot(config)
    
    stats = chatbot.get_collection_stats()
    if stats['count'] == 0:
        print("Collection is already empty.")
        return
    
    # Confirm action
    response = input(f"⚠️  This will delete {stats['count']} documents. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Delete collection and recreate
    collection_name = config.collection_name
    chatbot.delete_collection(collection_name)
    chatbot.switch_collection(collection_name)
    print("✅ Collection cleared successfully!")

def backup_collection(args):
    """Create a backup of the collection"""
    config = load_config()
    chatbot = EnhancedRAGChatbot(config)
    
    print(f"Creating backup...")
    backup_path = chatbot.export_collection(os.path.dirname(args.output))
    
    # Rename to desired output path
    if args.output != backup_path:
        os.rename(backup_path, args.output)
        backup_path = args.output
    
    print(f"✅ Backup saved to: {backup_path}")

def restore_collection(args):
    """Restore collection from backup"""
    config = load_config()
    chatbot = EnhancedRAGChatbot(config)
    
    if not os.path.exists(args.input):
        print(f"Error: Backup file not found: {args.input}")
        return
    
    print(f"Restoring from backup: {args.input}")
    new_collection = chatbot.import_collection(args.input)
    print(f"✅ Restored to collection: {new_collection}")

def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot Management Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Load documents command
    load_parser = subparsers.add_parser('load-docs', help='Load documents into the collection')
    load_parser.add_argument('--dir', help='Directory path to load documents from')
    load_parser.add_argument('--file', help='Single file path to load')
    load_parser.set_defaults(func=load_documents)
    
    # List sources command
    list_parser = subparsers.add_parser('list-sources', help='List all document sources')
    list_parser.set_defaults(func=list_sources)
    
    # Delete source command
    delete_parser = subparsers.add_parser('delete-source', help='Delete a document source')
    delete_parser.add_argument('--name', required=True, help='Source name to delete')
    delete_parser.set_defaults(func=delete_source)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show collection statistics')
    stats_parser.set_defaults(func=show_stats)
    
    # Clear collection command
    clear_parser = subparsers.add_parser('clear-collection', help='Clear all documents from collection')
    clear_parser.set_defaults(func=clear_collection)
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create a backup of the collection')
    backup_parser.add_argument('--output', required=True, help='Output backup file path')
    backup_parser.set_defaults(func=backup_collection)
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore collection from backup')
    restore_parser.add_argument('--input', required=True, help='Input backup file path')
    restore_parser.set_defaults(func=restore_collection)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute the command
    args.func(args)

if __name__ == "__main__":
    main()
from rag_chatbot import LocalRAGChatbot
from llama_index.core import Document
import os

def main():
    # Initialize the chatbot
    # Make sure your local servers are running on these ports
    chatbot = LocalRAGChatbot(
        embed_base_url="http://localhost:8000/v1",     # e.g., vLLM or FastChat embedding server
        llm_base_url="http://localhost:8000/v1",        # e.g., vLLM or FastChat LLM server
        embed_model="all-minilm-l6-v2-embedding",                 # Your embedding model name
        llm_model="mistral-small-3.2-24b",  # Your LLM model name
        collection_name="my_knowledge_base",
        persist_dir="./chroma_db"
    )
    
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
            text="""The Python programming language was created by Guido van Rossum and released in 1881. 
            Python's design philosophy emphasizes code readability with significant whitespace. 
            It supports multiple programming paradigms including procedural, object-oriented, and functional programming.""",
            metadata={"source": "python_history.txt", "category": "programming"}
        )
    ]
    
    chatbot.build_index(documents)
    
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
    
    # Query about new content
    print("\nQ: What are qubits?")
    print(f"A: {chatbot.query('What are qubits?')}")
    
    print("\nQ: When was Bitcoin launched?")
    print(f"A: {chatbot.query('When was Bitcoin launched?')}")
    
    # Example 4: Advanced querying
    print("\n=== Example 4: Advanced queries ===")
    complex_queries = [
        "Compare quantum computing with classical computing",
        "How do blockchain and AI relate to modern technology?",
        "Explain the relationship between Python and machine learning"
    ]
    
    for query in complex_queries:
        print(f"\nQ: {query}")
        response = chatbot.query(query)
        print(f"A: {response}")


if __name__ == "__main__":
    main()
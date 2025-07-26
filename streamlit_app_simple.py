import streamlit as st
from rag_chatbot import EnhancedRAGChatbot, RAGConfig
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for better chat appearance
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    div[data-testid="stVerticalBlock"] div:has(div.chat-message) {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    # Load configuration from environment or use defaults
    config = RAGConfig(
        embed_base_url=os.getenv("EMBED_BASE_URL", "http://localhost:8001/v1"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        embed_model=os.getenv("EMBED_MODEL", "all-minilm-l6-v2-embedding"),
        llm_model=os.getenv("LLM_MODEL", "mistral-small-3.2-24b"),
        system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant. Answer questions based on the provided context. Be concise and accurate."),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        deepinfra_api_key=os.getenv("DEEPINFRA_API_KEY"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        hf_api_key=os.getenv("HF_API_KEY"),
        collection_name=os.getenv("COLLECTION_NAME", "rag_collection"),
        persist_dir=os.getenv("PERSIST_DIR", "./chroma_db"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "3")),
        query_cache_ttl=int(os.getenv("QUERY_CACHE_TTL", "1800")),
        enable_monitoring=True
    )
    
    st.session_state.chatbot = EnhancedRAGChatbot(config)
    st.session_state.messages = []
    
    # Check if collection has documents and build index if needed
    stats = st.session_state.chatbot.get_collection_stats()
    st.session_state.has_documents = stats['count'] > 0
    
    # Build index from existing collection if documents exist but index not built
    if st.session_state.has_documents and st.session_state.chatbot.index is None:
        with st.spinner("Building index from existing documents..."):
            # Create empty index from existing vector store
            from llama_index.core import VectorStoreIndex
            st.session_state.chatbot.index = VectorStoreIndex.from_vector_store(
                st.session_state.chatbot.vector_store,
                storage_context=st.session_state.chatbot.storage_context
            )
            st.session_state.chatbot.query_engine = st.session_state.chatbot._create_query_engine()

# Title
st.title("🤖 RAG Chatbot")

# Check if documents are loaded
if not st.session_state.has_documents:
    st.warning("⚠️ No documents found in the collection. Please load documents using the command line tools.")
    st.code("""
# Example: Load documents from directory
python -c "
from rag_chatbot import EnhancedRAGChatbot
chatbot = EnhancedRAGChatbot()
docs = chatbot.load_documents(directory_path='./your_documents')
chatbot.build_index(docs)
"
    """, language="bash")
    
    if st.button("🔄 Refresh"):
        st.rerun()
else:
    # Show document count
    stats = st.session_state.chatbot.get_collection_stats()
    st.caption(f"📚 Connected to collection with {stats['count']} document chunks")
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        processed_prompt = prompt
        if not any(cluster in prompt.lower() for cluster in ['compute1', 'compute2', 'lsf', 'slurm']):
            processed_prompt = f"{prompt} compute1 lsf scheduler"

        # Add user message
        st.session_state.messages.append({"role": "user", "content": processed_prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            # Create placeholder for streaming response
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in st.session_state.chatbot.query_stream(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")  # Add cursor
            
            # Remove cursor when done
            response_placeholder.markdown(full_response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.8em;'>Chat with your documents using local AI</div>",
    unsafe_allow_html=True
)
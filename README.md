# Local RAG Chatbot with LlamaIndex and Chroma

A minimal RAG (Retrieval-Augmented Generation) chatbot using locally hosted LLMs through OpenAI-compatible APIs.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start your local OpenAI-compatible servers:
   - **Embedding server**: Port 8001 (e.g., vLLM, FastChat, LocalAI)
   - **LLM server**: Port 8000

Example with vLLM:
```bash
# Terminal 1: Embedding server
python -m vllm.entrypoints.openai.api_server \
  --model nomic-ai/nomic-embed-text-v1.5 \
  --port 8001 \
  --served-model-name nomic-embed-text

# Terminal 2: LLM server  
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --port 8000
```

3. Run the examples:
```bash
python example_usage.py
```

## Usage

```python
from rag_chatbot import LocalRAGChatbot
from llama_index.core import Document

# Initialize chatbot
chatbot = LocalRAGChatbot(
    embed_base_url="http://localhost:8001/v1",
    llm_base_url="http://localhost:8000/v1",
    embed_model="nomic-embed-text",
    llm_model="meta-llama/Llama-3.2-3B-Instruct"
)

# Load documents
docs = [Document(text="Your content here", metadata={"source": "doc1"})]
chatbot.build_index(docs)

# Query
response = chatbot.query("Your question here")
print(response)
```

## Features

- Local LLM and embedding models via OpenAI-compatible API
- Persistent vector storage with ChromaDB
- Document chunking and metadata support
- Add documents incrementally
- Configurable chunk size and retrieval parameters
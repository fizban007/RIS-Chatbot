#!/bin/bash
# Single HF_HOME variable handles all Hugging Face caches
export HF_HOME="/storage2/fs1/dt-summer-corp/Active/common/projects/ai-on-washu-infrastructure/chatbot/ragbot-dev/cache"
export TORCH_HOME="/storage2/fs1/dt-summer-corp/Active/common/projects/ai-on-washu-infrastructure/chatbot/ragbot-dev/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="/storage2/fs1/dt-summer-corp/Active/common/projects/ai-on-washu-infrastructure/chatbot/ragbot-dev/cache/torch_compile"
export XDG_CACHE_HOME="/storage2/fs1/dt-summer-corp/Active/common/projects/ai-on-washu-infrastructure/chatbot/ragbot-dev/cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice &

STREAMLIT_SERVER_ADDRESS=0.0.0.0 streamlit run streamlit_app_simple.py &

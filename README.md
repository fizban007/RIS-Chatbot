# RIS-Bot

RIS-Bot is a chatbot tool for answering questions about the Research Infrastructure Services (RIS) HPC platform at Washignton University in St. Louis. RIS-Bot performs Retrieval Augmented Generation (RAG) on the documentation sourced from [WashU RIS Documentation](https://docs.ris.wustl.edu), with a self-contained pipeline for data retrieval, vector embedding and RAG generation.

# Deployment

## Mounting core libraries
RIS-bot requires certain core libraries such as CUDA. This installation guide will cover how to run the program on RIS where the appropriate versions of these libraries have already been installed. The user should install these libraries themselves before following next steps if deploying to a different platform.

```
export LSF_DOCKER_VOLUMES="/storage2/fs1/dt-summer-corp/Active/common/projects/ai-on-washu-infrastructure/chatbot/libs:/usr/local/modules /storage2/fs1/dt-summer-corp:/storage2/fs1/dt-summer-corp $HOME:$HOME"
```

## Mapping Ports
By default, the chat web interface will run on port `8501` `(0.0.0.0:8501)` on the Docker image, and we map this to port `8003` `(compute1-exec-xxx.ris.wustl.edu:8003)` for users to access.
 ```
 export LSF_DOCKER_PORTS='8003:8501'
 ```

## Job Submission
 Submit the job requesting 1 GPU. Use the ris_chatbot docker image.
 ```
 bsub -Is -G compute-artsci -q artsci-interactive -n 8 -R 'select[port8003=1]' -R 'gpuhost rusage[mem=120GB]' -gpu 'num=1' -a 'docker(fizban007/ris_chatbot)'  /usr/bin/bash
 ```

## Switch to working directory
The development version exists in `storage2` and can be accessed via:
```
cd /storage2/fs1/dt-summer-corp/Active/common/projects/ai-on-washu-infrastructure/chatbot/ragbot-dev
```

Otherwise, change to your working directory and clone the [RIS-Chatbot repository](https://github.com/Digital-Transformation-Summer-Corps/RIS-Chatbot):
```
cd YOUR_WORKING_DIR
git clone https://github.com/Digital-Transformation-Summer-Corps/RIS-Chatbot.git
cd RIS-Chatbot
```

## Environment Setup
Change `.env.example` to `.env` and change the settings if necessary:
```
cp .env.example .env
```

*cp does not cause any changes that Git tracks, so there will be no conflicts*

You would need to add your own Gemini API key for validation (QA generation & LLM-as-a-judge).

Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Install the requirements:
```
pip install -r requirements.txt
```

## Setting up the RAG database
Export all pages from Confluence. The first time you run this, it may ask for some authentication details. Choose the first option and input the root URL of the RIS Confluence instance: `https://washu.atlassian.net/`. This will take a while to run (~5 minutes) and export all the pages to the `RIS User Documentation` directory:
```
python confluence.py
```

Clean up the markdown files to change the code block syntax from ```java to ```:
```
python fix_markdown.py
```

Compute the embeddings and store them in the RAG database:
```
python manage_rag.py load-docs --dir ./RIS\ User\ Documentation/RIS\ User\ Documentation
```

If you need to clear the database and re-index it:
```
python manage_rag.py clear-collection
```

## Setting up the LLM server
Clone the llama.cpp repository. We use it to run the LLM server that powers the chatbot:
```
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Target a cuda-based build:
```
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
```

Build the server. If you only requested 1 CPU core then it may take a while:
```
cmake --build build -j
```

Download the model. May need to authenticate when running the first time:
```
huggingface-cli download unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF --include "Mistral-Small-3.2-24B-Instruct-2506-UD-Q8_K_XL.gguf" --local-dir models/
```

Run the server. Eventually we may need to explore switching to vllm since it's better optimized for many users:
```
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/ ./build/bin/llama-server --jinja --port 8000 --model models/Mistral-Small-3.2-24B-Instruct-2506-UD-Q8_K_XL.gguf -ngl 99 -fa -c 65536 --mlock --cache-reuse 256 --temp 0.15 --top-k -1 --top-p 1.00 &
```

Return to the main directory:
```
cd ..
```

## Alternative: Run the server with vllm
Install vllm (eventually we can put this in requirements.txt):
```
pip install vllm
```

Serve the same model as above. This one by default is using the unquantized model so it's slower on single user, but throughput should be better for multiple users. HF_HOME is needed to avoid downloading the model to the home directory:
```
HF_HOME=huggingface/ vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice &
```

Note that if we use vllm, then the LLM_MODEL in .env needs to be set to `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, which is the same as the model we downloaded above.

## Run the streamlit app
Start the Streamlit application:
```
STREAMLIT_SERVER_ADDRESS=0.0.0.0 streamlit run streamlit_app_simple.py &
```

## Access the Website
You can access the web UI at `compute1-exec-xxx.ris.wustl.edu:8003`.

# Validation
We build a custom benchmark with a collection of questions generated by LLMs from the documentation to test the knowledge and helpfulness of RIS-bot. The benchmark is judged by an LLM on the closeness of the chatbot's answer to the documentation provided.

## Generate Questions
```
cd validation
mv .env.example .env
# add the following:
# - Gemini API key
# - OpenAI key (if you want to use GPT models, o3 requires identity verification)
# - Path to RIS documentation (a copy is included in the validation folder, dated 07/12/2025)
python generate_questions_gemini.py
```
This will create three questions for each page in RIS Documentation using Gemini 2.5-Pro with the following prompt:
>Create 3 frequently asked questions (FAQs) based on the following document. Write the kinds of questions that users commonly ask after reading this document\
\
Document: {document_name}
\
Content: {document_content}
\
Please provide exactly 3 questions, one per line, without numbering or bullet points:

OpenAI o3 is also available using `generate_questions_o3.py`. Feel free to engineer the prompts to generate different-style questions.

*TO-DO*: Variable number of questions based on document information content

## Benchmark Chatbot
Switch to main directory and run the LLM judging script.
```
cd ..
python validation.py
```

Grab a coffee, take a nap ~ It'll take a while. 💤

*TO-DO*: Append sources referenced by chatbot for comprehensive judging 

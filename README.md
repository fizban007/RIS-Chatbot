## Overview

RIS-Bot is a Retrieval-Augmented-Generation-based (RAG-based) pipeline for assisting researchers with WashU's High Performance Computing platform for research, Research Infrastructure Services (RIS). The pipeline consists of the following components:  
1. **Data Collector:**  
RIS-bot gains its knowledge from [WashU RIS Documentation](https://docs.ris.wustl.edu) hosted on Confluence. `confluence.py` can be used to scrape the entire documentation space at once or perform a partial scrape of documents updated within a given time frame. The time frame can be set within the program. One of the future goals of this project is to remove this parameter and automatically scrape any documents that have a later modification time on Confluence than locally.  
3. **Vector Database:**  
The scraped documentation is embedded by an embedding model of the user's choice (specified in .env) and stored in a vector database for efficient inference-time retrieval. The vector database can be deleted and re-embedded to update its contents. It is a future goal of this project to allow for re-embedding of only files that were recently updated by `confluence.py`. All functionality related to the vector database is run through `manage_rag.py`.  
At inference-time, the query is matched against the vector database using a cosine similarity test to produce the most semantically similar documents. These documents are then provided to the chatbot to provide context for answering the user questions.
4. **LLM Server:**  
The inference LLM is hosted on one of the compute-node / server's ports. The user may select any open-source model through the `.env` file or provide an API key for paywalled models (not extensively tested).  
5. **Web Server:**  
On a separate port, the web server provides a Graphical Web UI that the user can connect to for querying the chatbot. The server instantiates an `EnhancedRAGChatbot()` object which queries the vector database, augments the user's questions with the system prompt and retrieved context, and contacts the LLM Server to generate answers.


## Prerequisites

### System Requirements:

- **CUDA 12.4**

- **Docker image**

If you have CUDA 12.4 installed on your system, you can run RIS-bot in the `fizban007/ris_chatbot` Docker image, or build your own compatible image from `Dockerfile.chatbot`. Otherwise, you will need to download CUDA 12.4 first. CUDA 12.4 is available to members of Digital Transformations Summer Corps at `/storage2/fs1/dt-summer-corp/Active/common/projects/ai-on-washu-infrastructure/chatbot/libs`.

### Program Requirements:

- **LlamaIndex components** (indexing documentation to facilitate RAG)

- **ChromaDB** (vector database for storing vectorized documentation)

- **huggingface hub** (downloading models for embedding and inference)

- **OpenAI and Gemini** (API support for validation Q&A generation and LLM-based judging)

- **Llama.cpp / VLLM** (LLM server)

- **PyTorch** (VLLM dependency)

It is recommended to create a separate virtual environment using tools such as `uv` or `conda`. Follow along for more detailed steps using `venv`.

## Environment Setup (with example for RIS)
1.  Make sure the program has access to CUDA 12.4, your storage location, and your home directory
   
2. By default, the web UI will be hosted on port `8501` `(0.0.0.0:8501)` of the Docker container. You can map this to any available port of your choice. 
   **This is the port users will connect to.**

3. Connect to a compute node with 1 GPU and run your Docker container.

4. Switch to your working directory and clone this repository
   ```
   cd <YOUR_WORKING_DIR>
   git clone https://github.com/Digital-Transformation-Summer-Corps/RIS-Chatbot.git
   cd RIS-Chatbot
   ```

5. Set your own configurations as environment variables. A template is provided as `.env.example`. Note that you would need to add your own Gemini API key for validation (QA generation & LLM-as-a-judge).
   ```
   cp .env.example .env
   ```
   *cp does not cause any changes that Git tracks, so there will be no conflicts*


6. Create and activate a virtual environment in the chatbot directory. Then, install the requirements.
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   *(Example Using RIS)*
   First create a configuration file for your job,
   ```bash
   vim ~/risbot.bsub
   ```
   And paste the following contents:
   ```bash
   #!/bin/bash
   #BSUB -q artsci-interactive
   #BSUB -G compute-artsci
   #BSUB -n 8
   #BSUB -R 'select[port8003=1]'
   #BSUB -R 'gpuhost rusage[mem=120GB]'
   #BSUB -gpu 'num=1'
   #BSUB -a "docker(fizban007/ris_chatbot)"
   
   # Option 1: cloning the repo yourself
   # cd <YOUR_WORKING_DIR>
   # git clone https://github.com/Digital-Transformation-Summer-Corps/RIS-Chatbot.git
   # cd RIS-Chatbot
   # Option 2: for dt-summer-corp and admin users, a working copy exists on storage2
   cd /storage2/fs1/dt-summer-corp/Active/common/projects/ai-on-washu-infrastructure/chatbot/ragbot-dev

   # Create venv environment if it doesn't exist
   if [ ! -d "venv" ]; then
       echo "Creating virtual environment..."
       python -m venv venv
       source venv/bin/activate
       pip install requirements.txt
   # Otherwise, activate existing venv
   else
       echo "Virtual environment already exists"
       source venv/bin/activate
   fi
   ```
   Then, create a bash script for exporting environment variables and submitting the job.
   ```bash
   cd ~
   vim risbot.sh
   ```
   and paste in the following contents.
   ``` bash
   # Replace `<PATH TO CUDA 12.4>` with your CUDA 12.4 folder (e.g., `/storage2/fs1/dt-summer-corp/Active/common/projects/ai-on-washu-infrastructure/chatbot/libs` for admin / dt-summer-corp members)
   # Replace `<YOUR STORAGE LOCATION>` with your storage location (e.g. `/storage2/fs1/dt-summer-corp/Active`).
   export LSF_DOCKER_VOLUMES="<PATH TO CUDA 12.4>:/usr/local/modules <YOUR STORAGE LOCATION>:<YOUR STORAGE LOCATION> $HOME:$HOME"
   # Replace `<PORT OF CHOICE>` with your desired port.
   export LSF_DOCKER_PORTS='<PORT OF CHOICE>:8501'
   # Replace <COMPUTE GROUP> and <INTERACTIVE QUEUE> with your compute group and accessible interactive queue
   bsub -Is < ~/risbot.bsub
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
### vLLM
### llama.cpp
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

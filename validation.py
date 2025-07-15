from google import genai
from rag_chatbot import EnhancedRAGChatbot, RAGConfig
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
import re

load_dotenv()

VALIDATION_QUESTIONS_FILE = os.getenv('VALIDATION_QUESTIONS_FILE')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
LLM_BASE_URL = os.getenv('LLM_BASE_URL')
LLM_MODEL = os.getenv('LLM_MODEL')
EMBED_BASE_URL = os.getenv('EMBED_BASE_URL')
EMBED_MODEL = os.getenv('EMBED_MODEL')
DATA_DIR = os.getenv('DATA_DIR')
VALIDATION_PROMPT_TEMPLATE_FILE = os.getenv('VALIDATION_PROMPT_TEMPLATE_FILE')
if VALIDATION_PROMPT_TEMPLATE_FILE and os.path.exists(VALIDATION_PROMPT_TEMPLATE_FILE):
    with open(VALIDATION_PROMPT_TEMPLATE_FILE, 'r') as f:
        VALIDATION_PROMPT_TEMPLATE = f.read()
else:
    raise ValueError(f"Validation prompt template file not found: {VALIDATION_PROMPT_TEMPLATE_FILE}")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Configure to use llama.cpp server
config = RAGConfig(
    llm_base_url=LLM_BASE_URL if LLM_BASE_URL else "http://localhost:8000/v1",
    llm_model=LLM_MODEL if LLM_MODEL else "mistral-small-3.2-24b",
    embed_base_url=EMBED_BASE_URL if EMBED_BASE_URL else "http://localhost:8001/v1",
    embed_model=EMBED_MODEL if EMBED_MODEL else "nomic-embed-text-v1.5"
)

# Read validation questions
val_questions = pd.read_csv(VALIDATION_QUESTIONS_FILE)

# Initialize chatbot
chatbot = EnhancedRAGChatbot(config)

# Load documents from the data directory
print("Loading documents from directory...")
documents = chatbot.load_documents(directory_path=DATA_DIR)
print(f"Loaded {len(documents)} documents")

# Build the index from documents
print("Building index from documents...")
chatbot.build_index(documents)
print("Index built successfully!")

# Initialize counters and results list
total_score = 0
total_questions = 0
results = []

# Create timestamp for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for root_dir, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.md'):
            print(f"Checking knowledge from {file}...")
            file_path = os.path.join(root_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Get questions for this specific file
            file_questions = val_questions[val_questions['Document Name'] == file]['Question']
            
            for question in file_questions:
                print(f"Question: {question}")
                
                # Query the LLM (if you have documents loaded)
                try:
                    chatbot_response = chatbot.query(question)
                    print(f"ChatbotResponse: {chatbot_response}")
                except Exception as e:
                    # Add result showing the error
                    results.append({
                        'question': question,
                        'chatbot_response': f"ERROR: {str(e)}",
                        'gemini_response': "SKIPPED - LLM unavailable",
                        'score': 0,
                        'file': file,
                        'file_path': file_path
                    })
                    continue
                
                # Check if the response is correct using Gemini
                prompt = VALIDATION_PROMPT_TEMPLATE.format(question=question, response=chatbot_response, context=file_content)
                gemini_response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                
                gemini_text = gemini_response.text
                print(f"GeminiResponse: {gemini_text}")
                
                # Parse the numeric score from Gemini's response
                try:
                    # Extract the score (should be 1-5)
                    score_text = gemini_text.strip()
                    # Try to extract number from the response
                    score_match = re.search(r'\b([1-5])\b', score_text)
                    if score_match:
                        score = int(score_match.group(1))
                    else:
                        print(f"Warning: Could not parse score from '{score_text}', defaulting to 1")
                        score = 1
                except (ValueError, AttributeError):
                    print(f"Warning: Could not parse score from '{gemini_text}', defaulting to 1")
                    score = 1
                
                total_score += score
                total_questions += 1
                
                # Add result to list
                results.append({
                    'question': question,
                    'chatbot_response': chatbot_response,
                    'gemini_response': gemini_text,
                    'score': score,
                    'file': file,
                    'file_path': file_path
                })
                
                print(f"Score: {score}/5 | Running Average: {total_score/total_questions:.2f}")

# Create DataFrame from results
df = pd.DataFrame(results)

# Reset index to start from 1 instead of 0
df.index = range(1, len(df) + 1)

# Save to CSV file with timestamp
output_filename = f"val_{timestamp}.csv"
df.to_csv(output_filename, index=True, index_label='question_number')

print(f"\nResults saved to: {output_filename}")
print(f"Total questions: {total_questions}")
print(f"Total score: {total_score}")
if total_questions > 0:
    average_score = total_score / total_questions
    print(f"Average score: {average_score:.2f}/5.0 ({average_score*20:.1f}%)")
else:
    print("No questions were processed. Please check your DATA_DIR and validation questions file.")
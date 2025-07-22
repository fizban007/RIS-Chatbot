import os
import csv
from openai import OpenAI
from pathlib import Path
import time
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration - these values are loaded from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "RIS_docs")  # Default fallback
OUTPUT_CSV = "document_questions_o3.csv"
SUPPORTED_EXTENSIONS = [ '.md']  # Add more text file extensions as needed

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

def read_text_file(file_path: str) -> str:
    """Read and return the contents of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def generate_questions(document_content: str, document_name: str) -> List[str]:
    """Generate 3 questions about the document using OpenAI API."""
    
    prompt = f"""Given the following text, generate 3 questions a new user might ask while trying to understand or use this system.

Document: {document_name}

Content:
{document_content}

Please provide exactly 3 questions, one per line, without numbering or bullet points:"""

    try:
        response = client.responses.create(
            model="o3",
            input=prompt,
        )
        questions_text = response.output_text.strip()
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        # Ensure we have exactly 3 questions
        if len(questions) < 3:
            questions.extend([f"What are the main topics discussed in {document_name}?"] * (3 - len(questions)))
        elif len(questions) > 3:
            questions = questions[:3]
            
        return questions
        
    except Exception as e:
        print(f"Error generating questions for {document_name}: {e}")
        # Return default questions if API call fails
        return [
            f"What is the main purpose of {document_name}?",
            f"What are the key points covered in {document_name}?",
            f"What information can be learned from {document_name}?"
        ]

def find_text_files(folder_path: str) -> List[Tuple[str, str]]:
    """Find all text files in the folder and its subdirectories."""
    text_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = Path(file).suffix.lower()
            
            if file_extension in SUPPORTED_EXTENSIONS:
                text_files.append((file, file_path))
    
    return text_files

def main():
    """Main function to process documents and generate questions."""
    
    # Check if the path exists
    if not os.path.exists(DOCUMENTS_FOLDER):
        print(f"Error: Path '{DOCUMENTS_FOLDER}' does not exist.")
        print("Please update the DOCUMENTS_FOLDER variable with the correct path.")
        return
    
    # Check if it's a file or folder
    if os.path.isfile(DOCUMENTS_FOLDER):
        # Single file processing
        file_name = os.path.basename(DOCUMENTS_FOLDER)
        file_extension = Path(DOCUMENTS_FOLDER).suffix.lower()
        
        if file_extension not in SUPPORTED_EXTENSIONS:
            print(f"Error: File '{file_name}' has unsupported extension '{file_extension}'")
            print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
            return
        
        print(f"Processing single file: {file_name}")
        text_files = [(file_name, DOCUMENTS_FOLDER)]
        
    else:
        # Folder processing (existing behavior)
        print(f"Scanning for text files in folder: {DOCUMENTS_FOLDER}")
        text_files = find_text_files(DOCUMENTS_FOLDER)
        
        if not text_files:
            print(f"No text files found in {DOCUMENTS_FOLDER}")
            return
        
        print(f"Found {len(text_files)} text files.")
    
    # Prepare CSV data
    csv_data = []
    csv_headers = ['Document Name', 'Document Path', 'Question']
    
    # Process each file
    for i, (file_name, file_path) in enumerate(text_files, 1):
        print(f"Processing {i}/{len(text_files)}: {file_name}")
        
        # Read document content
        content = read_text_file(file_path)
        if not content:
            print(f"Skipping {file_name} - could not read content")
            continue
        
        # Generate questions
        questions = generate_questions(content, file_name)
        
        # Add to CSV data
        for question in questions:
            csv_data.append([file_name, file_path, question])
        
        # Add a small delay to avoid hitting API rate limits
        time.sleep(0.5)
    
    # Save to CSV
    print(f"\nSaving results to {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerows(csv_data)
    
    print(f"Complete! Generated {len(csv_data)} questions for {len(text_files)} documents.")
    print(f"Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 
# Document Question Generator

This script uses the OpenAI API to generate three questions about documents, then saves the results to a CSV file. It can process either:
- A single file (generates questions for just that file)
- A folder and its subdirectories (generates questions for all supported files found)

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get your OpenAI API key:**
   - Go to [OpenAI's website](https://platform.openai.com/api-keys)
   - Create an account or log in
   - Generate an API key

3. **Configure environment variables:**
   - Copy `config.example` to `.env`:
     ```bash
     cp config.example .env
     ```
   - Edit the `.env` file and replace:
     - `your-openai-api-key-here` with your actual OpenAI API key
     - `path/to/your/documents` with either:
       - The path to a single document file, or
       - The path to a folder containing documents

## Usage

Run the script:
```bash
python generate_questions.py
```

The script will:
1. Check if the specified path is a file or folder
   - **If it's a file**: Process just that single file
   - **If it's a folder**: Scan the folder and all subdirectories for text files
2. Read each document's content
3. Generate 3 questions per document using OpenAI's API
4. Save all results to `document_questions.csv`

## Supported File Types

By default, the script processes these file types:
- `.txt`
- `.md`
- `.rst`
- `.log`

You can modify the `SUPPORTED_EXTENSIONS` list in the script to include additional file types.

## Output

The script generates a CSV file with the following columns:
- **Document Name**: The filename of the document
- **Document Path**: The full path to the document
- **Question**: One of the generated questions (3 rows per document)

## Configuration Options

Environment variables (set in `.env` file):
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `DOCUMENTS_FOLDER`: Path to either a single document file or a folder containing documents (required)

Script variables (modify in the script if needed):
- `OUTPUT_CSV`: Name of the output CSV file (default: "document_questions.csv")
- `SUPPORTED_EXTENSIONS`: List of file extensions to process
- `max_content_length`: Maximum characters to send to OpenAI (default: 8000)

## Notes

- The script includes a small delay between API calls to avoid rate limits
- Large documents are truncated to avoid token limits
- If the API call fails, default questions are generated
- The script handles different text encodings (UTF-8 and Latin-1)
- The `.env` file is automatically ignored by git (listed in `.gitignore`) to keep your API key secure 
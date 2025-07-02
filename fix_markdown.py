import os
import re

# Directory containing the markdown files
markdown_dir = "RIS User Documentation"

# Function to process a single markdown file
def process_markdown_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace ```java with ```
        updated_content = re.sub(r'```java\b', '```', content)
        
        # Only write back if changes were made
        if updated_content != content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(updated_content)
            print(f"Updated: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Walk through all files in the directory recursively
for root, dirs, files in os.walk(markdown_dir):
    for file in files:
        if file.endswith('.md'):
            file_path = os.path.join(root, file)
            process_markdown_file(file_path)

print("Finished processing all markdown files.")
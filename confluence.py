from atlassian import Confluence
import subprocess
import json
import os
import re

confluence = Confluence(
  url='https://washu.atlassian.net/',
  username='cyuran@wustl.edu',
  # I believe the password is not required since our confluence page is public.
)

space_key = 'RUD' # Replace with your space key
markdown_path_url_dict = {}

def get_page_hierarchy(space_key):
    # Get all pages in the space
    pages = confluence.get_all_pages_from_space(space_key, expand='ancestors')
    
    # Build hierarchy dictionary
    hierarchy = {}
    root_pages = []
    
    for page in pages:
        page_id = page['id']
        title = page['title']
        ancestors = page.get('ancestors', [])
        
        if not ancestors:
            # Root page (no parent)
            root_pages.append({
                'id': page_id,
                'title': title,
                'children': []
            })
            hierarchy[page_id] = root_pages[-1]
        else:
            # Child page
            parent_id = ancestors[-1]['id']  # Direct parent is last ancestor
            page_data = {
                'id': page_id,
                'title': title,
                'children': []
            }
            hierarchy[page_id] = page_data
    
    # Build parent-child relationships
    for page in pages:
        page_id = page['id']
        ancestors = page.get('ancestors', [])
        
        if ancestors:
            parent_id = ancestors[-1]['id']
            if parent_id in hierarchy:
                hierarchy[parent_id]['children'].append(hierarchy[page_id])
    
    return root_pages

hierarchy = get_page_hierarchy(space_key)
print(hierarchy)

# Function to sanitize filename for safe file operations
def sanitize_filename(filename):
    """Remove or replace characters that are not safe for filenames"""
    # Replace spaces and special characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized

def export_page_with_metadata(page, path_prefix=""):
    """Export a single page and create its metadata JSON file"""
    page_id = page['id']
    title = page['title']
    title = sanitize_filename(title)
    
    print(f"Processing: {title} (ID: {page_id})")

    # Create the webpage URL
    webpage_url = f"https://washu.atlassian.net/wiki/spaces/RUD/pages/{page_id}/{title.replace(' ', '+')}"

    # Check if markdown file already exists
    markdown_path = path_prefix + title + '.md'

    markdown_path_url_dict[title + '.md'] = webpage_url
    
    if os.path.exists(markdown_path):
        print(f"Markdown file already exists: {markdown_path}, skipping export...")
        return
    # print(f"path is {path_prefix + title}")
    
    try:
        # Export the page using cf-export
        subprocess.run(['cf-export', 'page', page_id], check=True)
        
        # Create metadata dictionary
        metadata = {
            'webpage_url': webpage_url,
            'title': title,
            'page_id': page_id,
            'space_key': space_key
        }
        
        # Save metadata to JSON file
        with open(path_prefix + title + '.json', 'w', encoding='utf-8') as json_file:
            json.dump(metadata, json_file, indent=2, ensure_ascii=False)
        print(f"Saved metadata to: {path_prefix + title + '.json'}")

            
    except subprocess.CalledProcessError as e:
        print(f"Error exporting page {title} (ID: {page_id}): {e}")
        return
            

def walk_and_export_hierarchy(pages, path_prefix=""):
    """Recursively walk through page hierarchy and export each page"""
    for page in pages:
        # Add indentation to show hierarchy level
        # indent = "  " * level
        # print(f"{indent}Exporting: {page['title']}")
        
        if should_export:
            # Export this page
            markdown_path = export_page_with_metadata(page, path_prefix)
            print(f"Markdown path: {markdown_path}")
            if markdown_path:
                # Ensure the directory exists before writing
                updated_pages_file = os.path.join("RIS User Documentation", "updated_pages.txt")
                os.makedirs(os.path.dirname(updated_pages_file), exist_ok=True)
                print(f"Writing to: '{updated_pages_file}'!")
                with open(updated_pages_file, 'a') as f:
                    f.write(f"{markdown_path}\n")
        else:
            # The markdown file and metadata should already exist
            markdown_path = path_prefix + page['title'] + '.md'
            # Read the JSON metadata file to get the webpage URL
            json_path = path_prefix + page['title'] + '.json'
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as json_file:
                        metadata = json.load(json_file)
                        webpage_url = metadata.get('webpage_url', '')
                        markdown_path_url_dict[page['title'] + '.md'] = webpage_url
                        print(f"Read webpage URL from {json_path}: {webpage_url}")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading JSON metadata from {json_path}: {e}")
                    webpage_url = ''
            else:
                print(f"JSON metadata file not found: {json_path}")
                webpage_url = ''

        # Recursively export children
        if page['children']:
            print(f"Processing children of: {page['title']}")
            walk_and_export_hierarchy(page['children'], path_prefix + page['title'] + "/")

# Walk through the hierarchy and export all pages
print("Starting hierarchical export...")
walk_and_export_hierarchy(hierarchy, "RIS User Documentation/")
os.system("mv RIS\\ User\\ Documentation/RIS\\ User\\ Documentation.json RIS\\ User\\ Documentation/RIS\\ User\\ Documentation/RIS\\ User\\ Documentation.json")
print("\nHierarchical export completed! All pages and metadata have been exported.")

# Update markdown files to replace internal links with webpage URLs
print("\nUpdating internal links in markdown files...")

def update_markdown_links(file_path, url_mapping):
    """Update markdown file to replace internal links with webpage URLs"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        original_content = content
        
        # Pattern to match markdown links: [text](path.md)
        import re
        
        # Find all markdown links
        link_pattern = r'\[([^\]]*)\]\((.*?\.md.*?)\)'
        
        def replace_link(match):
            link_text = match.group(1)
            link_path = match.group(2)
            
            # Clean up the path - remove any anchors or fragments
            clean_path = link_path.split('#')[0]
            # Replace %20 with spaces in the path
            clean_path = clean_path.replace('%20', ' ')
            # Try to find the corresponding URL
            for md_file, url in url_mapping.items():
                if clean_path.endswith(md_file) or md_file.endswith(clean_path):
                    # If there was an anchor in the original link, preserve it
                    if '#' in link_path:
                        anchor = '#' + link_path.split('#')[1]
                        return f'[{link_text}]({url}{anchor})'
                    else:
                        return f'[{link_text}]({url})'
            
            # If no match found, keep the original link
            return match.group(0)
        
        # Replace all matching links
        content = re.sub(link_pattern, replace_link, content)
        
        # Only write back if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"Updated links in: {file_path}")
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

# Walk through all markdown files and update links
import os

def update_all_markdown_files(directory, url_mapping):
    """Recursively update all markdown files in directory"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                update_markdown_links(file_path, url_mapping)

# Update all markdown files with the URL mapping
update_all_markdown_files("RIS User Documentation/", markdown_path_url_dict)

print("Finished updating internal links to webpage URLs!")


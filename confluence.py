from atlassian import Confluence
import subprocess
import json
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests, json
from requests.auth import HTTPBasicAuth

host = "https://washu.atlassian.net/"
username = 'c.daedalus@wustl.edu'
space_key = 'RUD' # Replace with your space key
space_id = 1621884932  # We got this from the API call
time_window = 14 # The number of days you want to look back for updated pages

confluence = Confluence(
  url=host,
  username=username,
  # I believe the password is not required since our confluence page is public.
)

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
    
    print("\n"+"="*100)
    print(f"Processing: {title} (ID: {page_id})")

    # Create the webpage URL
    webpage_url = f"https://washu.atlassian.net/wiki/spaces/RUD/pages/{page_id}/{title.replace(' ', '+')}"

    # Check if markdown file already exists
    markdown_path = path_prefix + title + '.md'

    markdown_path_url_dict[title + '.md'] = webpage_url
    
    if os.path.exists(markdown_path):
        print(f"Markdown file already exists: {markdown_path}, deleting...")
        os.remove(markdown_path)
    # print(f"path is {path_prefix + title}")
    
    try:
        # Export the page using cf-export
        results = subprocess.run(['cf-export', 'page', page_id], check=True)
        
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
        return markdown_path
            
    except subprocess.CalledProcessError as e:
        print(f"Error exporting page {title} (ID: {page_id}): {e}")
        return None
            

def walk_and_export_hierarchy(pages, page_ids, path_prefix=""):
    """Recursively walk through page hierarchy and export each page"""
    for page in pages:
        # If we have specific page IDs to filter, only export if this page is in the list
        should_export = not page_ids or page['id'] in page_ids
        
        if should_export:
            # Export this page
            markdown_path = export_page_with_metadata(page, path_prefix)
            print(f"Markdown path: {markdown_path}")
            if markdown_path:
                # Ensure the directory exists before writing
                os.makedirs(os.path.dirname(path_prefix + 'updated_pages.txt'), exist_ok=True)
                print(f"Writing to: {path_prefix + 'updated_pages.txt'!}")
                with open(path_prefix + 'updated_pages.txt', 'a') as f:
                    f.write(f"{markdown_path}\n")
        
        # Recursively export children
        if page['children']:
            print(f"Processing children of: {page['title']}")
            # Assumes parent directory exists, either from export or from previous run
            child_path_prefix = path_prefix + sanitize_filename(page['title']) + "/"
            walk_and_export_hierarchy(page['children'], page_ids, child_path_prefix)

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
    print("Finished updating internal links to webpage URLs!")

def get_conf_update():
    # base URL for the Confluence API
    base_url = f"{host}/wiki/rest/api/content"   

    # Get the date one week ago
    week_ago = datetime.now(ZoneInfo("UTC")) - timedelta(days=time_window)
    print(f"Looking for pages modified after: {week_ago.strftime('%Y-%m-%d')}")
    
    # Make the request to get all pages in the space
    response = requests.request("GET",
                                base_url,
                                headers = {
                                    "Accept": "application/json"
                                },
                                params = {
                                    "spaceKey": space_key,
                                    "limit": 100,  # Increased limit to get more pages
                                    "expand": "version,history",  # Changed expand to include history
                                })
    
    print(f"Status Code: {response.status_code}")
    
    # Check if the response is successful
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return [], []

    try:
        # Parse the response as JSON
        data = response.json()
        
        # Filter results by last modified date
        results = data['results']
        filtered_results = []
        
        print(f"Total pages found: {len(results)}")
        
        for item in results:
            if 'version' in item and 'when' in item['version']:
                last_modified = item['version']['when']
                
                # Parse the date and check if it's within the last week
                try:
                    modified_date = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    
                    if modified_date >= week_ago:
                        filtered_results.append(item)
                        
                except Exception as e:
                    print(f"Error parsing date {last_modified}: {e}")

        # Get the page IDs and titles from filtered results
        page_ids = [item['id'] for item in filtered_results]
        page_titles = [item['title'] for item in filtered_results]

        print(f"Found {len(page_ids)} pages modified in the last {time_window} days")
        
        return page_ids, page_titles
        
    except requests.exceptions.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return [], []
    
if __name__ == '__main__':
    # Ensure the output directory exists
    os.makedirs("RIS User Documentation", exist_ok=True)
    
    # If not running for the first time (i.e. updated_pages.txt exists), get updated pages from the last week
    updated_pages_ids = []
    if os.path.exists("RIS User Documentation/RIS User Documentation/updated_pages.txt"):
        updated_pages_ids, updated_pages_titles = get_conf_update()
        if updated_pages_ids:
            print(f"Updated pages in the last week: ")
            for title in updated_pages_titles:
                print(f"- {title}")
        else:
            print("No pages updated in the last week")
    else:
        print("First run detected - will export all pages")
    
    # Clear the updated_pages.txt file either way
    with open("RIS User Documentation/RIS User Documentation/updated_pages.txt", 'w') as f:
        f.write("")
    
    # Get page hierarchy
    hierarchy = get_page_hierarchy(space_key)

    # Walk through the hierarchy and export all or updated pages
    print("Starting hierarchical export...")
    walk_and_export_hierarchy(hierarchy, updated_pages_ids, "RIS User Documentation/")
    if os.path.exists("RIS User Documentation/RIS User Documentation.json"):
        os.system("mv RIS\\ User\\ Documentation/RIS\\ User\\ Documentation.json RIS\\ User\\ Documentation/RIS\\ User\\ Documentation/RIS\\ User\\ Documentation.json")
    print("\nHierarchical export completed! All pages and metadata have been exported.")
    
    # Update markdown files to replace internal links with webpage URLs
    print("\nUpdating internal links in markdown files...")
    update_all_markdown_files("RIS User Documentation/", markdown_path_url_dict)
    

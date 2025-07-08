import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

host = "https://washu.atlassian.net"
space_key = "RUD"
space_id = 1621884932  # We got this from the API call
time_window = 14

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

if __name__ == "__main__":
    _, page_titles = get_conf_update()
    if page_titles:
        print(f"\nPages modified in the last {time_window} days:")
        for title in page_titles:
            print(f"  - {title}")
    else:
        print(f"\nNo pages modified in the last {time_window} days.")
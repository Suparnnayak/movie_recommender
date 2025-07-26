import requests

API_KEY = "your_omdb_api_key"  # Replace with your OMDB API key

def fetch_movie_data(title):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={title}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("Response") == "True":
            return {
                "title": data.get("Title", "N/A"),
                "poster": data.get("Poster", ""),
                "description": data.get("Plot", "No description available")
            }
    return None

from flask import Flask, render_template, request
import pickle
import requests

app = Flask(__name__)

# Load TF-IDF and metadata
with open('artifacts/tfidf.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('artifacts/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Compute similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(tfidf_matrix)

# Movie title lookup
movie_titles = metadata['title'].tolist()
movie_indices = {title.lower(): idx for idx, title in enumerate(movie_titles)}

# OMDB helper
def get_movie_data(title):
    api_key = "7a86ac11"  # Replace with your OMDB API key
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'poster': data.get('Poster', ''),
            'description': data.get('Plot', 'No description available.')
        }
    return {'poster': '', 'description': 'No description available.'}

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error = None

    if request.method == 'POST':
        movie = request.form['movie'].strip().lower()

        if movie in movie_indices:
            idx = movie_indices[movie]
            sim_scores = list(enumerate(similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            for i, _ in sim_scores:
                title = movie_titles[i]
                data = get_movie_data(title)
                recommendations.append({
                    'title': title,
                    'poster': data['poster'],
                    'description': data['description']
                })
        else:
            error = f"No match found for '{movie}'. Please try a different movie."

    return render_template('index.html', recommendations=recommendations, error=error)

if __name__ == '__main__':
    app.run(debug=True)

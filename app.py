from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load TF-IDF and metadata
with open('artifacts/tfidf.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('artifacts/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Compute similarity matrix (optional if already saved, but recomputing here)
similarity = cosine_similarity(tfidf_matrix)

# Extract titles list
movie_titles = metadata['title'].tolist()
movie_indices = {title.lower(): idx for idx, title in enumerate(movie_titles)}

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
            recommendations = [movie_titles[i] for i, _ in sim_scores]
        else:
            error = f"No match found for '{movie}'. Please try a different movie."

    return render_template('index.html', recommendations=recommendations, error=error)

if __name__ == '__main__':
    app.run(debug=True)

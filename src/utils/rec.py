import pickle
from fuzzywuzzy import process
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.tfidf_matrix = data['tfidf_matrix']
        self.metadata = data['metadata']

    def recommend(self, movie_title, top_n=5):
        titles = self.metadata['title'].values

        # Use fuzzywuzzy to find the best match
        best_match = process.extractOne(movie_title, titles)[0]

        # Find index of the best matched title
        idx = self.metadata[self.metadata['title'] == best_match].index[0]

        # Compute cosine similarity dynamically
        similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Get similarity scores for that movie
        sim_scores = list(enumerate(similarity[idx]))

        # Sort based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top_n similar movie indices (excluding the movie itself)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]

        # Return the top recommended titles
        return self.metadata.iloc[movie_indices][['title']]

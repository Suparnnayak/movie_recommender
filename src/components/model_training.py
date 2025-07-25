import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

class ModelTraining:
    def __init__(self, tfidf_matrix, metadata, model_path):
        self.tfidf_matrix = tfidf_matrix 
        self.metadata = metadata
        self.model_path = model_path
       
    def save(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'tfidf_matrix': self.tfidf_matrix,
                'metadata': self.metadata
            }, f)

    def train_model(self):
        with open(self.tfidf_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)

        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        similarity = cosine_similarity(tfidf_matrix)

        with open(self.model_path, 'wb') as f:
            pickle.dump((similarity, metadata), f)

        return similarity

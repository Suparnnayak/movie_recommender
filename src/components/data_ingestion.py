import os
import pandas as pd

class DataIngestion:
    def __init__(self, movies_path: str, credits_path: str):
        self.movies_path = movies_path
        self.credits_path = credits_path

    def load_and_merge(self):
        # Load the CSV files
        movies = pd.read_csv(self.movies_path)
        credits = pd.read_csv(self.credits_path)

        # Merge on title
        df = movies.merge(credits, on='title')

        # Columns we need
        expected_columns = ['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
        for col in expected_columns:
            if col not in df.columns:
                raise Exception(f"Missing column: {col}")

        return df

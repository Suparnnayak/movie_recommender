import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class DataTransformation:
    def __init__(self, tfidf_path, metadata_path):
        self.tfidf_path = tfidf_path
        self.metadata_path = metadata_path

    def _convert(self, obj):
        try:
            L = []
            for i in ast.literal_eval(obj):
                L.append(i['name'])
            return L
        except:
            return []

    def _fetch_director(self, obj):
        try:
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    return [i['name']]
            return []
        except:
            return []

    def _preprocess(self, df):
        df.dropna(inplace=True)

        df['genres'] = df['genres'].apply(self._convert)
        df['keywords'] = df['keywords'].apply(self._convert)
        df['cast'] = df['cast'].apply(lambda x: self._convert(x)[:3])
        df['crew'] = df['crew'].apply(self._fetch_director)

        df['overview'] = df['overview'].apply(lambda x: x.split())

        df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
        df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())
        return df[['id', 'title', 'tags']]

    def vectorize(self, df):
        df = self._preprocess(df)
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['tags'])

        with open(self.tfidf_path, 'wb') as f:
            pickle.dump(tfidf_matrix, f)

        with open(self.metadata_path, 'wb') as f:
            pickle.dump(df[['id', 'title']], f)

        return tfidf_matrix, df

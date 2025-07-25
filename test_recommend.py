from src.utils.rec import Recommender

recommender = Recommender(model_path="artifacts/model.pkl")

try:
    movie_title = "Avatar"
    recommendations = recommender.recommend(movie_title)
    print(f"\nTop recommendations for '{movie_title}':\n")
    print(recommendations)
except ValueError as e:
    print(e)

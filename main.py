from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

def run_pipeline():
    # 🔧 Update paths
    movies_path = 'notebook/data/tmdb_5000_movies.csv'
    credits_path = 'notebook/data/tmdb_5000_credits.csv'
    tfidf_path = 'artifacts/tfidf.pkl'
    metadata_path = 'artifacts/metadata.pkl'
    model_path = 'artifacts/model.pkl'  

    # Step 1: Load and merge
    ingestor = DataIngestion(movies_path, credits_path)
    df = ingestor.load_and_merge()

    # Step 2: Vectorize
    transformer = DataTransformation(tfidf_path, metadata_path)
    tfidf_matrix, metadata = transformer.vectorize(df)

    # Step 3: Train and save
    trainer = ModelTraining(tfidf_matrix, metadata, model_path)
    trainer.save()


if __name__ == "__main__":
    run_pipeline()

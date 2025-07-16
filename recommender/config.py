"""
Configuration file for the Hybrid Collaborative Filtering Recommendation System
Contains all hyperparameters, file paths, and model configurations
"""

import os

# Data Configuration
DATA_PATH = "data/sample_data.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FULL_PATH = os.path.join(BASE_DIR, DATA_PATH)

# Weights for Hybrid Combination
WEIGHTS = {
    "user_based": 0.3,
    "item_based": 0.3,
    "content_based": 0.4
}

# TF-IDF Configuration
TFIDF_PARAMS = {
    "max_features": 5000,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.8
}

# Collaborative Filtering Parameters
COLLAB_PARAMS = {
    "min_ratings_per_user": 5,
    "min_ratings_per_item": 5,
    "similarity_threshold": 0.1
}

# Evaluation Configuration
EVAL_CONFIG = {
    "top_k": 10,
    "test_size": 0.2,
    "random_state": 42
}

# UI Configuration
UI_CONFIG = {
    "default_recommendations": 5,
    "max_recommendations": 20,
    "page_title": "Hybrid Recommendation System",
    "page_icon": "🎯"
}

# Model Configuration
MODEL_CONFIG = {
    "user_similarity_metric": "cosine",
    "item_similarity_metric": "cosine",
    "content_similarity_metric": "cosine",
    "rating_scale": (1, 5),
    "implicit_feedback_threshold": 4.0
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "logs/recommender.log"
}

# BERT Configuration (for advanced content-based filtering)
BERT_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "max_length": 512,
    "batch_size": 32
}

# MLflow Configuration
MLFLOW_CONFIG = {
    "experiment_name": "hybrid_recommender",
    "tracking_uri": "sqlite:///mlflow.db"
}

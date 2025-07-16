"""
Item-Based Collaborative Filtering Module
Implements item-item similarity computation and rating prediction
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path to import config and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, COLLAB_PARAMS
from utils.logger import get_logger, log_performance

logger = get_logger(__name__)

class ItemBasedCF:
    """
    Item-Based Collaborative Filtering implementation
    """
    
    def __init__(self):
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.users = None
        self.items = None
        self.item_means = None
        
    @log_performance
    def fit(self, user_item_matrix: pd.DataFrame) -> 'ItemBasedCF':
        """
        Fit the item-based collaborative filtering model
        
        Args:
            user_item_matrix (pd.DataFrame): User-item rating matrix
            
        Returns:
            ItemBasedCF: Fitted model instance
        """
        try:
            logger.info("Fitting Item-Based Collaborative Filtering model...")
            
            self.user_item_matrix = user_item_matrix.copy()
            self.item_user_matrix = user_item_matrix.T  # Transpose for item-user matrix
            self.users = user_item_matrix.index.tolist()
            self.items = user_item_matrix.columns.tolist()
            
            # Calculate item means for mean-centered ratings
            self.item_means = self.item_user_matrix.mean(axis=1)
            
            # Compute item similarity matrix
            self.item_similarity_matrix = self._compute_item_similarity()
            
            logger.info(f"Item-Based CF model fitted successfully. Items: {len(self.items)}")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting Item-Based CF model: {str(e)}")
            raise
    
    def _compute_item_similarity(self) -> pd.DataFrame:
        """
        Compute cosine similarity between items
        
        Returns:
            pd.DataFrame: Item similarity matrix
        """
        logger.info("Computing item similarity matrix...")
        
        # Convert to numpy array for similarity computation
        item_matrix = self.item_user_matrix.values
        
        # Handle case where all ratings are zero for an item
        # Replace zero vectors with small random values to avoid division by zero
        zero_rows = np.all(item_matrix == 0, axis=1)
        if np.any(zero_rows):
            logger.warning(f"Found {np.sum(zero_rows)} items with no ratings")
            item_matrix[zero_rows] = np.random.normal(0, 0.01, (np.sum(zero_rows), item_matrix.shape[1]))
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(item_matrix)
        
        # Convert to DataFrame for easier indexing
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.items,
            columns=self.items
        )
        
        # Set diagonal to 0 (item shouldn't be similar to itself for recommendation purposes)
        np.fill_diagonal(similarity_df.values, 0)
        
        # Apply similarity threshold
        threshold = COLLAB_PARAMS.get('similarity_threshold', 0.1)
        similarity_df[similarity_df < threshold] = 0
        
        logger.info(f"Item similarity matrix computed. Average similarity: {similarity_df.values.mean():.4f}")
        
        return similarity_df
    
    @log_performance
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a specific user-item pair
        
        Args:
            user_id (int): User ID
            item_id (int): Item ID
            
        Returns:
            float: Predicted rating
        """
        try:
            if self.item_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            if user_id not in self.users:
                logger.warning(f"User {user_id} not found. Returning global average.")
                return self.user_item_matrix.values.mean()
            
            if item_id not in self.items:
                logger.warning(f"Item {item_id} not found. Returning global average.")
                return self.user_item_matrix.values.mean()
            
            # Get similar items that this user has rated
            item_similarities = self.item_similarity_matrix.loc[item_id]
            user_ratings = self.user_item_matrix.loc[user_id]
            
            # Find items that user has rated and are similar to target item
            rated_items = user_ratings[user_ratings > 0].index
            similar_items = item_similarities[rated_items]
            similar_items = similar_items[similar_items > 0]
            
            if len(similar_items) == 0:
                # No similar items found, return item's average rating
                return self.item_means[item_id]
            
            # Calculate weighted average using mean-centered ratings
            numerator = 0
            denominator = 0
            
            for similar_item in similar_items.index:
                similarity = similar_items[similar_item]
                rating = user_ratings[similar_item]
                item_mean = self.item_means[similar_item]
                
                numerator += similarity * (rating - item_mean)
                denominator += abs(similarity)
            
            if denominator == 0:
                return self.item_means[item_id]
            
            # Predicted rating = item_mean + weighted_average_of_deviations
            predicted_rating = self.item_means[item_id] + (numerator / denominator)
            
            # Ensure rating is within valid range
            min_rating, max_rating = MODEL_CONFIG['rating_scale']
            predicted_rating = np.clip(predicted_rating, min_rating, max_rating)
            
            return predicted_rating
            
        except Exception as e:
            logger.error(f"Error predicting rating for user {user_id}, item {item_id}: {str(e)}")
            return self.item_means.get(item_id, self.user_item_matrix.values.mean())
    
    @log_performance
    def predict_ratings_for_user(self, user_id: int, items: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Predict ratings for all items (or specified items) for a user
        
        Args:
            user_id (int): User ID
            items (List[int], optional): List of item IDs. If None, predict for all items.
            
        Returns:
            Dict[int, float]: Dictionary of item_id -> predicted_rating
        """
        try:
            if self.item_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            if items is None:
                items = self.items
            
            predictions = {}
            
            for item_id in items:
                predictions[item_id] = self.predict_rating(user_id, item_id)
            
            logger.info(f"Generated {len(predictions)} predictions for user {user_id}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting ratings for user {user_id}: {str(e)}")
            return {}
    
    @log_performance
    def recommend_items(self, user_id: int, n_recommendations: int = 10, 
                       exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend top-N items for a user
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            exclude_rated (bool): Whether to exclude already rated items
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, predicted_rating) tuples
        """
        try:
            if self.item_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            # Get all item predictions
            predictions = self.predict_ratings_for_user(user_id)
            
            if exclude_rated and user_id in self.users:
                # Exclude items already rated by the user
                rated_items = self.user_item_matrix.loc[user_id]
                rated_items = rated_items[rated_items > 0].index.tolist()
                predictions = {item_id: rating for item_id, rating in predictions.items() 
                             if item_id not in rated_items}
            
            # Sort by predicted rating and return top-N
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            recommendations = sorted_predictions[:n_recommendations]
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
            return []
    
    def get_similar_items(self, item_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """
        Get most similar items to a given item
        
        Args:
            item_id (int): Item ID
            n_items (int): Number of similar items to return
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, similarity_score) tuples
        """
        try:
            if self.item_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            if item_id not in self.items:
                logger.warning(f"Item {item_id} not found")
                return []
            
            similarities = self.item_similarity_matrix.loc[item_id]
            similarities = similarities[similarities > 0]  # Only positive similarities
            
            # Sort by similarity and return top-N
            sorted_similarities = similarities.sort_values(ascending=False)
            similar_items = [(int(iid), float(sim)) for iid, sim in sorted_similarities.head(n_items).items()]
            
            return similar_items
            
        except Exception as e:
            logger.error(f"Error finding similar items for item {item_id}: {str(e)}")
            return []
    
    def get_item_recommendations_based_on_history(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations based on items similar to what user has already rated highly
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, similarity_score) tuples
        """
        try:
            if self.item_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            if user_id not in self.users:
                logger.warning(f"User {user_id} not found")
                return []
            
            # Get user's highly rated items (above threshold)
            user_ratings = self.user_item_matrix.loc[user_id]
            threshold = MODEL_CONFIG.get('implicit_feedback_threshold', 4.0)
            highly_rated_items = user_ratings[user_ratings >= threshold].index.tolist()
            
            if not highly_rated_items:
                logger.info(f"No highly rated items found for user {user_id}")
                return []
            
            # Find items similar to highly rated items
            item_scores = {}
            
            for rated_item in highly_rated_items:
                similar_items = self.get_similar_items(rated_item, n_items=20)
                user_rating = user_ratings[rated_item]
                
                for item_id, similarity in similar_items:
                    if item_id not in user_ratings.index or user_ratings[item_id] == 0:
                        # Weight similarity by user's rating for the source item
                        score = similarity * user_rating
                        if item_id in item_scores:
                            item_scores[item_id] += score
                        else:
                            item_scores[item_id] = score
            
            # Sort by score and return top-N
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = sorted_items[:n_recommendations]
            
            logger.info(f"Generated {len(recommendations)} history-based recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating history-based recommendations for user {user_id}: {str(e)}")
            return []
    
    def get_model_statistics(self) -> Dict:
        """
        Get model statistics
        
        Returns:
            Dict: Model statistics
        """
        if self.item_similarity_matrix is None:
            return {"error": "Model not fitted"}
        
        similarity_values = self.item_similarity_matrix.values
        similarity_values = similarity_values[similarity_values > 0]  # Exclude zeros and diagonal
        
        stats = {
            "n_users": len(self.users),
            "n_items": len(self.items),
            "avg_similarity": float(similarity_values.mean()) if len(similarity_values) > 0 else 0,
            "max_similarity": float(similarity_values.max()) if len(similarity_values) > 0 else 0,
            "min_similarity": float(similarity_values.min()) if len(similarity_values) > 0 else 0,
            "similarity_sparsity": float(np.sum(self.item_similarity_matrix.values == 0) / self.item_similarity_matrix.size),
            "avg_item_rating": float(self.item_means.mean())
        }
        
        return stats

# Convenience function for easy usage
def create_item_based_model(user_item_matrix: pd.DataFrame) -> ItemBasedCF:
    """
    Create and fit an Item-Based Collaborative Filtering model
    
    Args:
        user_item_matrix (pd.DataFrame): User-item rating matrix
        
    Returns:
        ItemBasedCF: Fitted model
    """
    model = ItemBasedCF()
    model.fit(user_item_matrix)
    return model

if __name__ == "__main__":
    # Test the item-based CF model
    try:
        # Create sample data for testing
        np.random.seed(42)
        n_users, n_items = 10, 20
        
        # Create sparse rating matrix
        user_ids = list(range(1, n_users + 1))
        item_ids = list(range(101, 101 + n_items))
        
        # Generate random ratings (with some sparsity)
        data = []
        for user in user_ids:
            n_ratings = np.random.randint(5, 15)  # Each user rates 5-15 items
            rated_items = np.random.choice(item_ids, n_ratings, replace=False)
            for item in rated_items:
                rating = np.random.uniform(1, 5)
                data.append([user, item, rating])
        
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
        user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
        
        # Test the model
        model = create_item_based_model(user_item_matrix)
        
        # Test predictions
        test_user = 1
        recommendations = model.recommend_items(test_user, n_recommendations=5)
        similar_items = model.get_similar_items(101, n_items=3)
        history_recs = model.get_item_recommendations_based_on_history(test_user, n_recommendations=3)
        
        print("Item-Based CF test successful!")
        print(f"Recommendations for user {test_user}: {recommendations}")
        print(f"Similar items to item 101: {similar_items}")
        print(f"History-based recommendations for user {test_user}: {history_recs}")
        print(f"Model statistics: {model.get_model_statistics()}")
        
    except Exception as e:
        print(f"Error testing Item-Based CF: {e}")

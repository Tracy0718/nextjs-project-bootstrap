"""
User-Based Collaborative Filtering Module
Implements user-user similarity computation and rating prediction
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

class UserBasedCF:
    """
    User-Based Collaborative Filtering implementation
    """
    
    def __init__(self):
        self.user_similarity_matrix = None
        self.user_item_matrix = None
        self.users = None
        self.items = None
        self.user_means = None
        
    @log_performance
    def fit(self, user_item_matrix: pd.DataFrame) -> 'UserBasedCF':
        """
        Fit the user-based collaborative filtering model
        
        Args:
            user_item_matrix (pd.DataFrame): User-item rating matrix
            
        Returns:
            UserBasedCF: Fitted model instance
        """
        try:
            logger.info("Fitting User-Based Collaborative Filtering model...")
            
            self.user_item_matrix = user_item_matrix.copy()
            self.users = user_item_matrix.index.tolist()
            self.items = user_item_matrix.columns.tolist()
            
            # Calculate user means for mean-centered ratings
            self.user_means = user_item_matrix.mean(axis=1)
            
            # Compute user similarity matrix
            self.user_similarity_matrix = self._compute_user_similarity()
            
            logger.info(f"User-Based CF model fitted successfully. Users: {len(self.users)}")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting User-Based CF model: {str(e)}")
            raise
    
    def _compute_user_similarity(self) -> pd.DataFrame:
        """
        Compute cosine similarity between users
        
        Returns:
            pd.DataFrame: User similarity matrix
        """
        logger.info("Computing user similarity matrix...")
        
        # Convert to numpy array for similarity computation
        user_matrix = self.user_item_matrix.values
        
        # Handle case where all ratings are zero for a user
        # Replace zero vectors with small random values to avoid division by zero
        zero_rows = np.all(user_matrix == 0, axis=1)
        if np.any(zero_rows):
            logger.warning(f"Found {np.sum(zero_rows)} users with no ratings")
            user_matrix[zero_rows] = np.random.normal(0, 0.01, (np.sum(zero_rows), user_matrix.shape[1]))
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(user_matrix)
        
        # Convert to DataFrame for easier indexing
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.users,
            columns=self.users
        )
        
        # Set diagonal to 0 (user shouldn't be similar to themselves for recommendation purposes)
        np.fill_diagonal(similarity_df.values, 0)
        
        # Apply similarity threshold
        threshold = COLLAB_PARAMS.get('similarity_threshold', 0.1)
        similarity_df[similarity_df < threshold] = 0
        
        logger.info(f"User similarity matrix computed. Average similarity: {similarity_df.values.mean():.4f}")
        
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
            if self.user_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            if user_id not in self.users:
                logger.warning(f"User {user_id} not found. Returning global average.")
                return self.user_item_matrix.values.mean()
            
            if item_id not in self.items:
                logger.warning(f"Item {item_id} not found. Returning user average.")
                return self.user_means[user_id]
            
            # Get similar users who have rated this item
            user_similarities = self.user_similarity_matrix.loc[user_id]
            item_ratings = self.user_item_matrix[item_id]
            
            # Find users who have rated this item and are similar to target user
            rated_users = item_ratings[item_ratings > 0].index
            similar_users = user_similarities[rated_users]
            similar_users = similar_users[similar_users > 0]
            
            if len(similar_users) == 0:
                # No similar users found, return user's average rating
                return self.user_means[user_id]
            
            # Calculate weighted average using mean-centered ratings
            numerator = 0
            denominator = 0
            
            for similar_user in similar_users.index:
                similarity = similar_users[similar_user]
                rating = item_ratings[similar_user]
                user_mean = self.user_means[similar_user]
                
                numerator += similarity * (rating - user_mean)
                denominator += abs(similarity)
            
            if denominator == 0:
                return self.user_means[user_id]
            
            # Predicted rating = user_mean + weighted_average_of_deviations
            predicted_rating = self.user_means[user_id] + (numerator / denominator)
            
            # Ensure rating is within valid range
            min_rating, max_rating = MODEL_CONFIG['rating_scale']
            predicted_rating = np.clip(predicted_rating, min_rating, max_rating)
            
            return predicted_rating
            
        except Exception as e:
            logger.error(f"Error predicting rating for user {user_id}, item {item_id}: {str(e)}")
            return self.user_means.get(user_id, self.user_item_matrix.values.mean())
    
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
            if self.user_similarity_matrix is None:
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
            if self.user_similarity_matrix is None:
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
    
    def get_similar_users(self, user_id: int, n_users: int = 10) -> List[Tuple[int, float]]:
        """
        Get most similar users to a given user
        
        Args:
            user_id (int): User ID
            n_users (int): Number of similar users to return
            
        Returns:
            List[Tuple[int, float]]: List of (user_id, similarity_score) tuples
        """
        try:
            if self.user_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            if user_id not in self.users:
                logger.warning(f"User {user_id} not found")
                return []
            
            similarities = self.user_similarity_matrix.loc[user_id]
            similarities = similarities[similarities > 0]  # Only positive similarities
            
            # Sort by similarity and return top-N
            sorted_similarities = similarities.sort_values(ascending=False)
            similar_users = [(int(uid), float(sim)) for uid, sim in sorted_similarities.head(n_users).items()]
            
            return similar_users
            
        except Exception as e:
            logger.error(f"Error finding similar users for user {user_id}: {str(e)}")
            return []
    
    def get_model_statistics(self) -> Dict:
        """
        Get model statistics
        
        Returns:
            Dict: Model statistics
        """
        if self.user_similarity_matrix is None:
            return {"error": "Model not fitted"}
        
        similarity_values = self.user_similarity_matrix.values
        similarity_values = similarity_values[similarity_values > 0]  # Exclude zeros and diagonal
        
        stats = {
            "n_users": len(self.users),
            "n_items": len(self.items),
            "avg_similarity": float(similarity_values.mean()) if len(similarity_values) > 0 else 0,
            "max_similarity": float(similarity_values.max()) if len(similarity_values) > 0 else 0,
            "min_similarity": float(similarity_values.min()) if len(similarity_values) > 0 else 0,
            "similarity_sparsity": float(np.sum(self.user_similarity_matrix.values == 0) / self.user_similarity_matrix.size),
            "avg_user_rating": float(self.user_means.mean())
        }
        
        return stats

# Convenience function for easy usage
def create_user_based_model(user_item_matrix: pd.DataFrame) -> UserBasedCF:
    """
    Create and fit a User-Based Collaborative Filtering model
    
    Args:
        user_item_matrix (pd.DataFrame): User-item rating matrix
        
    Returns:
        UserBasedCF: Fitted model
    """
    model = UserBasedCF()
    model.fit(user_item_matrix)
    return model

if __name__ == "__main__":
    # Test the user-based CF model
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
        model = create_user_based_model(user_item_matrix)
        
        # Test predictions
        test_user = 1
        recommendations = model.recommend_items(test_user, n_recommendations=5)
        similar_users = model.get_similar_users(test_user, n_users=3)
        
        print("User-Based CF test successful!")
        print(f"Recommendations for user {test_user}: {recommendations}")
        print(f"Similar users to user {test_user}: {similar_users}")
        print(f"Model statistics: {model.get_model_statistics()}")
        
    except Exception as e:
        print(f"Error testing User-Based CF: {e}")

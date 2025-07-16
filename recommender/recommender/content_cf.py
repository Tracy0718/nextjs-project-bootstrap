"""
Content-Based Filtering Module
Implements content-based recommendations using TF-IDF vectorization
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import sys
import os
import re

# Add parent directory to path to import config and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, TFIDF_PARAMS
from utils.logger import get_logger, log_performance

logger = get_logger(__name__)

class ContentBasedCF:
    """
    Content-Based Collaborative Filtering implementation using TF-IDF
    """
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.content_similarity_matrix = None
        self.item_profiles = None
        self.user_profiles = None
        self.data = None
        self.items = None
        self.users = None
        
    @log_performance
    def fit(self, data: pd.DataFrame) -> 'ContentBasedCF':
        """
        Fit the content-based filtering model
        
        Args:
            data (pd.DataFrame): Dataset with user_id, item_id, rating, and content fields
            
        Returns:
            ContentBasedCF: Fitted model instance
        """
        try:
            logger.info("Fitting Content-Based Filtering model...")
            
            self.data = data.copy()
            self.users = data['user_id'].unique().tolist()
            self.items = data['item_id'].unique().tolist()
            
            # Build TF-IDF matrix from item content
            self._build_tfidf_matrix()
            
            # Compute content similarity matrix
            self._compute_content_similarity()
            
            # Build item profiles
            self._build_item_profiles()
            
            # Build user profiles based on their rating history
            self._build_user_profiles()
            
            logger.info(f"Content-Based CF model fitted successfully. Items: {len(self.items)}, Users: {len(self.users)}")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting Content-Based CF model: {str(e)}")
            raise
    
    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix from item content"""
        logger.info("Building TF-IDF matrix...")
        
        # Get unique items with their content
        item_content = self.data.groupby('item_id').first().reset_index()
        
        # Ensure we have the combined_content field
        if 'combined_content' not in item_content.columns:
            item_content['combined_content'] = (
                item_content['title'].fillna('') + ' ' + 
                item_content['category'].fillna('') + ' ' + 
                item_content['description'].fillna('')
            ).str.strip()
        
        # Initialize TF-IDF vectorizer with parameters from config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_PARAMS.get('max_features', 5000),
            stop_words=TFIDF_PARAMS.get('stop_words', 'english'),
            ngram_range=TFIDF_PARAMS.get('ngram_range', (1, 2)),
            min_df=TFIDF_PARAMS.get('min_df', 2),
            max_df=TFIDF_PARAMS.get('max_df', 0.8),
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform the content
        content_texts = item_content['combined_content'].fillna('').tolist()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_texts)
        
        # Store item order for indexing
        self.item_order = item_content['item_id'].tolist()
        
        logger.info(f"TF-IDF matrix built. Shape: {self.tfidf_matrix.shape}, "
                   f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
    
    def _compute_content_similarity(self):
        """Compute cosine similarity matrix for items based on content"""
        logger.info("Computing content similarity matrix...")
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Convert to DataFrame for easier indexing
        self.content_similarity_matrix = pd.DataFrame(
            similarity_matrix,
            index=self.item_order,
            columns=self.item_order
        )
        
        # Set diagonal to 0 (item shouldn't be similar to itself for recommendation purposes)
        np.fill_diagonal(self.content_similarity_matrix.values, 0)
        
        logger.info(f"Content similarity matrix computed. Average similarity: {similarity_matrix.mean():.4f}")
    
    def _build_item_profiles(self):
        """Build item profiles from TF-IDF vectors"""
        logger.info("Building item profiles...")
        
        # Convert TF-IDF matrix to dense format for easier manipulation
        tfidf_dense = self.tfidf_matrix.toarray()
        
        # Create item profiles dictionary
        self.item_profiles = {}
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        for i, item_id in enumerate(self.item_order):
            # Get top features for this item
            item_vector = tfidf_dense[i]
            top_indices = np.argsort(item_vector)[-10:][::-1]  # Top 10 features
            
            profile = {
                'vector': item_vector,
                'top_features': [(feature_names[idx], item_vector[idx]) for idx in top_indices if item_vector[idx] > 0]
            }
            
            self.item_profiles[item_id] = profile
        
        logger.info(f"Item profiles built for {len(self.item_profiles)} items")
    
    def _build_user_profiles(self):
        """Build user profiles based on their rating history"""
        logger.info("Building user profiles...")
        
        self.user_profiles = {}
        
        for user_id in self.users:
            user_data = self.data[self.data['user_id'] == user_id]
            
            # Weight item profiles by user ratings
            user_vector = np.zeros(self.tfidf_matrix.shape[1])
            total_weight = 0
            
            for _, row in user_data.iterrows():
                item_id = row['item_id']
                rating = row['rating']
                
                if item_id in self.item_profiles:
                    # Weight by rating (higher ratings contribute more)
                    weight = rating / 5.0  # Normalize to 0-1
                    user_vector += weight * self.item_profiles[item_id]['vector']
                    total_weight += weight
            
            # Normalize user profile
            if total_weight > 0:
                user_vector = user_vector / total_weight
            
            # Get top features for this user
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            top_indices = np.argsort(user_vector)[-10:][::-1]  # Top 10 features
            
            profile = {
                'vector': user_vector,
                'top_features': [(feature_names[idx], user_vector[idx]) for idx in top_indices if user_vector[idx] > 0],
                'n_ratings': len(user_data)
            }
            
            self.user_profiles[user_id] = profile
        
        logger.info(f"User profiles built for {len(self.user_profiles)} users")
    
    @log_performance
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a specific user-item pair based on content similarity
        
        Args:
            user_id (int): User ID
            item_id (int): Item ID
            
        Returns:
            float: Predicted rating
        """
        try:
            if self.user_profiles is None or self.item_profiles is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            if user_id not in self.user_profiles:
                logger.warning(f"User {user_id} not found. Returning global average.")
                return self.data['rating'].mean()
            
            if item_id not in self.item_profiles:
                logger.warning(f"Item {item_id} not found. Returning global average.")
                return self.data['rating'].mean()
            
            # Calculate similarity between user profile and item profile
            user_vector = self.user_profiles[user_id]['vector']
            item_vector = self.item_profiles[item_id]['vector']
            
            # Compute cosine similarity
            similarity = cosine_similarity([user_vector], [item_vector])[0][0]
            
            # Convert similarity to rating scale
            # Use user's average rating as baseline and adjust based on similarity
            user_data = self.data[self.data['user_id'] == user_id]
            user_avg_rating = user_data['rating'].mean()
            
            # Scale similarity to rating adjustment (-1 to +1 range)
            rating_adjustment = similarity * 2 - 1  # Convert 0-1 to -1 to +1
            
            # Predict rating
            min_rating, max_rating = MODEL_CONFIG['rating_scale']
            predicted_rating = user_avg_rating + rating_adjustment
            predicted_rating = np.clip(predicted_rating, min_rating, max_rating)
            
            return predicted_rating
            
        except Exception as e:
            logger.error(f"Error predicting rating for user {user_id}, item {item_id}: {str(e)}")
            return self.data['rating'].mean()
    
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
            if self.user_profiles is None or self.item_profiles is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            if items is None:
                items = self.items
            
            predictions = {}
            
            for item_id in items:
                predictions[item_id] = self.predict_rating(user_id, item_id)
            
            logger.info(f"Generated {len(predictions)} content-based predictions for user {user_id}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting ratings for user {user_id}: {str(e)}")
            return {}
    
    @log_performance
    def recommend_items(self, user_id: int, n_recommendations: int = 10, 
                       exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend top-N items for a user based on content similarity
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            exclude_rated (bool): Whether to exclude already rated items
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, predicted_rating) tuples
        """
        try:
            if self.user_profiles is None or self.item_profiles is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            # Get all item predictions
            predictions = self.predict_ratings_for_user(user_id)
            
            if exclude_rated:
                # Exclude items already rated by the user
                user_data = self.data[self.data['user_id'] == user_id]
                rated_items = user_data['item_id'].tolist()
                predictions = {item_id: rating for item_id, rating in predictions.items() 
                             if item_id not in rated_items}
            
            # Sort by predicted rating and return top-N
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            recommendations = sorted_predictions[:n_recommendations]
            
            logger.info(f"Generated {len(recommendations)} content-based recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating content-based recommendations for user {user_id}: {str(e)}")
            return []
    
    def get_similar_items(self, item_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """
        Get most similar items to a given item based on content
        
        Args:
            item_id (int): Item ID
            n_items (int): Number of similar items to return
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, similarity_score) tuples
        """
        try:
            if self.content_similarity_matrix is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            if item_id not in self.content_similarity_matrix.index:
                logger.warning(f"Item {item_id} not found")
                return []
            
            similarities = self.content_similarity_matrix.loc[item_id]
            similarities = similarities[similarities > 0]  # Only positive similarities
            
            # Sort by similarity and return top-N
            sorted_similarities = similarities.sort_values(ascending=False)
            similar_items = [(int(iid), float(sim)) for iid, sim in sorted_similarities.head(n_items).items()]
            
            return similar_items
            
        except Exception as e:
            logger.error(f"Error finding similar items for item {item_id}: {str(e)}")
            return []
    
    def get_user_preferences(self, user_id: int) -> Dict:
        """
        Get user preferences based on their profile
        
        Args:
            user_id (int): User ID
            
        Returns:
            Dict: User preferences and top features
        """
        try:
            if user_id not in self.user_profiles:
                return {"error": f"User {user_id} not found"}
            
            profile = self.user_profiles[user_id]
            user_data = self.data[self.data['user_id'] == user_id]
            
            preferences = {
                "user_id": user_id,
                "n_ratings": profile['n_ratings'],
                "avg_rating": user_data['rating'].mean(),
                "top_content_features": profile['top_features'][:5],
                "favorite_categories": user_data['category'].value_counts().head(3).to_dict(),
                "rating_distribution": user_data['rating'].value_counts().sort_index().to_dict()
            }
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting preferences for user {user_id}: {str(e)}")
            return {"error": str(e)}
    
    def get_item_features(self, item_id: int) -> Dict:
        """
        Get item features and content analysis
        
        Args:
            item_id (int): Item ID
            
        Returns:
            Dict: Item features and analysis
        """
        try:
            if item_id not in self.item_profiles:
                return {"error": f"Item {item_id} not found"}
            
            profile = self.item_profiles[item_id]
            item_data = self.data[self.data['item_id'] == item_id].iloc[0]
            
            features = {
                "item_id": item_id,
                "title": item_data['title'],
                "category": item_data['category'],
                "description": item_data['description'],
                "top_content_features": profile['top_features'][:5],
                "avg_rating": self.data[self.data['item_id'] == item_id]['rating'].mean(),
                "n_ratings": len(self.data[self.data['item_id'] == item_id])
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting features for item {item_id}: {str(e)}")
            return {"error": str(e)}
    
    def get_model_statistics(self) -> Dict:
        """
        Get model statistics
        
        Returns:
            Dict: Model statistics
        """
        if self.content_similarity_matrix is None:
            return {"error": "Model not fitted"}
        
        similarity_values = self.content_similarity_matrix.values
        similarity_values = similarity_values[similarity_values > 0]  # Exclude zeros and diagonal
        
        stats = {
            "n_users": len(self.users),
            "n_items": len(self.items),
            "tfidf_features": self.tfidf_matrix.shape[1],
            "avg_content_similarity": float(similarity_values.mean()) if len(similarity_values) > 0 else 0,
            "max_content_similarity": float(similarity_values.max()) if len(similarity_values) > 0 else 0,
            "min_content_similarity": float(similarity_values.min()) if len(similarity_values) > 0 else 0,
            "vocabulary_size": len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0,
            "avg_profile_features": np.mean([len(profile['top_features']) for profile in self.user_profiles.values()]) if self.user_profiles else 0
        }
        
        return stats

# Convenience function for easy usage
def create_content_based_model(data: pd.DataFrame) -> ContentBasedCF:
    """
    Create and fit a Content-Based Filtering model
    
    Args:
        data (pd.DataFrame): Dataset with content information
        
    Returns:
        ContentBasedCF: Fitted model
    """
    model = ContentBasedCF()
    model.fit(data)
    return model

if __name__ == "__main__":
    # Test the content-based CF model
    try:
        # Create sample data for testing
        np.random.seed(42)
        
        # Sample data with content
        data = []
        categories = ['Electronics', 'Books', 'Sports', 'Home']
        
        for user_id in range(1, 6):
            for item_id in range(101, 121):
                if np.random.random() > 0.3:  # 70% chance of rating
                    rating = np.random.uniform(1, 5)
                    category = np.random.choice(categories)
                    title = f"Product {item_id}"
                    description = f"This is a {category.lower()} product with great features"
                    
                    data.append([user_id, item_id, rating, title, category, description])
        
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'title', 'category', 'description'])
        
        # Test the model
        model = create_content_based_model(df)
        
        # Test predictions
        test_user = 1
        recommendations = model.recommend_items(test_user, n_recommendations=5)
        similar_items = model.get_similar_items(101, n_items=3)
        user_prefs = model.get_user_preferences(test_user)
        
        print("Content-Based CF test successful!")
        print(f"Recommendations for user {test_user}: {recommendations}")
        print(f"Similar items to item 101: {similar_items}")
        print(f"User preferences for user {test_user}: {user_prefs}")
        print(f"Model statistics: {model.get_model_statistics()}")
        
    except Exception as e:
        print(f"Error testing Content-Based CF: {e}")

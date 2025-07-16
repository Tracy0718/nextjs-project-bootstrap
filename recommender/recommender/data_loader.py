"""
Data Loading and Preprocessing Module for Hybrid Collaborative Filtering Recommendation System
Handles loading, cleaning, and preprocessing of the dataset
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path to import config and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_FULL_PATH, COLLAB_PARAMS, MODEL_CONFIG
from utils.logger import get_logger, log_performance

logger = get_logger(__name__)

class DataLoader:
    """
    Data loading and preprocessing class for the recommendation system
    """
    
    def __init__(self):
        self.data = None
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.users = None
        self.items = None
        
    @log_performance
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and preprocess the dataset
        
        Args:
            file_path (str, optional): Path to the CSV file. Uses config default if None.
            
        Returns:
            pd.DataFrame: Cleaned and preprocessed dataset
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data format is invalid
        """
        try:
            if file_path is None:
                file_path = DATA_FULL_PATH
                
            logger.info(f"Loading data from: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Load the CSV file
            self.data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Validate required columns
            required_columns = ['user_id', 'item_id', 'rating', 'title', 'category', 'description']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Preprocess the data
            self.data = self._preprocess_data(self.data)
            
            logger.info("Data preprocessing completed successfully")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the dataset
        
        Args:
            data (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        logger.info("Starting data preprocessing...")
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Clean text fields
        data = self._clean_text_fields(data)
        
        # Normalize ratings
        data = self._normalize_ratings(data)
        
        # Filter users and items based on minimum ratings
        data = self._filter_sparse_data(data)
        
        # Create combined content field for content-based filtering
        data['combined_content'] = (
            data['title'].fillna('') + ' ' + 
            data['category'].fillna('') + ' ' + 
            data['description'].fillna('')
        ).str.strip()
        
        logger.info(f"Data preprocessing completed. Final shape: {data.shape}")
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # Fill missing text fields with empty strings
        text_columns = ['title', 'category', 'description', 'image_url']
        for col in text_columns:
            if col in data.columns:
                data[col] = data[col].fillna('')
        
        # Drop rows with missing user_id, item_id, or rating
        initial_rows = len(data)
        data = data.dropna(subset=['user_id', 'item_id', 'rating'])
        dropped_rows = initial_rows - len(data)
        
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing critical values")
        
        return data
    
    def _clean_text_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize text fields"""
        logger.info("Cleaning text fields...")
        
        text_columns = ['title', 'category', 'description']
        
        for col in text_columns:
            if col in data.columns:
                # Convert to lowercase
                data[col] = data[col].str.lower()
                
                # Remove extra whitespace
                data[col] = data[col].str.strip()
                
                # Remove special characters (keep alphanumeric and spaces)
                data[col] = data[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', str(x)) if pd.notna(x) else '')
                
                # Remove multiple spaces
                data[col] = data[col].apply(lambda x: re.sub(r'\s+', ' ', str(x)) if pd.notna(x) else '')
        
        return data
    
    def _normalize_ratings(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize ratings to the specified scale"""
        logger.info("Normalizing ratings...")
        
        min_rating, max_rating = MODEL_CONFIG['rating_scale']
        
        # Ensure ratings are within the expected range
        data['rating'] = data['rating'].clip(min_rating, max_rating)
        
        # Log rating statistics
        logger.info(f"Rating statistics - Min: {data['rating'].min()}, Max: {data['rating'].max()}, Mean: {data['rating'].mean():.2f}")
        
        return data
    
    def _filter_sparse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out users and items with too few ratings"""
        logger.info("Filtering sparse data...")
        
        initial_users = data['user_id'].nunique()
        initial_items = data['item_id'].nunique()
        initial_ratings = len(data)
        
        # Filter users with minimum ratings
        user_counts = data['user_id'].value_counts()
        valid_users = user_counts[user_counts >= COLLAB_PARAMS['min_ratings_per_user']].index
        data = data[data['user_id'].isin(valid_users)]
        
        # Filter items with minimum ratings
        item_counts = data['item_id'].value_counts()
        valid_items = item_counts[item_counts >= COLLAB_PARAMS['min_ratings_per_item']].index
        data = data[data['item_id'].isin(valid_items)]
        
        final_users = data['user_id'].nunique()
        final_items = data['item_id'].nunique()
        final_ratings = len(data)
        
        logger.info(f"Filtering results - Users: {initial_users} -> {final_users}, "
                   f"Items: {initial_items} -> {final_items}, "
                   f"Ratings: {initial_ratings} -> {final_ratings}")
        
        return data
    
    @log_performance
    def create_matrices(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create user-item and item-user matrices
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: User-item matrix and item-user matrix
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Creating user-item and item-user matrices...")
        
        # Create user-item matrix
        self.user_item_matrix = self.data.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        # Create item-user matrix (transpose)
        self.item_user_matrix = self.user_item_matrix.T
        
        # Store unique users and items
        self.users = self.user_item_matrix.index.tolist()
        self.items = self.user_item_matrix.columns.tolist()
        
        logger.info(f"Matrices created - Users: {len(self.users)}, Items: {len(self.items)}")
        
        return self.user_item_matrix, self.item_user_matrix
    
    def get_user_ratings(self, user_id: int) -> pd.Series:
        """
        Get ratings for a specific user
        
        Args:
            user_id (int): User ID
            
        Returns:
            pd.Series: User's ratings
        """
        if self.user_item_matrix is None:
            raise ValueError("Matrices not created. Call create_matrices() first.")
        
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found in the dataset")
        
        return self.user_item_matrix.loc[user_id]
    
    def get_item_ratings(self, item_id: int) -> pd.Series:
        """
        Get ratings for a specific item
        
        Args:
            item_id (int): Item ID
            
        Returns:
            pd.Series: Item's ratings
        """
        if self.item_user_matrix is None:
            raise ValueError("Matrices not created. Call create_matrices() first.")
        
        if item_id not in self.items:
            raise ValueError(f"Item {item_id} not found in the dataset")
        
        return self.item_user_matrix.loc[item_id]
    
    def get_item_details(self, item_id: int) -> dict:
        """
        Get details for a specific item
        
        Args:
            item_id (int): Item ID
            
        Returns:
            dict: Item details
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        item_data = self.data[self.data['item_id'] == item_id].iloc[0]
        
        return {
            'item_id': item_data['item_id'],
            'title': item_data['title'],
            'category': item_data['category'],
            'description': item_data['description'],
            'image_url': item_data.get('image_url', ''),
            'avg_rating': self.data[self.data['item_id'] == item_id]['rating'].mean()
        }
    
    def get_data_statistics(self) -> dict:
        """
        Get dataset statistics
        
        Returns:
            dict: Dataset statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stats = {
            'total_ratings': len(self.data),
            'unique_users': self.data['user_id'].nunique(),
            'unique_items': self.data['item_id'].nunique(),
            'rating_range': (self.data['rating'].min(), self.data['rating'].max()),
            'avg_rating': self.data['rating'].mean(),
            'sparsity': 1 - (len(self.data) / (self.data['user_id'].nunique() * self.data['item_id'].nunique())),
            'categories': self.data['category'].unique().tolist()
        }
        
        return stats

# Convenience function for easy import
def load_and_preprocess_data(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, DataLoader]:
    """
    Convenience function to load and preprocess data
    
    Args:
        file_path (str, optional): Path to the CSV file
        
    Returns:
        Tuple[pd.DataFrame, DataLoader]: Processed data and DataLoader instance
    """
    loader = DataLoader()
    data = loader.load_data(file_path)
    loader.create_matrices()
    return data, loader

if __name__ == "__main__":
    # Test the data loader
    try:
        data, loader = load_and_preprocess_data()
        print("Data loading test successful!")
        print(f"Dataset shape: {data.shape}")
        print(f"Statistics: {loader.get_data_statistics()}")
    except Exception as e:
        print(f"Error testing data loader: {e}")

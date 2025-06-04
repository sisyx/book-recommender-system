import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import joblib
import os

logger = logging.getLogger(__name__)

@dataclass
class ContentBasedConfig:
    similarity_metric: str = "cosine"  # cosine, euclidean, hamming
    top_k_similar: int = 10
    similarity_threshold: float = 0.4
    batch_size: int = 1000
    # use_sparse: bool = True
    scaler_type: str = "minmax"  # minmax, standard, none
    save_path: str = "models/cb.joblib"


class ContentBasedFilter:
    def __init__(self, config: ContentBasedConfig):
        self.config = config
        self.features_df = None
        self.similarity_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.scaler = None
        self.is_fitted = False
        
        # Initialize scaler
        if self.config.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.config.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = None

    def _prepare_features(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """
        prepare and encode features for similarity computation

        Args:
            books_df: dataframe with book features

        Returns:
            Processes features dataframe
        """
        logger.info("Preparing content-based features...")


        # Create a copy to avoid modifying original
        df: pd.DataFrame = books_df.copy()

        logger.info(df.head())

        # handle missing values
        df = df.fillna(0)

        # Separate numerical and categorical features
        numerical_cols = ['rating', 'publish_year', 'counts_of_review']
        categorical_cols = ['language']

        # Keep ID column separate
        item_ids = df["id"].copy()
        df_features = df.drop(columns=["id"])

        processed_features = []
        feature_names = []

        # Process numerical features

        for col in numerical_cols:
            if col in df_features.columns:
                if col == "publish_year":
                    df_features[f"{col}_period"] = df_features[col].apply(self._get_period_index)
                    # One-hot encode periods
                    period_encoded = pd.get_dummies(df_features[f'{col}_period'], prefix='period')
                    processed_features.append(period_encoded)
                    feature_names.extend(period_encoded.columns.tolist())
                else:
                    # Scale numerical features
                    feature_data = df_features[[col]].values
                    if self.scaler is not None:
                        if not hasattr(self.scaler, 'scale_'):  # Not fitted yet
                            feature_data = self.scaler.fit_transform(feature_data)
                        else:
                            feature_data = self.scaler.transform(feature_data)
                    
                    feature_df = pd.DataFrame(feature_data, columns=[col], index=df_features.index)
                    processed_features.append(feature_df)
                    feature_names.append(col)

            
        # Process categorical features
        for col in categorical_cols:
            if col in df_features.columns:
                encoded = pd.get_dummies(df_features[col], prefix=col)
                processed_features.append(encoded)
                feature_names.extend(encoded.columns.tolist())
        

        # Combine all features
        if processed_features:
            final_features = pd.concat(processed_features, axis=1)
            final_features['id'] = item_ids
        else:
            logger.warning("No features found to process")
            final_features = pd.DataFrame({'id': item_ids})

        logger.info(f"Processed {len(final_features.columns)-1} features for {len(final_features)} items")
        return final_features
    

    def _get_period_index(self, year: int) -> int: # must update
        """Convert publish year to literature period index (reuse your logic)"""
        if year < 1800:
            return 0
        elif year < 1900:
            return 1
        elif year < 1950:
            return 2
        elif year < 2000:
            return 3
        else:
            return 4

    def _compute_similarity_batch(self, features_matrix: np.ndarray,
                                  batch_size: int = 1000) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute Sparse similarity matrix in batches to handle memory efficiently
        """

        n_items = features_matrix.shape[0]
        similarities = {}

        for idx in range(0, n_items, batch_size):
            end_idx = min(idx + batch_size, n_items)
            current_batch = features_matrix[idx:end_idx]
            logger.info(f"processing items {idx}:{end_idx}")

            if self.config.similarity_metric == "cosine":
                # Cosine similarity is more efficient for sparse data
                batch_similarity = cosine_similarity(current_batch, features_matrix)

            else:
                # For other metrics, use pairwise_distances
                if self.config.similarity_metric == "euclidean":
                    distances = pairwise_distances(current_batch, features_matrix, metric='euclidean')
                elif self.config.similarity_metric == "hamming":
                    distances = pairwise_distances(current_batch, features_matrix, metric='hamming')
                else:
                    raise ValueError(f"Unsupported similarity metric: {self.config.similarity_metric}")

                # Convert distances to similarities
                batch_similarity = 1 / (1 + distances)

            for local_idx, global_idx in enumerate(range(idx, end_idx)):
                item_id = self.idx_to_item[global_idx]
                similarities[item_id] = self._extract_top_k_similarities(
                    batch_similarity[local_idx], global_idx
                )


        return similarities

    def _extract_top_k_similarities(self, similarity_row: np.ndarray, item_idx: int) -> List[Tuple[int, float]]:
        """Extract top-k similarities from a similarity row"""
        # Remove self-similarity
        similarity_row[item_idx] = 0
        
        # Find items above threshold
        above_threshold = similarity_row >= self.config.similarity_threshold
        candidate_indices = np.where(above_threshold)[0]
        candidate_scores = similarity_row[above_threshold]
        
        if len(candidate_indices) == 0:
            return []
        
        # Keep only top-k
        if len(candidate_indices) > self.config.top_k_similar:
            top_k_indices = np.argsort(candidate_scores)[-self.config.top_k_similar:][::-1]
            final_indices = candidate_indices[top_k_indices]
            final_scores = candidate_scores[top_k_indices]
        else:
            sort_indices = np.argsort(candidate_scores)[::-1]
            final_indices = candidate_indices[sort_indices]
            final_scores = candidate_scores[sort_indices]
        
        # Convert to item IDs
        result = []
        for idx, score in zip(final_indices, final_scores):
            similar_item_id = self.idx_to_item[idx]
            result.append((similar_item_id, float(score)))
        
        return result

    def fit(self, books_df: pd.DataFrame):
        """
        Train the content-based filtering model
        
        Args:
            books_df: DataFrame with book features including 'id', 'rating', 'publish_year', 'language'
        """
        logger.info("Training content-based filtering model...")
        self.features_df = self._prepare_features(books_df)

        # create item mappings
        item_ids = self.features_df["id"].values
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}

        # Extract feature matrix (exclude ID column)
        feature_columns = [col for col in self.features_df.columns if col != 'id']
        features_matrix = self.features_df[feature_columns].values
        
        # Compute sparse similarity matrix
        logger.info("Computing similarity matrix...")
        self.similarity_matrix = self._compute_similarity_batch(features_matrix, batch_size=1000)

        self.is_fitted = True
        logger.info(f"Content-based model trained with {len(self.similarity_matrix)} items")

    def get_similar_items(self, item_id: int, top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Get similar items for a given item
        
        Args:
            item_id: Item ID to find similarities for
            top_k: Number of similar items to return (None for all)
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similarities")
        
        if item_id not in self.similarity_matrix:
            logger.warning(f"Item {item_id} not found in similarity matrix")
            return []
        
        similar_items = self.similarity_matrix[item_id]
        
        if top_k is not None:
            similar_items = similar_items[:top_k]
        
        return similar_items

    def save(self):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        filepath: str = self.config.save_path
        
        model_data = {
            'config': self.config,
            'similarity_matrix': self.similarity_matrix,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self):
        """Load a trained model"""
        filepath: str = self.config.save_path

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model_data = joblib.load(filepath)
        
        self.config = model_data['config']
        self.similarity_matrix = model_data['similarity_matrix']
        self.item_to_idx = model_data['item_to_idx']
        self.idx_to_item = model_data['idx_to_item']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
    
    def _get_book_id(self, book_name: str, books_df: pd.DataFrame) -> int:
        normalized_name = book_name.strip().lower()
        books_df['normalized_name'] = books_df['name'].str.strip().str.lower()
        
        exact_matches = books_df[books_df['normalized_name'] == normalized_name]
        
        if len(exact_matches) >= 1:
            return exact_matches.iloc[0]['id']
    
    def _get_book_name(self, book_id: str, books_df: pd.DataFrame) -> int:
        matches = books_df[books_df['id'] == book_id]
        
        if len(matches) >= 1:
            return matches.iloc[0]['name']

    
    # def recommend_based_on_profile(self, user_profile: List[Tuple[int, float]], 
    #                               top_k: int = 10, exclude_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
    #     """
    #     Recommend items based on user's item profile (items they've interacted with)
        
    #     Args:
    #         user_profile: List of (item_id, rating/weight) tuples representing user preferences
    #         top_k: Number of recommendations to return
    #         exclude_items: Items to exclude from recommendations
            
    #     Returns:
    #         List of (item_id, predicted_score) tuples
    #     """
    #     if not self.is_fitted:
    #         raise ValueError("Model must be fitted before making recommendations")
        
    #     if not user_profile:
    #         return []
        
    #     exclude_items = exclude_items or []
    #     item_scores = {}
        
    #     # For each item in user's profile
    #     for item_id, user_rating in user_profile:
    #         if item_id in self.similarity_matrix:
    #             similar_items = self.similarity_matrix[item_id]
                
    #             # Weight similar items by user's rating and similarity score
    #             for similar_item_id, similarity_score in similar_items:
    #                 if similar_item_id not in exclude_items:
    #                     if similar_item_id not in item_scores:
    #                         item_scores[similar_item_id] = 0
                        
    #                     # Weighted score: user_rating * similarity_score
    #                     item_scores[similar_item_id] += user_rating * similarity_score
        
    #     # Sort by predicted score and return top-k
    #     recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    #     return recommendations[:top_k]

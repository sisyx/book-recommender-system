import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for collaborative filtering training"""
    num_features: int = 20  # Reduced complexity
    learning_rate: float = 0.001  # Much lower learning rate
    regularization: float = 0.01  # Higher regularization
    max_iterations: int = 1000
    batch_size: int = 1024
    early_stopping_patience: int = 15  # Less patience for quicker stopping
    validation_split: float = 0.1
    min_improvement: float = 1e-4
    learning_rate_decay: float = 0.95  # Learning rate decay
    decay_every: int = 50  # Decay every 50 epochs
    save_path: str = "models/cf.joblib"

class CollaborativeFilter:
    """
    Matrix factorization based Collaborative filtering using Stochastic Gradient Descent
    with enhanced overfitting prevention
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_lr = config.learning_rate  # Track current learning rate
        self.user_features = None
        self.item_features = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = 0.0

        # Mapping for sparse matrix
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Fitted Flag
        self.is_fitted = False

        # Best model parameters for early stopping
        self.best_params = None

    def _create_sparse_matrix(self, ratings_df: pd.DataFrame) -> Tuple[csr_matrix, np.ndarray]:
        """
        Create sparse user-item rating matrix from ratings dataframe

        Returns:
            ratings_matrix: sparse matrix of shape [n_users, n_items]
            ratings_array: array of (user_idx, item_idx, rating) for training
        """

        logger.info("Creating sparse rating matrix...")

        # Create user and item mappings
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df["book_id"].unique()

        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)

        logger.info(f"Matrix dimensions: {n_users} users x {n_items} items")

        user_indices = ratings_df["user_id"].map(self.user_to_idx).values
        item_indices = ratings_df["book_id"].map(self.item_to_idx).values
        ratings = ratings_df["rating"].values

        # create sparse matrix
        ratings_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )

        # create training array
        ratings_array = np.column_stack([user_indices, item_indices, ratings])

        return ratings_matrix, ratings_array

    def _initialize_parameters(self, n_users: int, n_items: int):
        """Initialize model parameters with smaller random values"""
        logger.info("Initializing model parameters...")

        # Use smaller initialization to prevent large updates
        scale = 0.01  # Much smaller initialization

        self.user_features = np.random.normal(0, scale, (n_users, self.config.num_features))
        self.item_features = np.random.normal(0, scale, (n_items, self.config.num_features))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = 0.0

    def _rescale_ratings(self, ratings_df: pd.DataFrame):
        min_rating = ratings_df["rating"].min()
        max_rating = ratings_df["rating"].max()
        
        ratings_df["rating"] = 1- ((max_rating - ratings_df["rating"]) / (max_rating - min_rating + 1))

    def _train_val_split(self, ratings_df: pd.DataFrame, ratings_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split ratings into train and validation sets by selecting 2 items from each of the top 100 users"""
        # 1. Find top 100 users
        user_rating_counts = ratings_df['user_id'].value_counts()
        top_users = user_rating_counts.head(100).index

        # 2. Get validation indexes
        val_mask = np.zeros(len(ratings_df), dtype=bool)
        for user in top_users:
            user_ratings = ratings_df[ratings_df['user_id'] == user]
            sampled_idx = user_ratings.sample(n=2, random_state=42).index
            val_mask[sampled_idx] = True

        # 3. Split the arrays
        validation_array = ratings_array[val_mask]
        train_array = ratings_array[~val_mask]  # Use boolean negation for train set

        return train_array, validation_array
        
    def _compute_loss(self, ratings_array: np.ndarray) -> float:
        """Compute RMSE loss for given ratings"""

        if len(ratings_array) == 0:
            return 0.0

        user_indices = ratings_array[:, 0].astype(int)
        item_indices = ratings_array[:, 1].astype(int)
        true_ratings = ratings_array[:, 2]

        predictions = (
            np.sum(self.user_features[user_indices] * self.item_features[item_indices], axis=1) +
            self.user_biases[user_indices] +
            self.item_biases[item_indices] +
            self.global_bias
        )

        # Compute RMSE
        mse = np.mean((true_ratings - predictions) ** 2)
        return np.sqrt(mse)
    
    def _compute_regularization_loss(self) -> float:
        """Compute L2 regularization loss"""
        reg_loss = (
            np.sum(self.user_features ** 2) +
            np.sum(self.item_features ** 2) +
            np.sum(self.user_biases ** 2) +
            np.sum(self.item_biases ** 2)
        )
        return self.config.regularization * reg_loss
    
    def _sgd_step(self, user_idx: int, item_idx: int, rating: float):
        """Single SGD step for one rating with gradient clipping"""
        # Current prediction
        prediction = (
            np.dot(self.user_features[user_idx], self.item_features[item_idx]) +
            self.user_biases[user_idx] +
            self.item_biases[item_idx] +
            self.global_bias
        )

        # Error
        error = rating - prediction

        # Store current values for updates
        user_features_old = self.user_features[user_idx].copy()
        item_features_old = self.item_features[item_idx].copy()

        # Compute gradients
        user_grad = error * item_features_old - self.config.regularization * user_features_old
        item_grad = error * user_features_old - self.config.regularization * item_features_old
        user_bias_grad = error - self.config.regularization * self.user_biases[user_idx]
        item_bias_grad = error - self.config.regularization * self.item_biases[item_idx]

        # Gradient clipping to prevent exploding gradients
        max_grad = 5.0
        user_grad = np.clip(user_grad, -max_grad, max_grad)
        item_grad = np.clip(item_grad, -max_grad, max_grad)
        user_bias_grad = np.clip(user_bias_grad, -max_grad, max_grad)
        item_bias_grad = np.clip(item_bias_grad, -max_grad, max_grad)

        # Update parameters with current learning rate
        self.user_features[user_idx] += self.current_lr * user_grad
        self.item_features[item_idx] += self.current_lr * item_grad
        self.user_biases[user_idx] += self.current_lr * user_bias_grad
        self.item_biases[item_idx] += self.current_lr * item_bias_grad

    def _update_learning_rate(self, epoch: int):
        """Update learning rate with decay"""
        if epoch > 0 and epoch % self.config.decay_every == 0:
            self.current_lr *= self.config.learning_rate_decay
            logger.info(f"Learning rate decayed to: {self.current_lr:.6f}")

    def _save_best_params(self):
        """Save current parameters as best parameters"""
        self.best_params = {
            'user_features': self.user_features.copy(),
            'item_features': self.item_features.copy(),
            'user_biases': self.user_biases.copy(),
            'item_biases': self.item_biases.copy(),
            'global_bias': self.global_bias
        }

    def _restore_best_params(self):
        """Restore best parameters"""
        if self.best_params is not None:
            self.user_features = self.best_params['user_features']
            self.item_features = self.best_params['item_features']
            self.user_biases = self.best_params['user_biases']
            self.item_biases = self.best_params['item_biases']
            self.global_bias = self.best_params['global_bias']

    def fit(self, ratings_df: pd.DataFrame):
        """
        Train the collaborative filtering model with aggressive overfitting prevention
        
        Args:
            ratings_df: DataFrame with columns ['user_id', 'book_id', 'rating']
        """
        logger.info("Starting collaborative filtering training...")
        
        # Rescale ratings to between 0-1
        self._rescale_ratings(ratings_df)
        
        print(ratings_df.info())

        # Create sparse matrix and training data
        ratings_matrix, ratings_array = self._create_sparse_matrix(ratings_df)

        # Split into train/validation
        train_ratings, val_ratings = self._train_val_split(ratings_df, ratings_array)

        # Initialize parameters
        n_users, n_items = ratings_matrix.shape
        self._initialize_parameters(n_users, n_items)

        # Compute global bias


        self.global_bias = np.mean(ratings_array[:, 2])

        logger.info(f"Training on {len(train_ratings)} ratings, validating on {len(val_ratings)} ratings")
        logger.info(f"Model complexity: {self.config.num_features} features")
        logger.info(f"Regularization: {self.config.regularization}")

        best_val_loss = float('inf')
        patience_counter = 0
        
        # Reset learning rate
        self.current_lr = self.config.learning_rate

        for epoch in range(self.config.max_iterations):
            # Update learning rate
            self._update_learning_rate(epoch)
            
            # Shuffle training data
            train_shuffled = train_ratings[np.random.permutation(len(train_ratings))]

            # Process in mini-batches
            for i in range(0, len(train_shuffled), self.config.batch_size):
                batch = train_shuffled[i:i + self.config.batch_size]
                
                for user_idx, item_idx, rating in batch:
                    self._sgd_step(int(user_idx), int(item_idx), rating)

            # Compute losses every 3 epochs for more frequent monitoring
            if epoch % 3 == 0:
                train_loss = self._compute_loss(train_ratings)
                val_loss = self._compute_loss(val_ratings) if len(val_ratings) > 0 else 0.0
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.learning_rates.append(self.current_lr)
                
                # Calculate gap between train and validation
                gap = val_loss - train_loss
                
                logger.info(f"Epoch {epoch}: Train RMSE = {train_loss:.4f}, Val RMSE = {val_loss:.4f}, Gap = {gap:.4f}, LR = {self.current_lr:.6f}")
                
                # Enhanced early stopping
                if val_loss < best_val_loss - self.config.min_improvement:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_best_params()
                    logger.info(f"New best validation RMSE: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    
                # Stop if validation loss is increasing and gap is too large
                if gap > 0.15 and patience_counter >= 3:  # Stop early if overfitting badly
                    logger.info(f"Stopping early due to large train-val gap: {gap:.4f}")
                    break
                    
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Always restore best parameters
        logger.info(f"Restoring best parameters (Val RMSE: {best_val_loss:.4f})")
        self._restore_best_params()
        
        self.is_fitted = True
        logger.info("Training completed!")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        logger.info(f"Loading Model From {self.config.save_path}")
        self.load_model()
        if not self.is_fitted:
            logger.info("No trained models found. please train the model first!")

        # Check if user/item exist in training data
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_bias  # Return global average for unknown users/items
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        prediction = (
            np.dot(self.user_features[user_idx], self.item_features[item_idx]) +
            self.user_biases[user_idx] +
            self.item_biases[item_idx] +
            self.global_bias
        )
        
        return prediction
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 10, 
                          exclude_rated: bool = True, ratings_df: Optional[pd.DataFrame] = None) -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for a user
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            ratings_df: Original ratings dataframe (needed if exclude_rated=True)
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get all items
        all_items = list(self.item_to_idx.keys())
        
        # Exclude already rated items if requested
        if exclude_rated and ratings_df is not None:
            user_rated_items = set(ratings_df[ratings_df['user_id'] == user_id]['book_id'].values)
            all_items = [item for item in all_items if item not in user_rated_items]
        
        # Compute predictions for all items
        predictions = []
        for item_id in all_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def recommend_for_user__book_name(
        self, 
        user_id: int, 
        books_df: pd.DataFrame, 
        n_recommendations: int = 10, 
        exclude_rated: bool = True, 
        ratings_df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Get recommendations for a user with book names instead of IDs.
        
        Args:
            user_id: ID of the user to get recommendations for
            books_df: DataFrame containing book information with 'id' and 'name' columns
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude books the user has already rated
            ratings_df: DataFrame containing user ratings (required if exclude_rated=True)
            
        Returns:
            List of tuples (book_name, score) ordered by recommendation score
        """
        if not self.is_fitted:
            logger.info(f"Model not fitted. loading saved models from {self.config.save_path}")
            self.load_model()
            
            if not self.is_fitted:
                logger.info(f"Failed to load a model {self.config.save_path}, aborting...")
                return

        if exclude_rated and ratings_df is None:
            raise ValueError("ratings_df must be provided when exclude_rated=True")
        
        # Get raw recommendations (list of (book_id, score) tuples)
        raw_recommendations = self.recommend_for_user(
            user_id=user_id,
            n_recommendations=n_recommendations,
            exclude_rated=exclude_rated,
            ratings_df=ratings_df
        )
        
        # Create mapping from book_id to book_name
        id_to_name = books_df.set_index('id')['name'].to_dict()
        
        # Convert to book names
        recommendations_with_names = []
        for book_id, _ in raw_recommendations:
            try:
                book_name = id_to_name[book_id]
                recommendations_with_names.append(book_name)
            except KeyError:
                logger.warning(f"Book ID {book_id} not found in books_df")
                continue
        
        return recommendations_with_names

    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }

    def save(self):
        """Save the trained model"""
        filepath: str = self.config.save_path
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "config": self.config,
            "user_features": self.user_features,
            "user_biases": self.user_biases,
            "item_features": self.item_features,
            "item_biases": self.item_biases,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "global_bias": self.global_bias,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self):
        """Load a trained model"""
        filepath: str = self.config.save_path

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        
        self.config = model_data["config"]
        self.user_features = model_data["user_features"]
        self.user_biases = model_data["user_biases"]
        self.item_features = model_data["item_features"]
        self.item_biases = model_data["item_biases"]
        self.user_to_idx = model_data["user_to_idx"]
        self.idx_to_user = model_data["idx_to_user"]
        self.item_to_idx = model_data["item_to_idx"]
        self.idx_to_item = model_data["idx_to_item"]
        self.global_bias = model_data["global_bias"]
        
        # Load training history if available
        self.train_losses = model_data.get("train_losses", [])
        self.val_losses = model_data.get("val_losses", [])
        
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


# Example usage with better hyperparameters
def example_usage():
    """Example of how to use the overfitting-resistant collaborative filtering"""
    
    # Create configuration optimized to prevent overfitting
    config = TrainingConfig(
        num_features=20,  # Reduced model complexity
        learning_rate=0.001,  # Lower learning rate
        regularization=0.2,  # Higher regularization
        max_iterations=500,  # Fewer iterations
        batch_size=2048,  # Larger batches
        early_stopping_patience=3,  # Less patience
        learning_rate_decay=0.9,  # Aggressive decay
        decay_every=25  # More frequent decay
    )
    
    # Initialize model
    cf_model = CollaborativeFilter(config)
    
    # Example of creating sample data for testing
    np.random.seed(42)
    n_users, n_items = 1000, 500
    n_ratings = 50000
    
    sample_data = pd.DataFrame({
        'user_id': np.random.randint(1, n_users + 1, n_ratings),
        'book_id': np.random.randint(1, n_items + 1, n_ratings),
        'rating': np.random.randint(1, 6, n_ratings)  # Ratings 1-5
    })
    
    # Remove duplicates
    sample_data = sample_data.drop_duplicates(subset=['user_id', 'book_id'])
    
    print(f"Sample data shape: {sample_data.shape}")
    print("Training model...")
    
    # Train the model
    cf_model.fit(sample_data)
    
    # Make predictions
    prediction = cf_model.predict(user_id=123, item_id=456)
    print(f"Prediction for user 123, item 456: {prediction:.4f}")
    
    # Get recommendations
    recommendations = cf_model.recommend_for_user(
        user_id=123, 
        n_recommendations=10,
        exclude_rated=True,
        ratings_df=sample_data
    )
    print(f"Top 5 recommendations for user 123: {recommendations[:5]}")
    
    # Plot training history
    history = cf_model.get_training_history()
    if history['train_losses']:
        print(f"Final train RMSE: {history['train_losses'][-1]:.4f}")
        print(f"Final val RMSE: {history['val_losses'][-1]:.4f}")
        print(f"Final gap: {history['val_losses'][-1] - history['train_losses'][-1]:.4f}")

if __name__ == "__main__":
    example_usage()

# import essentials
import pandas as pd
import numpy as np

# import preprocessing tools
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import pairwise_distances

# import garbage collection for memory effiency
import gc

# import utility functions
from utils.utils import get_period_index
from utils.progressline import progress_bar

from CollaborativeFilter import CollaborativeFilter, TrainingConfig
from ContentBased import ContentBasedConfig, ContentBasedFilter

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookRecommender():
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self):
        self.fit_collaborative()

        self.fit_contentbased()

    def fit_collaborative(self):
        config = TrainingConfig(max_iterations=1000)
        self.cf_model = CollaborativeFilter(config)

        # Load ratings data (fix the column name issue)
        ratings_df = pd.read_csv("dataset/ratings.csv")

        logger.info(ratings_df.head())

        # Train the model

        self.cf_model.fit(ratings_df)

        self.cf_model.save("models/cf.joblib")

    def fit_contentbased(self):
        config = ContentBasedConfig()
        self.cb_model = ContentBasedFilter(config)
        
        books_df = pd.read_csv("dataset/ratings.csv")

        self.cb_model.fit(books_df)

        self.cb_model.save("models/cb.joblib")

if __name__=="__main__":
    recommender = BookRecommender()
    recommender.fit()

    prediction = recommender.cf_model.predict(user_id=1, item_id=1316509)
    
    # Get recommendations
    recommendations = recommender.cf_model.recommend_for_user(user_id=123, n_recommendations=10)
    
    print("Collaborative filtering implementation is ready!")

    print(recommendations)

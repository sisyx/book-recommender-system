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

        # Train the model

        self.cf_model.fit(ratings_df)

        self.cf_model.save()

        del ratings_df

    def fit_contentbased(self):
        config = ContentBasedConfig()
        self.cb_model = ContentBasedFilter(config)
        
        books_df = pd.read_csv("dataset/books.csv")

        self.cb_model.fit(books_df)

        self.cb_model.save()
        
        del books_df

    def load_models(self):
        cf_config = TrainingConfig(max_iterations=1000)
        self.cf_model = CollaborativeFilter(cf_config)
        cb_config = ContentBasedConfig()
        self.cb_model = ContentBasedFilter(cb_config)
        self.cf_model.load_model()
        self.cb_model.load_model()

    def _run_console_api(self):
        if not self.cf_model.is_fitted or not self.cb_model.is_fitted:
            logger.info("Please train or load a model first")
            return

        books_df = pd.read_csv("dataset/books.csv")
        ratings_df = pd.read_csv("dataset/ratings.csv")
        
        while True:
            what_to_do = input("what do you want me to do? (provide recommendations: 1, show similar books to books you read: 2): ")
            if what_to_do == "2":
                what_book_name = input("please enter the name of the book: ")
                books_id = self.cb_model._get_book_id(what_book_name, books_df=books_df)
                similars = self.cb_model.get_similar_items(item_id=books_id, top_k=10)
                number = 1
                for id, score in similars:
                    book_name = self.cb_model._get_book_name(id, books_df=books_df)
                    print(f"{number}: {book_name}")
                    number += 1
            else:
                continue

if __name__=="__main__":
    recommender = BookRecommender()
    recommender.load_models()
    recommender._run_console_api()

    # prediction = recommender.cf_model.predict(user_id=1, item_id=1316509)
    
    # Get recommendations
    # recommendations = recommender.cf_model.recommend_for_user(user_id=123, n_recommendations=10)
    
    # print("Collaborative filtering implementation is ready!")

    # print(recommendations)

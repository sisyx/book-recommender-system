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
from dataclasses import dataclass

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookRecommender():
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self):
        # don't use collaborative filtering as we don't have any users in reallity
        # self.fit_collaborative()

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
        # don't use collaborative filtering as we don't have any users in reallity
        # cf_config = TrainingConfig(max_iterations=1000)
        # self.cf_model = CollaborativeFilter(cf_config)
        cb_config = ContentBasedConfig()
        self.cb_model = ContentBasedFilter(cb_config)
        # self.cf_model.load_model()
        self.cb_model.load_model()
    
    def _run_console_api(self):
        if not self.cb_model.is_fitted:
            logger.info("Please train or load a model first")
            return

        books_df = pd.read_csv("dataset/books.csv")

        # don't use collaborative filtering as we don't have any users in reallity
        # ratings_df = pd.read_csv("dataset/ratings.csv")
        num = 1
        fav_books = []
        print("> Please Enter your favorite books. (hit enter after each book, use (/q) when you wrote all books)")
        while True:
            fav_book = input(f"{num}: ")
            if fav_book == "/q":
                if len(fav_books) >= 0:
                    all_similars = []
                    for fav in fav_books:
                        books_id = self.cb_model._get_book_id(fav, books_df=books_df)
                        similars = self.cb_model.get_similar_items(item_id=books_id, top_k=10)
                        all_similars.extend(similars)

                    all_similars.sort(key=lambda x: x[1], reverse=True)

                    # take only top-k
                    all_similars = all_similars[0:self.cb_model.config.top_k_similar]

                    number = 1
                    print("You Can Read These Books Next: ")
                    for id, score in all_similars:
                        book_name = self.cb_model._get_book_name(id, books_df=books_df)
                        print(f"{number}: {book_name} | ({(score  * 100)}% Similar)")
                        number += 1
                fav_books = []
                num = 1
                print("> Please Enter your favorite books. (hit enter after each book, use (/q) when you wrote all books)")
            else:
                fav_books.append(fav_book)
                num += 1

if __name__=="__main__":
    recommender = BookRecommender()
    # recommender.fit()
    recommender.load_models()
    recommender._run_console_api()

    # prediction = recommender.cf_model.predict(user_id=1, item_id=1316509)
    
    # Get recommendations
    # recommendations = recommender.cf_model.recommend_for_user(user_id=123, n_recommendations=10)
    
    # print("Collaborative filtering implementation is ready!")

    # print(recommendations)

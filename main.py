# import essentials
import pandas as pd
import numpy as np

# import preprocessing tools
from sklearn.preprocessing import MinMaxScaler

# from CollaborativeFilter import CollaborativeFilter, TrainingConfig
from ContentBased import ContentBasedConfig, ContentBasedFilter
from dataclasses import dataclass
from pathlib import Path

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

        # Train the model

        self.cf_model.fit(self.ratings_df)

        self.cf_model.save()

    def fit_contentbased(self):
        config = ContentBasedConfig()
        self.cb_model = ContentBasedFilter(config)

        self.cb_model.fit(self.books_df)

        self.cb_model.save()

    def load_models(self):
        self.books_df = pd.read_csv(Path(__file__).resolve().parent / "dataset/books.csv")
        # don't use collaborative filtering as we don't have any users in reallity
        # self.ratings_df = pd.read_csv(Path(__file__).resolve().parent / "dataset/ratings.csv")
        # cf_config = TrainingConfig(max_iterations=1000)
        # self.cf_model = CollaborativeFilter(cf_config)
        cb_config = ContentBasedConfig()
        self.cb_model = ContentBasedFilter(cb_config)
        # self.cf_model.load_model()
        self.cb_model.load_model()

    def _similar_books__name(self, book_titles: list[str] = []):
        all_similars = []
        for fav in book_titles:
            books_id = self.cb_model._get_book_id(fav, books_df=self.books_df)
            similars = self.cb_model.get_similar_items(item_id=books_id, top_k=10)
            all_similars.extend(similars)

        all_similars.sort(key=lambda x: x[1], reverse=True)

        # take only top-k
        all_similars = all_similars[0:self.cb_model.config.top_k_similar]

        result = []
        for id, score in all_similars:
            book_name = self.cb_model._get_book_name(id, books_df=self.books_df)
            book_isbn = self.cb_model._get_book_isbn(id, books_df=self.books_df)
            result.append({"name": book_name, "isbn": book_isbn})
            
        return result
    
    def _run_console_api(self):
        if not self.cb_model.is_fitted:
            logger.info("Please train or load a model first")
            return

        # don't use collaborative filtering as we don't have any users in reallity
        num = 1
        fav_books = []
        print("> Please Enter your favorite books. (hit enter after each book, use (/q) when you wrote all books)")
        while True:
            fav_book = input(f"{num}: ")
            if fav_book == "/q":
                if len(fav_books) >= 0:
                    all_similars = []
                    for fav in fav_books:
                        books_id = self.cb_model._get_book_id(fav, books_df=self.books_df)
                        similars = self.cb_model.get_similar_items(item_id=books_id, top_k=10)
                        all_similars.extend(similars)

                    all_similars.sort(key=lambda x: x[1], reverse=True)

                    # take only top-k
                    all_similars = all_similars[0:self.cb_model.config.top_k_similar]

                    number = 1
                    print("You Can Read These Books Next: ")
                    for id, score in all_similars:
                        book_name = self.cb_model._get_book_name(id, books_df=self.books_df)
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
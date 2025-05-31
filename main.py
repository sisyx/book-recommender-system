# import essentials
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

# import preprocessing tools
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import pairwise_distances

# import models
from models.book import Book
from models.rating import Rating

# for enviroment variables
from dotenv import load_dotenv
import os

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
        # initialize enviromental variables
        load_dotenv()
        self.DATABASE_URL = os.environ["DATABASE_URL"]
        self.is_db_connected = False

        self.connect_db()

    def connect_db(self):
        self.engine = sa.create_engine(self.DATABASE_URL)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.is_db_connected = True

    def fit(self):
        if not self.is_db_connected:
            logger.info("Data Base is not Connected!")
            return 
        
        self.fit_collaborative()

        self.fit_contentbased()

    def fit_collaborative(self):
        config = TrainingConfig(max_iterations=1000)
        self.cf_model = CollaborativeFilter(config)

        # Load ratings data (fix the column name issue)
        ratings_df = pd.read_sql(self.session.query(Rating).statement, self.session.bind)


        # Train the model
        self.cf_model.fit(ratings_df[0:10000])

        self.cf_model.save("models/cf.joblib")

    def fit_contentbased(self):
        config = ContentBasedConfig()
        self.cb_model = ContentBasedFilter(config)
        
        books_df = pd.read_sql(self.session.query(Book).statement, self.session.bind)

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
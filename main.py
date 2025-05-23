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

class BookRecommender():
    def __init__(self, collaborative_weight=0.5, num_features=4, learning_rate=0.001, max_iterations=10000):
        self.num_features = num_features
        self.collaborative_weight = collaborative_weight
        self.collaborative_learning_rate = learning_rate
        self.iterations = max_iterations

        self.scaler = MinMaxScaler()

        # regularization term to protect model from overfitting
        self.regularization = 0.01

        # initialize enviromental variables
        load_dotenv()
        self.DATABASE_URL = os.environ["DATABASE_URL"]

        # other stuff
        self.collaborative_cost_hist = []

        self.load_data()
    

    def __initilize_parameters(self, users, books):
        """
            -- Colaborative Filtering --
            this function is here just to create two matrix
        """
        self.user_features = {}
        self.user_biases = {}
        for u in users:
            self.user_features[u] = np.random.normal(0, 0.1, self.num_features)
            self.user_biases[u] = 0

        self.book_features = {}
        self.book_biases = {}
        for b in books:
            self.book_features[b] = np.random.normal(0, 0.1, self.num_features)
            self.book_biases[b] = 0

        print("> User and Book features initialized successfully")
        

    def load_data(self):
        print("> Starting...")
        print("> Loading Data...")
        print("> Connecting to Database Server...")
        self.engine = sa.create_engine(self.DATABASE_URL)
        Session = sessionmaker(bind=self.engine)
        session = Session()
        print("> Successfullt Connected to Database Server!")

        try:
            # load & prepare content-based data
            print("> Loading books dataset...")
            books_df = pd.read_sql(session.query(Book).statement, session.bind)
            print("> Successfully Loaded books dataset!")
            self.books_df = books_df[["id", "publish_year", "rating", "language"]] #[0:1000] # Just for testing, limit the train data to 1000 items
            print("> Deleting books_df...")
            # remove books_df as we don't need it anymore (we have self.books_df)
            del books_df
            print("> Successfully Deleted books_df!")
            
            self.books_df["literature_period"] = self.books_df["publish_year"].apply(get_period_index)
            self.books_df["rating"] = self.scaler.fit_transform(self.books_df[["rating"]])

            # Create one-hot encoded columns
            encoded_df = pd.get_dummies(self.books_df['language'], prefix='lang')
            encoded_year_df = pd.get_dummies(self.books_df["literature_period"], prefix="period")
            # Add these columns to your original dataframe
            self.books_df = pd.concat([self.books_df.drop(columns={"language", "publish_year"}), encoded_df, encoded_year_df], axis=1)
            
            print(self.books_df.columns)
            print(self.books_df["rating"])

            # load & prepare collaborative-based-filtering data
            ratings_df = pd.read_sql(session.query(Rating).statement, session.bind)
            books = ratings_df["book_id"].unique()
            users = ratings_df["id"].unique()

            self.__initilize_parameters(users, books)

            print("> Creating book user map...")
            self.__create_book_user_rating_map(ratings_df)
            print("> Successfully Created book user map!")

            # remove ratings df completly with garbage collection
            print("> Deleting ratings_df...")
            del ratings_df
            gc.collect()
            print("> Successfully Deleted ratings_df!")

        finally:
            session.close()


    def __create_book_user_rating_map(self, ratings_df):
        """
            -- Colaborative Filtering --
        """
        self.book_map = {}
        for idx, rating_row in ratings_df.iterrows():
            book_id = rating_row["book_id"]
            user_id = rating_row["id"]
            rating = rating_row["rating"]
            # If book_id not in book_map, initialize it with an empty dict
            if book_id not in self.book_map:
                self.book_map[book_id] = {}
            # Add {user_id: rating} to the book's dictionary
            self.book_map[book_id][user_id] = rating


    def collaborative_cost(self):
        """
            -- Colaborative Filtering --
            -- we will use Mean Squared Error Cost Function
        """
        cost = 0
        num_ratings = 0
        for book_id, ratings in self.book_map.items():
            for user_id, rating_zip in zip(ratings.keys(), ratings.items()):
                rating = rating_zip[1] # rating_zip[0] is user_id and rating_zip [1] is actual rating
                # compute current cost and it to the total cost
                prediction = np.matmul(self.user_features[user_id], self.book_features[book_id]) + self.user_biases[user_id] + self.book_biases[book_id]
                cost += (rating - prediction) ** 2

                # count number of ratings
                num_ratings += 1
        cost = cost/num_ratings
        return cost


    def train_colaborative(self):
        print("iteration\tcost")
        
        # Convert dictionaries to arrays for vectorized operations
        # Assuming you have these mappings already or can create them:
        user_indices = {user_id: idx for idx, user_id in enumerate(self.user_features.keys())}
        book_indices = {book_id: idx for idx, book_id in enumerate(self.book_features.keys())}
        
        # Create arrays from dictionaries
        user_features_array = np.array([self.user_features[user_id] for user_id in user_indices])
        book_features_array = np.array([self.book_features[book_id] for book_id in book_indices])
        user_biases_array = np.array([self.user_biases[user_id] for user_id in user_indices])
        book_biases_array = np.array([self.book_biases[book_id] for book_id in book_indices])
        
        # Prepare ratings data as numpy arrays
        rating_data = []
        for book_id, ratings in self.book_map.items():
            book_idx = book_indices[book_id]
            for user_id, rating in ratings.items():
                user_idx = user_indices[user_id]
                rating_data.append((user_idx, book_idx, rating))
        
        ratings_array = np.array(rating_data)
        user_idx_array = ratings_array[:, 0].astype(int)
        book_idx_array = ratings_array[:, 1].astype(int)
        rating_values = ratings_array[:, 2]
        
        for i in range(self.iterations):
            # Shuffle indices
            shuffle_indices = np.random.permutation(len(ratings_array))
            user_idx_shuffled = user_idx_array[shuffle_indices]
            book_idx_shuffled = book_idx_array[shuffle_indices]
            rating_values_shuffled = rating_values[shuffle_indices]
            
            # Process in mini-batches
            batch_size = min(1000, len(ratings_array))
            for start_idx in range(0, len(ratings_array), batch_size):
                end_idx = min(start_idx + batch_size, len(ratings_array))
                
                # Get batch data
                batch_user_idx = user_idx_shuffled[start_idx:end_idx]
                batch_book_idx = book_idx_shuffled[start_idx:end_idx]
                batch_ratings = rating_values_shuffled[start_idx:end_idx]
                
                # Get user and book features for this batch
                batch_user_features = user_features_array[batch_user_idx]
                batch_book_features = book_features_array[batch_book_idx]
                batch_user_biases = user_biases_array[batch_user_idx]
                batch_book_biases = book_biases_array[batch_book_idx]
                
                # Calculate predictions (using NumPy's batch matrix multiplication)
                predictions = np.sum(batch_user_features * batch_book_features, axis=1) + \
                            batch_user_biases + batch_book_biases
                
                # Calculate errors
                errors = batch_ratings - predictions
                
                # Reshape for broadcasting
                errors_reshaped = errors.reshape(-1, 1)
                
                # Calculate gradients
                user_features_grad = -2 * errors_reshaped * batch_book_features + \
                                    2 * self.regularization * batch_user_features
                book_features_grad = -2 * errors_reshaped * batch_user_features + \
                                    2 * self.regularization * batch_book_features
                user_biases_grad = -2 * errors + 2 * self.regularization * batch_user_biases
                book_biases_grad = -2 * errors + 2 * self.regularization * batch_book_biases
                
                # Update parameters
                # Note: We need to be careful with updates to avoid race conditions
                for idx in range(len(batch_user_idx)):
                    user_idx = batch_user_idx[idx]
                    book_idx = batch_book_idx[idx]
                    
                    user_features_array[user_idx] -= self.collaborative_learning_rate * user_features_grad[idx]
                    book_features_array[book_idx] -= self.collaborative_learning_rate * book_features_grad[idx]
                    user_biases_array[user_idx] -= self.collaborative_learning_rate * user_biases_grad[idx]
                    book_biases_array[book_idx] -= self.collaborative_learning_rate * book_biases_grad[idx]
            
            # Update the original dictionaries periodically (every 10 iterations to reduce overhead)
            if i % 10 == 0:
                # Update dictionaries from arrays
                for user_id, idx in user_indices.items():
                    self.user_features[user_id] = user_features_array[idx]
                    self.user_biases[user_id] = user_biases_array[idx]
                    
                for book_id, idx in book_indices.items():
                    self.book_features[book_id] = book_features_array[idx]
                    self.book_biases[book_id] = book_biases_array[idx]
                
                # Calculate and record cost
                cost = self.collaborative_cost()
                print(f"{i}\t{cost}")
                self.collaborative_cost_hist.append((i, cost))
        
        # Final update of dictionaries
        for user_id, idx in user_indices.items():
            self.user_features[user_id] = user_features_array[idx]
            self.user_biases[user_id] = user_biases_array[idx]
            
        for book_id, idx in book_indices.items():
            self.book_features[book_id] = book_features_array[idx]
            self.book_biases[book_id] = book_biases_array[idx]


    def fit_collaborative(self):
        self.train_colaborative()


    def fit_contentbased(self, top_k=10, threshold=0.5, metric="hamming", batch_size=1000):
        """
            Compute top-k similarities for each item, only keeping similarities above threshold.
                
                Parameters:
                - k: number of top similar items to keep per item
                - threshold: minimum similarity score to consider
                - metric: distance metric (hamming for one-hot encodings)
                - batch_size: process items in batches to manage memory
        """
        k = top_k
        X = self.books_df.drop(columns={"id"}).to_numpy()
        X = np.vstack(X)
        print("> sample distances: ")
        X_s_distances = pairwise_distances(X[0:10], X[0:10], metric=metric)
        print(X_s_distances)
        n_items = X.shape[0]
        similarities = {}
        print(f"Processing {n_items} items in batches of {batch_size}...")

        for i in range(0, n_items, batch_size):
            end_i = min(i + batch_size, n_items)
            batch = X[i:end_i]

            print(f"processing items {i} to {end_i - 1}")
            distances = pairwise_distances(batch, X, metric=metric)
            similarity_scores = 1 - distances

            for batch_idx, global_idx in enumerate(range(i, end_i)):
                item_similarities = similarity_scores[batch_idx]

                above_threshold = item_similarities >= threshold
                candidate_items = np.where(above_threshold)[0]
                candidate_scores = item_similarities[above_threshold]


                if len(candidate_items) > 0:
                    if len(candidate_items) > k:
                        top_k_indices = np.argsort(candidate_scores)[-k:][::-1]
                        top_items = candidate_items[top_k_indices]
                        top_scores = candidate_scores[top_k_indices]

                    else:
                        sort_indices = np.argsort(candidate_scores)[::-1]
                        top_items = candidate_items[sort_indices]
                        top_scores = candidate_scores[sort_indices]
                    
                    similarities[global_idx] = list(zip(top_items.tolist(), top_scores.tolist()))
                
                else: similarities[global_idx] = []
        
        self.similarities = similarities

    
    def recommend_similar(self, book_idx):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        similars = self.similarities[book_idx]
        print(f"you may want to read these books: {similars}")
        books_df = pd.read_sql(session.query(Book).statement, session.bind)
        for key in similars:
            # print(books_df.iloc[key[0], ["name"] ])
            item = books_df[key[0]:key[0] + 1]
            print(item["name"])


if __name__=="__main__":
    recommender = BookRecommender(max_iterations=100)
    # recommender.fit_collaborative
    recommender.fit_contentbased(batch_size=20)
    recommender.recommend_similar(1)
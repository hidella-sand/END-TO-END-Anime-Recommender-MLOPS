import os
import pandas as pd
import numpy as np
import joblib
import sys

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

        self.rating_df = None
        self.anime_df = None
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None

        self.user2user_encoded = {}
        self.user2user_decoded = {}
        self.anime2anime_encoded = {}
        self.anime2anime_decoded = {}

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("✅ DataProcessor initialized successfully.")

    def load_data(self, usecols):
        """Load ratings CSV with only necessary columns."""
        try:
            self.rating_df = pd.read_csv(self.input_file, low_memory=True, usecols=usecols)
            logger.info("✅ Ratings data loaded successfully.")
        except Exception as e:
            raise CustomException("❌ Failed to load ratings data", sys) from e

    def filter_users(self, min_rating=400):
        """Filter out users with fewer than `min_rating` ratings."""
        try:
            n_ratings = self.rating_df["user_id"].value_counts()
            self.rating_df = self.rating_df[self.rating_df["user_id"].isin(n_ratings[n_ratings >= min_rating].index)].copy()
            logger.info(f"✅ Users filtered: Kept users with ≥{min_rating} ratings.")
        except Exception as e:
            raise CustomException("❌ Failed to filter users", sys) from e

    def scale_ratings(self):
        """Scale ratings to a [0, 1] range using Min-Max normalization."""
        try:
            min_rating = self.rating_df["rating"].min()
            max_rating = self.rating_df["rating"].max()
            self.rating_df["rating"] = self.rating_df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).astype(np.float64)
            logger.info("✅ Ratings scaled using Min-Max normalization.")
        except Exception as e:
            raise CustomException("❌ Failed to scale ratings", sys) from e

    def encode_data(self):
        """Encode user and anime IDs into integer labels."""
        try:
            # Encode users
            user_ids = self.rating_df["user_id"].unique().tolist()
            self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            self.user2user_decoded = {i: x for i, x in enumerate(user_ids)}
            self.rating_df["user"] = self.rating_df["user_id"].map(self.user2user_encoded)

            # Encode anime
            anime_ids = self.rating_df["anime_id"].unique().tolist()
            self.anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
            self.anime2anime_decoded = {i: x for i, x in enumerate(anime_ids)}
            self.rating_df["anime"] = self.rating_df["anime_id"].map(self.anime2anime_encoded)

            logger.info("✅ Successfully encoded user and anime IDs.")
        except Exception as e:
            raise CustomException("❌ Failed to encode user or anime IDs", sys) from e

    def split_data(self, test_size=1000, random_state=43):
        """Split data into train/test sets and reshape for model input."""
        try:
            self.rating_df = self.rating_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            X = self.rating_df[["user", "anime"]].values
            y = self.rating_df["rating"]

            train_indices = self.rating_df.shape[0] - test_size
            X_train, X_test, y_train, y_test = (
                X[:train_indices],
                X[train_indices:],
                y[:train_indices],
                y[train_indices:]
            )

            self.X_train_array = [X_train[:, 0], X_train[:, 1]]
            self.X_test_array = [X_test[:, 0], X_test[:, 1]]
            self.y_train = y_train
            self.y_test = y_test

            logger.info("✅ Data split into training and testing sets.")
        except Exception as e:
            raise CustomException("❌ Failed to split data", sys) from e

    def save_artifacts(self):
        """Save processed data and artifacts to disk."""
        try:
            artifacts = {
                "user2user_encoded": self.user2user_encoded,
                "user2user_decoded": self.user2user_decoded,
                "anim2anime_encoded": self.anime2anime_encoded,
                "anim2anime_decoded": self.anime2anime_decoded,
            }

            for name, data in artifacts.items():
                joblib.dump(data, os.path.join(self.output_dir, f"{name}.pkl"))
                logger.info(f"✅ Saved {name}.pkl to {self.output_dir}")

            joblib.dump(self.X_train_array, X_TRAIN_ARRAY)
            joblib.dump(self.X_test_array, X_TEST_ARRAY)
            joblib.dump(self.y_train, Y_TRAIN)
            joblib.dump(self.y_test, Y_TEST)
            self.rating_df.to_csv(RATING_DF, index=False)

            logger.info("✅ All training/testing arrays and ratings DataFrame saved successfully.")
        except Exception as e:
            raise CustomException("❌ Failed to save processed artifacts", sys) from e

    def process_anime_data(self):
        """Load, clean, and save anime metadata and synopsis."""
        try:
            df = pd.read_csv(ANIME_CSV)
            cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
            synopsis_df = pd.read_csv(ANIMESYNOPSIS_CSV, usecols=cols)
            df = df.replace("Unknown", np.nan)

            def getAnimeName(anime_id):
                try:
                    name = df[df.anime_id == anime_id].eng_version.values[0]
                    if pd.isna(name):
                        name = df[df.anime_id == anime_id].Name.values[0]
                    return name
                except:
                    logger.warning(f"⚠️ Could not retrieve name for anime_id={anime_id}")
                    return None

            df["anime_id"] = df["MAL_ID"]
            df["eng_version"] = df["English name"]
            df["eng_version"] = df["anime_id"].apply(lambda x: getAnimeName(x))

            df.sort_values(by=["Score"], ascending=False, kind="quicksort", na_position="last", inplace=True)
            df = df[["anime_id", "eng_version", "Score", "Genres", "Episodes", "Type", "Premiered", "Members"]]

            df.to_csv(DF, index=False)
            synopsis_df.to_csv(SYNOPSIS_DF, index=False)

            logger.info("✅ Anime metadata and synopsis files saved successfully.")
        except Exception as e:
            raise CustomException("❌ Failed to process anime and synopsis data", sys) from e

    def run(self):
        """Run the full data processing pipeline."""
        try:
            self.load_data(usecols=["user_id", "anime_id", "rating"])
            self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            self.save_artifacts()
            self.process_anime_data()

            logger.info("🎉 Data processing pipeline completed successfully.")
        except CustomException as e:
            logger.error(str(e))


if __name__ == "__main__":
    data_processor = DataProcessor(ANIMELIST_CSV, PROCESSED_DIR)
    data_processor.run()

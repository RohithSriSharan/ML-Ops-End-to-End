import os 
import pandas as pd
import yaml
from logger import get_logger
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

logger = get_logger(__name__,"data_validation.log")

# ----------------- PATHS -----------------
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
FEATURE_DIR = "features/"

class FeatureEngineering:
    def __init__(self, params:dict):
        self.max_features = params["features"]["max_features"]
        self.ngram_range = tuple(params["features"]["ngram_range"])
        self.max_df = params["features"]["max_df"]
        self.min_df = params["features"]["min_df"]
        self.vectorizer = None

    def run(self,train_path:str, test_path:str, output_dir:str):
        logger.info("Loading train and test datasets...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df['clean_review'].astype(str)
        y_train = train_df['label']

        X_test = test_df['clean_review'].astype(str)
        y_test = test_df['label']

        logger.info(
                    "Initializing TF-IDF (max_features=%s, ngram_range=%s, max_df=%.2f, min_df=%d)",
                    self.max_features, str(self.ngram_range), self.max_df, self.min_df
                )

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            max_df = self.max_df,
            min_df=self.min_df,
            stop_words='english'

        )

        logger.info("Fitting TF-IDF on train set...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        logger.info("Transforming test set...")
        X_test_vec = self.vectorizer.transform(X_test)

        os.makedirs(output_dir, exist_ok=True)

        # Save features and labels

        with open(os.path.join(output_dir,"X_train.pkl"),'wb') as f:
            pickle.dump(X_train_vec,f)
        with open(os.path.join(output_dir, "y_train.pkl"), "wb") as f:
            pickle.dump(y_train, f)
        with open(os.path.join(output_dir, "X_test.pkl"), "wb") as f:
            pickle.dump(X_test_vec, f)
        with open(os.path.join(output_dir, "y_test.pkl"), "wb") as f:
            pickle.dump(y_test, f)


        # Save vectorizer 

        with open(os.path.join(output_dir, "tfidf.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)

        logger.info("Feature engineering complete. Saved TF-IDF vectors to %s", output_dir)

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    fe = FeatureEngineering(params=params)
    fe.run(TRAIN_PATH, TEST_PATH, FEATURE_DIR)
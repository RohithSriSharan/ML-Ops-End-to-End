import os
import pickle
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from logger import get_logger

logger = get_logger(__name__, "model_building.log")

FEATURE_DIR = "features/"
MODEL_DIR = "models/"

# Point MLflow to remote server
mlflow.set_tracking_uri("http://ec2-13-126-189-174.ap-south-1.compute.amazonaws.com:5000")
mlflow.set_experiment("IMDB-Sentiment")

class ModelBuilder:
    def __init__(self, params: dict):
        self.max_iter = params["train"]["max_iter"]
        self.C = params["train"]["C"]
        self.model = None

    def load_data(self, feature_dir: str):
        logger.info("Loading training data...")
        with open(os.path.join(feature_dir, "X_train.pkl"), "rb") as f:
            X_train = pickle.load(f)
        with open(os.path.join(feature_dir, "y_train.pkl"), "rb") as f:
            y_train = pickle.load(f)
        return X_train, y_train

    def train(self, X_train, y_train, X_test, y_test):
        logger.info("Training Logistic Regression...")
        with mlflow.start_run():
            # Log params
            mlflow.log_param("max_iter", self.max_iter)
            mlflow.log_param("C", self.C)

            # Train
            self.model = LogisticRegression(
                max_iter=self.max_iter,
                C=self.C,
                solver="liblinear"
            )
            self.model.fit(X_train, y_train)
            logger.info("Training complete.")

            # Evaluate immediately (optional — or leave to evaluation.py)
            preds = self.model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Log model to MLflow (goes to S3)
            mlflow.sklearn.log_model(self.model, "model")

            logger.info("Run logged to MLflow with accuracy=%.4f, f1=%.4f", acc, f1)

    def save_model(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Model saved to %s", model_path)
        return model_path


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    builder = ModelBuilder(params=params)

    # Load train + test
    X_train, y_train = builder.load_data(FEATURE_DIR)
    with open(os.path.join(FEATURE_DIR, "X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(FEATURE_DIR, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)

    # Train + log to MLflow
    builder.train(X_train, y_train, X_test, y_test)

    # Still save locally for DVC
    builder.save_model(MODEL_DIR)

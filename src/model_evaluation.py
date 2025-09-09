import os
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
from logger import get_logger

logger = get_logger(__name__, "model_evaluation.log")

FEATURE_DIR = "features/"
MODEL_DIR = "models/"
METRICS_PATH = "metrics.json"
REPORT_PATH = "classification_report.txt"

class ModelEvaluator:
    def __init__(self):
        self.model = None

    def load_model(self, model_dir: str):
        model_path = os.path.join(model_dir, "model.pkl")
        logger.info("Loading model from %s", model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def load_test_data(self, feature_dir: str):
        logger.info("Loading test data...")
        with open(os.path.join(feature_dir, "X_test.pkl"), "rb") as f:
            X_test = pickle.load(f)
        with open(os.path.join(feature_dir, "y_test.pkl"), "rb") as f:
            y_test = pickle.load(f)
        return X_test, y_test

    def evaluate(self, X_test, y_test, metrics_path: str, report_path: str):
        logger.info("Evaluating model...")
        preds = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
        }

        logger.info("Accuracy: %.4f, F1-score: %.4f", metrics["accuracy"], metrics["f1_score"])

        # Save metrics.json for DVC
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save full classification report
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, preds))

        logger.info("Evaluation complete. Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.load_model(MODEL_DIR)
    X_test, y_test = evaluator.load_test_data(FEATURE_DIR)
    evaluator.evaluate(X_test, y_test, METRICS_PATH, REPORT_PATH)

import os
import pandas as pd
import numpy as np
import contractions
import re

from logger import get_logger

logger = get_logger(__name__,"data_preprocessing.log")

INPUT_PATH = "data/interim/IMDB_validated.csv"
OUTPUT_PATH = "data/processed/IMDB_clean.csv"


def clean_text(text:str) -> str:
    text = contractions.fix(text)                     # expand contractions
    text = text.lower()                               # lowercase
    text = re.sub(r"<.*?>", " ", text)                # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)             # keep only letters + spaces
    text = re.sub(r"\s+", " ", text).strip()          # collapse spaces
    return text

class DataPreprocessing:
    def __init__(self, input_path:str, output_path:str):
        self.input_path = input_path
        self.output_path = output_path

    
    
    
    def run(self):
        logger.info("Reading dataset from %s", self.input_path)
        df = pd.read_csv(self.input_path)

        logger.info("Handling nulls and duplicates")
        df = df.dropna(subset=['review','sentiment'])
        logger.info("Duplicated Rows before dropping: %s", df.duplicated().sum())
        df = df.drop_duplicates()
        logger.info("Duplicated Rows after dropping: %s", df.duplicated().sum())


        logger.info("Cleaning Text")
        df['clean_review'] = df['review'].astype(str).apply(clean_text)

        logger.info("Encoding sentiment labels...")
        df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

        os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
        df.to_csv(self.output_path)
        logger.info("Preprocessing complete. Data saved to %s", self.output_path)



# ----------------- MAIN -----------------
if __name__ == "__main__":
    processor = DataPreprocessing(
        input_path="data/interim/IMDB_validated.csv",
        output_path="data/processed/IMDB_clean.csv"
    )
    processor.run()
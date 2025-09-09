import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import get_logger

logger = get_logger(__name__, "data_split.log")

INPUT_PATH="data/processed/IMDB_clean.csv"
TRAIN_PATH="data/processed/train.csv"
TEST_PATH="data/processed/test.csv"


class DataSplit:
    def __init__(self,input_path:str, train_path:str, test_path:str , params:dict):
        self.input_path = input_path
        self.train_path = train_path
        self.test_path = test_path
        self.test_size = params['split']['test_size']
        self.random_state = params['split']['random_state']

    def run(self):

        logger.info("Reading Cleaned data from %s", self.input_path)
        df = pd.read_csv(self.input_path)

        logger.info("Splitting dataset: %.0f%% train, %.0f%% test", 
                    (1 - self.test_size) * 100, self.test_size * 100)

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df['label']
        )

        os.makedirs(os.path.dirname(self.train_path), exist_ok=True)
        train_df.to_csv(self.train_path , index =False)
        test_df.to_csv(self.test_path, index = False)

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    splitter = DataSplit(
        input_path=INPUT_PATH,
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        params=params
    )

    splitter.run()
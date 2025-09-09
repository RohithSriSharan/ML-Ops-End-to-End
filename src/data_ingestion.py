import os
import pandas as pd
from logger import get_logger


logger = get_logger(__name__,"data_ingestion.log")  # use module-level logger




class DataIngestion:
    def __init__(self, input_path:str, output_path:str):
        self.input_path = input_path
        self.output_path = output_path
        
    def run(self):
        logger.info("Reading dataset from %s", self.input_path)
        df = pd.read_csv(self.input_path)
        logger.info("Dataset loaded with shape %s", df.shape)
    
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info("Date saved to %s", self.output_path)

if __name__=="__main__":
    ingestion = DataIngestion(
        input_path="data/external/IMDB_Dataset.csv",
        output_path="data/raw/IMDB_Dataset.csv"
    )
    ingestion.run()
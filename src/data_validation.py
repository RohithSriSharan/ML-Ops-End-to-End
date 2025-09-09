import os 
import pandas as pd

from logger import get_logger

logger = get_logger(__name__,"data_validation.log")


class DataValidation:
    def __init__(self, input_path:str, output_path:str, expected_cols: list):

        self.input_path = input_path
        self.output_path = output_path
        self.expected_cols = expected_cols
        
    def run(self):
        logger.info("Reading dataset from %s", self.input_path)
        df = pd.read_csv(self.input_path)

        # 1. Schema check
        missing = [ col for col in self.expected_cols if col not in df.columns ]
        if missing:
            logger.error("Missing columns: %s" ,missing)
            raise ValueError("Schema Validation failed")
        
        # 2. Row count check
        if len(df) == 0:
            logger.error("Dataset is empty!")
            raise ValueError("No rows found in dataset")
        
        # 3. Null values check
        null_counts = df.isna().sum()
        if null_counts.any():
            logger.warning("Null values detected:\n%s", null_counts)

        # 4. Duplicated rows
        dup_count = df.duplicated(subset=["review"]).sum()
        if dup_count > 0:
            logger.warning("Found %d duplicated reviews.", dup_count)


        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv( self.output_path, index=False)
        logger.info("Validataion passed. Data saved to %s", self.output_path)


if __name__ == "__main__":

    validator = DataValidation(
        input_path="data/raw/IMDB Dataset.csv",
        output_path="data/interim/IMDB_validated.csv",
        expected_cols=["review", "sentiment"]
    )
    validator.run()
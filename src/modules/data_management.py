import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src import constants
from src.exception import CustomException
from src.classes import DataConfig

from src.logger import logging


class DataManagement:
    def __init__(self):
        self.ingestion_config = DataConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df = pd.read_csv(constants.PATH_TO_DATASET)
            logging.info("Read the dataset as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=constants.TEST_SIZE, random_state=constants.SEED)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

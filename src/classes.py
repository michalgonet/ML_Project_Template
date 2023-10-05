import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src import constants
from src.exception import CustomException

from src.logger import logging
from src.utils import save_object

@dataclass
class DataConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
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


@dataclass()
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_features = constants.NUMERICAL_FEATURES
            categorical_features = constants.CATEGORICAL_FEATURES
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipline", numerical_pipeline, numerical_features),
                    ("cat_pipline", categorical_pipeline, categorical_features)
                ]
            )
            return preprocessor
        except CustomException as e:
            raise (e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_obj()
            target_column_name = constants.TARGET_COLUMN
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on the training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]
            logging.info("Saved preprocessing objects")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)

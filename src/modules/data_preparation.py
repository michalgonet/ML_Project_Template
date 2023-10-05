import sys

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src import constants
from src.classes import DataTransformationConfig
from src.utils import save_object


def get_data_transformer_obj():
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


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = get_data_transformer_obj()
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

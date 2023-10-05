import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


@dataclass()
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


@dataclass()
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts', 'model.pkl')

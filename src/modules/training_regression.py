import sys

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src import constants
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.classes import ModelTrainerConfig


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': constants.DECISION_TREE_CRITERION,
                    'splitter': constants.DECISION_TREE_SPLITTER,
                    'max_features': constants.DECISION_TREE_MAX_FEATURES,
                },
                "Random Forest": {
                    'criterion': constants.RANDOM_FOREST_CRITERION,
                    'max_features': constants.RANDOM_FOREST_MAX_FEATURES,
                    'n_estimators': constants.RANDOM_FOREST_N_ESTIMATORS
                },
                "Gradient Boosting": {
                    'loss': constants.GRADIENT_BOOSTING_LOSS,
                    'learning_rate': constants.GRADIENT_BOOSTING_LR,
                    'subsample': constants.GRADIENT_BOOSTING_SUBSAMPLE,
                    'criterion': constants.GRADIENT_BOOSTING_CRITERION,
                    'max_features': constants.GRADIENT_BOOSTING_MAX_FEATURES,
                    'n_estimators': constants.GRADIENT_BOOSTING_N_ESTIMATORS
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': constants.K_NEIGHBOUR_N_NEIGHBOUR,
                    'weights': constants.K_NEIGHBOUR_WEIGHTS,
                    'algorithm': constants.K_NEIGHBOUR_ALGORITHM
                },
                "XGBRegressor": {
                    'learning_rate': constants.XGB_LR,
                    'n_estimators': constants.XGB_N_ESTIMATORS
                },
                "CatBoosting Regressor": {
                    'depth': constants.CATBOOST_DEPTH,
                    'learning_rate': constants.CATBOOST_LR,
                    'iterations': constants.CATBOOST_ITERS
                },
                "AdaBoost Regressor": {
                    'learning_rate': constants.ADABOOST_LR,
                    'loss': constants.ADABOOST_LOSS,
                    'n_estimators': constants.ADABOOST_N_ESTIMATORS
                }

            }
            model_report = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < constants.BEST_MODEL_THRESHOLD:
                raise CustomException("No best model found", sys)

            logging.info("Best found model on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square, best_model
        except Exception as e:
            raise CustomException(e, sys)

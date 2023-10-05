import os
import sys
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src import constants


def save_object(file_path: str, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        for k, v in models.items():
            model = v

            para = params[k]
            if constants.HYPER_TUNING:
                gs = GridSearchCV(model, para,
                                  cv=constants.GRID_SEARCH_CV,
                                  n_jobs=constants.GRID_SEARCH_N_JOBS,
                                  verbose=constants.GRID_SEARCH_VERBOSE,
                                  refit=constants.GRID_SEARCH_REFIT)
                gs.fit(x_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[k] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

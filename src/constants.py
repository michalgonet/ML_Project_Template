PATH_TO_DATASET: str = 'Data/students_performance_dataset.csv'
TEST_SIZE: float = 0.2
SEED: int = 42
NUMERICAL_FEATURES: list[str] = ["writing_score", "reading_score"]
CATEGORICAL_FEATURES: list[str] = ["gender",
                                   "race_ethnicity",
                                   "parental_level_of_education",
                                   "lunch",
                                   "test_preparation_course"]
TARGET_COLUMN: str = "math_score"
BEST_MODEL_THRESHOLD: float = 0.6

HYPER_TUNING: bool = False

GRID_SEARCH_CV: int = 3
GRID_SEARCH_N_JOBS: int = 3
GRID_SEARCH_VERBOSE: int = 0
GRID_SEARCH_REFIT: bool = False

DECISION_TREE_CRITERION: list[str] = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
DECISION_TREE_SPLITTER: list[str] = ['best', 'random']
DECISION_TREE_MAX_FEATURES: list[str] = ['sqrt', 'log2']

RANDOM_FOREST_CRITERION: list[str] = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
RANDOM_FOREST_MAX_FEATURES: list[str] = ['sqrt', 'log2', None]
RANDOM_FOREST_N_ESTIMATORS: list[int] = [8, 16, 32, 64, 128, 256]

GRADIENT_BOOSTING_LOSS: list[str] = ['squared_error', 'huber', 'absolute_error', 'quantile']
GRADIENT_BOOSTING_LR: list[float] = [.1, .01, .05, .001]
GRADIENT_BOOSTING_SUBSAMPLE: list[float] = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
GRADIENT_BOOSTING_CRITERION: list[str] = ['squared_error', 'friedman_mse']
GRADIENT_BOOSTING_MAX_FEATURES: list[str] = ['auto', 'sqrt', 'log2']
GRADIENT_BOOSTING_N_ESTIMATORS: list[int] = [8, 16, 32, 64, 128, 256]

K_NEIGHBOUR_N_NEIGHBOUR: list[int] = [5, 7, 9, 11]
K_NEIGHBOUR_WEIGHTS: list[str] = ['uniform', 'distance']
K_NEIGHBOUR_ALGORITHM: list[str] = ['ball_tree', 'kd_tree', 'brute']

XGB_LR: list[float] = [0.1, 0.01, 0.05, 0.001]
XGB_N_ESTIMATORS: list[int] = [8, 16, 32, 64, 128, 256]

CATBOOST_DEPTH: list[int] = [6, 8, 10]
CATBOOST_LR: list[float] = [0.01, 0.05, 0.1]
CATBOOST_ITERS: list[int] = [30, 50, 100]

ADABOOST_LR: list[float] = [0.1, 0.01, 0.5, 0.001]
ADABOOST_LOSS: list[str] = ['linear', 'square', 'exponential']
ADABOOST_N_ESTIMATORS: list[int] = [8, 16, 32, 64, 128, 256]

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

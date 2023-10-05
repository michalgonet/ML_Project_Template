from src.modules.data_management import DataManagement
from src.modules.data_preparation import DataTransformation
from src.modules.training_regression import ModelTrainer

if __name__ == "__main__":
    obj = DataManagement()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    dataset = data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    r2, best_model = model_trainer.initiate_model_trainer(dataset[0], dataset[1])
    print(r2, best_model)

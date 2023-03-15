import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from src.logger import logging
from src.exception import CustomException

from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split train and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }
            model_reports: dict = evaluate_models(X_train, X_test, y_train, y_test, models)

            # Get best model score from reports
            best_model_score = max(sorted(model_reports.values()))

            # Get best model name from resports
            best_model_name = list(model_reports.keys())[
                list(model_reports.values()).index(best_model_score)
            ] 
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found.')
            
            logging.info(f'Best model found on both training and test dataset.')
            
            # Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Prediction and Score
            prediction = best_model.predict(X_test)
            score = r2_score(y_test, prediction)
            return score            
        except Exception as e:
            raise CustomException(e, sys)

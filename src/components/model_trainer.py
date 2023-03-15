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
            params = {
                'Linear Regression': {},
                'Decision Tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': ['auto', 'sqrt', 'log2']
                },
                'Random Forest': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    # 'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None]
                },
                'Gradient Boosting': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001, 0.0001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    # 'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    # 'criterion': ['friedman_mse', 'squared_error'],
                    # 'max_features': ['auto', 'sqrt', 'log2']
                },
                'K-Neighbors Regressor': {
                    'n_neighbors': [5, 10, 15, 20, 25],
                    'weights': ['uniform', 'distance'],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                'XGBRegressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001, 0.0001],
                },
                'CatBoosting Regressor': {
                    'depth': [2, 4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [10, 20, 30, 40, 50]
                },
                'AdaBoost Regressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001, 0.0001],
                    # 'loss': ['linear', 'square', 'exponential']
                }
            }
            model_reports: dict = evaluate_models(X_train, X_test, y_train, y_test, models, params)

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
            logging.info(f'Best Model: {best_model_name}, Score: {score}')

            return score            
        except Exception as e:
            raise CustomException(e, sys)

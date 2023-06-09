import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    try:
        results = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            # Hyper parameter Tune
            param = params[list(models.keys())[i]]
            grid_search = GridSearchCV(model, param, cv=3)
            grid_search.fit(X_train, y_train)
            model.set_params(**grid_search.best_params_)
        
            # Train Model
            model.fit(X_train, y_train)

            # Make Prediction
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Get train and test score
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            results[list(models.keys())[i]] = test_score

        return results
    except Exception as e:
        raise CustomException(e, sys)
    
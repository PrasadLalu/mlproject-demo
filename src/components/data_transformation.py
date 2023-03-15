import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.utils import save_object
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_features = ['reading score', 'writing score']
            categorical_features = [
                'gender',                     
                'race/ethnicity', 
                'parental level of education', 
                'lunch',
                'test preparation course'
            ]

            num_pipeline = Pipeline(
                steps = [
                    ('impute', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Numerical features: {numerical_features}')
            logging.info(f'Categorical features: {categorical_features}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_set, test_set):
        try:
            train_df = pd.read_csv(train_set)
            test_df = pd.read_csv(test_set)

            logging.info('Read train and test data completed.')
            logging.info('Obtaining preprocessing object')

            target_feature = 'math score'
            numerical_features = ['reading score', 'writing score']

            input_feature_train_df = train_df.drop(columns=[target_feature], axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=[target_feature], axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info(
                f'Applying preprocessing object on training dataframe and testing dataframe.'
            )

            preprocessing_obj = self.get_data_transformer_object()
           
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path = self.data_transformation.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info(f'Saved preprocessing object.')

            return (
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
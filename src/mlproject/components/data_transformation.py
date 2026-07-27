import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object

@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:


            num_feature= ['reading_score', 'writing_score']
            cat_feature= ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                            'test_preparation_course']

            num_pipeline= Pipeline(steps=[

                ('imputer',SimpleImputer(strategy='median')),
                ('Scaleing',StandardScaler())
            ]) 

            cat_pipeline=Pipeline(steps=[

                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('One_Hot_Encodeing',OneHotEncoder())
            ])



            preprocessor = ColumnTransformer(
                [
                ('Numerical_Pipeline',num_pipeline,num_feature),
                ('Categorical_Pipeline',cat_pipeline,cat_feature)]

            )



            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the Train and Test data")

            preprocessing_obj = self.get_data_transformation()


# # Divide a train Dataset into a input and output feature

#             x_train = train_df.drop('math_score',axis=1)
#             y_train = train_df.math_score

# # Divide a test Dataset into a input and output feature

#             x_test = test_df.drop('math_score',axis=1)
#             y_test = test_df.math_score

#             logging.info("Divide A Train and Test Data Into A Input And Output Feature")

            preprocessed_train = preprocessing_obj.fit_transform(train_df)
            preprocessed_test = preprocessing_obj.transform(test_df)

            logging.info("Save Preprocessing Object")

            save_object(

                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                preprocessed_train,
                preprocessed_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)

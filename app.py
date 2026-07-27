from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import Dataingestion,Dataingestionconfig
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
import sys

if __name__ == "__main__" :
    logging.info("The Execution is Started")

    try:
         data_ingestion = Dataingestion()
         train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

         data_Transformation = DataTransformation()
         data_Transformation.initiate_data_transformation(train_data_path,test_data_path)


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
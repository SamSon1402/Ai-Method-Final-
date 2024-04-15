import logging
import pandas as pd
from zenml import step
from src.model_dev import RandomForestClassifierModel
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig

@step


def train_model(X_train :pd.DataFrame,
                X_test : pd.DataFrame,
                y_train : pd.DataFrame,
                y_test : pd.DataFrame,
                config : ModelNameConfig,
                ) -> ClassifierMixin:
    


    try:

        model = None
        if config.model_name == "RandomForestClassifier":
            model = RandomForestClassifierModel()
            trained_model = model.train(X_train , y_train)
            return trained_model
        else:

            raise ValueError("Model {} is not supported ".format(config.model_name))
        
    except Exception as e:
        logging.error("Erroe in Training the data {}".format(e))
        raise e
        
    


    






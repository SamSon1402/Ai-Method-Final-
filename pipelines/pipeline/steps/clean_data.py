import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning , DataDivideStrategy , DataPreProcessStrategy
from typing import Tuple
from typing_extensions import Annotated

@step

def clean_df(df : pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame , "X_train"],
    Annotated[pd.DataFrame , "X_test"],
    Annotated[pd.Series , "y_train"],
    Annotated[pd.Series , "y_test"],
]:


    try:

        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df , process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data , divide_strategy)
        X_train , X_test , y_train ,y_test  = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
    except Exception as e:
        logging.error("Error while cleaning data : {}".format(e))
        raise e


import logging
from abc import ABC , abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


class DataStrategy(ABC):

    @abstractmethod

    def handle_data(self, data : pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame :

        try:

            data = round((data.isnull().sum()*100 / data.shape[0]),2)
            data['Tenure'] = data['Tenure'].fillna(method = 'bfill')
            s_imp = SimpleImputer(missing_values=np.nan , strategy= 'most_frequent')
            data['WarehouseToHome'] = s_imp.fit_transform(pd.DataFrame(data['WarehouseToHome']))
            data['DaySinceLastOrder'] = data['DaySinceLastOrder'].fillna(method = 'bfill')

            data = data.select_dtypes(include = [np.number])
            return data
        except Exception as e:
            logging.error("Error in PreProcessing the data : {}".format(e))
            raise e
        

class DataDivideStrategy(DataStrategy):


    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:

        try:
            X = data.drop(["Churn"] , axis=1)
            y = data["Churn"]
            X_train ,X_test , y_train , y_test =train_test_split(X , y , test_size=0.2 , random_state=42)
            return X_train , X_test , y_train , y_test
        except Exception as e:
            logging.error("erroe while Dividing the data : {}".format(e))
            raise e
        


class DataCleaning:

    def __init__(self , data : pd.DataFrame , strategy : DataStrategy) :
        
        self.data = data
        self.strategy = strategy


    def handle_data(self) -> Union[pd.Series , pd.DataFrame]:

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error while handlimg the data {}".format(e))
            raise e
        
            




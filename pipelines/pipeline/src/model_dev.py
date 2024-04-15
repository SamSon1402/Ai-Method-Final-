import logging
from abc import ABC ,  abstractmethod
from sklearn.ensemble import RandomForestClassifier


class Model(ABC):

    @abstractmethod

    def train(self , X_train , y_train):

        pass


class RandomForestClassifierModel(Model):

     def train(self, X_train, y_train , **kwargs):
         

         try:
             rfc = RandomForestClassifier()
             rfc.fit(X_train , y_train)
             logging.info("Training Model Completed")
             return rfc
         except Exception as e:
             logging.error("Erroe while Training the Model : {}".format(e))
             
         
         





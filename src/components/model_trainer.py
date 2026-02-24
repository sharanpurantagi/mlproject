from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression , Lasso , Ridge
from xgboost import XGBRegressor
import sys
import os
from src.exception import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
from src.utils import evaluate_models
from sklearn.metrics import r2_score
from src.utils import save_object

class ModelTrainerConfig:
    def __init__(self):
        self.tarined_model_file_path=os.path.join("artifacts" , "model.pkl")


class model_trainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_training(self, training_arr , test_arr , preprocessor_path):
        try:
            xtrain, xtest, ytrain, ytest = (
                training_arr[:,:-1],
                test_arr[:,:-1],
                training_arr[:,-1],
                test_arr[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGBoostRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(),
                "AdaboostRegressor": AdaBoostRegressor()
            }
            params = {

    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [None, 5]
    },

    "decision Tree": {
        "criterion": ["squared_error"],
        "max_depth": [None, 5]
    },

    "Gradient Boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.1, 0.05]
    },

    "K-Neighbours Regressor": {
        "n_neighbors": [3, 5],
        "weights": ["uniform"]
    },

    "XGBoostRegressor": {
        "n_estimators": [50],
        "learning_rate": [0.1],
        "max_depth": [3]
    },

    "CatBoostRegressor": {
        "iterations": [100],
        "learning_rate": [0.1],
        "depth": [4]
    },

    "AdaboostRegressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.1]
    }
}
            model_report:dict=evaluate_models(xtrain , xtest , ytrain , ytest , models ,  params)

            # Best Model Score
            best_model_score=max(sorted(model_report.values()))
            
            # Best model name:
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            save_object(file_path=self.model_trainer_config.tarined_model_file_path , obj=best_model)
            predicted=best_model.predict(xtest)
            r_score=r2_score(ytest , predicted)
            return r_score
            





        except Exception as e:
            raise CustomException(e, sys)
    

        
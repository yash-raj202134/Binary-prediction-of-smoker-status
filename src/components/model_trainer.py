import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False)
            }

            params = {
                "Logistic Regression": {
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'max_iter': [50, 100, 200, 500]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                },
                "XGBoost": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 4, 5, 6, 7],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
                    'reg_alpha': [0, 0.1, 0.5, 1],
                    'reg_lambda': [0, 1, 1.5, 2, 3],
                    'min_child_weight': [1, 3, 5, 7],
                    'scale_pos_weight': [1, 2, 3, 4]
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128],
                    'bagging_temperature': [0, 1, 5, 10],
                    'random_strength': [0, 0.1, 0.5, 1],
                    'scale_pos_weight': [None, 1, 2, 3]
                },
                "AdaBoost": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [.1, .01, 0.5, .001],
                    'algorithm': ['SAMME', 'SAMME.R']
                },
                "K-Nearest Neighbors":{
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'p': [1, 2]  
                }
}


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param = params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.76:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)

            accuracy_sco = accuracy_score(y_test,predicted)

            return accuracy_sco

        except Exception as e:
            raise CustomException(e,sys)
    






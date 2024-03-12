import sys
import pandas as pd
import numpy as np
import os
from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):

        try:

            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","proprocessor.pkl")
            print("Before Loading")
    
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


        
class CustomData:
    def __init__(self,
        age:int,
        height: int,
        weight: int,
        systolic:int,
        relaxation:int,
        fasting_blood_sugar:int,
        Cholesterol:int,
        triglyceride:int,
        HDL:int,
        LDL:int,
        hemoglobin:float,
        Urine_protein:int,
        serum_creatinine:float,
        AST:int,
        ALT:int,
        Gtp:int,
        dental_caries:int,
        eyesight_diff:float,
        hearing_diff:int
        ):

        self.age = age
        self.height = height
        self.weight = weight
        self.systolic = systolic
        self.relaxation = relaxation
        self.fasting_blood_sugar = fasting_blood_sugar
        self.Cholesterol = Cholesterol
        self.triglyceride = triglyceride
        self.HDL = HDL
        self.LDL = LDL
        self.hemoglobin = hemoglobin
        self.Urine_protein = Urine_protein
        self.serum_creatinine = serum_creatinine
        self.AST = AST
        self.ALT = ALT
        self.Gtp = Gtp
        self.dental_caries = dental_caries
        self.eyesight_diff = eyesight_diff
        self.hearing_diff = hearing_diff

    

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "height(cm)": [self.height],
                "weight(kg)": [self.weight],
                "systolic": [self.systolic],
                "relaxation": [self.relaxation],
                "fasting blood sugar": [self.fasting_blood_sugar],
                "Cholesterol":[self.Cholesterol],
                "triglyceride": [self.triglyceride],
                "HDL": [self.HDL],
                "LDL": [self.LDL],
                "hemoglobin": [self.hemoglobin],
                "Urine protein": [self.Urine_protein],
                "serum creatinine": [self.serum_creatinine],
                "AST": [self.AST],
                "ALT": [self.ALT],
                "Gtp": [self.Gtp],
                "dental caries": [self.dental_caries],
                "eyesight_diff": [self.eyesight_diff],
                "hearing_diff": [self.hearing_diff],

            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
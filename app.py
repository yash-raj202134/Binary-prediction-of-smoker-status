from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData ,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=request.form.get('age'),
            height=request.form.get('height(cm)'),
            weight=request.form.get('weight(kg)'),
            systolic=request.form.get('systolic'),
            relaxation=request.form.get('relaxation'),
            fasting_blood_sugar=request.form.get('fasting blood sugar'),
            Cholesterol=request.form.get('Cholesterol'),
            triglyceride=request.form.get('triglyceride'),
            HDL=request.form.get('HDL'),
            LDL=request.form.get('LDL'),
            hemoglobin=float(request.form.get('hemoglobin')),
            Urine_protein=request.form.get('Urine protein'),
            serum_creatinine=float(request.form.get('serum creatinine')),
            AST=request.form.get('AST'),
            ALT=request.form.get('ALT'),
            Gtp=request.form.get('Gtp'),
            dental_caries=request.form.get('dental caries'),
            eyesight_diff=float(request.form.get('eyesight_diff')),
            hearing_diff=request.form.get('hearing_diff')
                                      
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print(f"after Prediction: \n{results}")
        if results[0] == 0:
            status = 'Non Smoker'
        else:
            status = 'Smoker'

        return render_template('home.html',results=status)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug =True)        


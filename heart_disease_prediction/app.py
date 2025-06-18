from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import RadioField, BooleanField, IntegerField, DecimalField, SubmitField
from wtforms.validators import DataRequired, NumberRange
import joblib
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

# Define the form 
class InputForm(FlaskForm):
    age = IntegerField('Age' , validators=[DataRequired(), NumberRange(min=18, max=100,message="Enter Your Age")])
    sex = RadioField('Gender',choices=['Male', 'Female'],validators=[DataRequired()])
    cp = RadioField('Chest Pain',choices=['typical angina','asymptomatic','non-anginal','atypical angina'], validators=[DataRequired()])
    trestbps = IntegerField('Resting Blood Pressure', validators=[DataRequired()])
    chol = IntegerField('Cholestrol',validators=[DataRequired()])
    fbs = RadioField('Fasting Blood Sugar',choices=['True','False'],validators=[DataRequired()])
    restecg = RadioField('Resting ECG',choices=['lv hypertrophy','normal','st-t abnormality'], validators=[DataRequired()])
    thalch = IntegerField('Maximum Heart Rate achieved',validators=[DataRequired()])
    exang = RadioField('Exercise Included Angina',choices=['True','False'],validators=[DataRequired()])
    oldpeak = DecimalField('ST depression',validators=[DataRequired(),NumberRange(min=0.0, max=4.0)])
    slope = RadioField('Slope', choices=['downsloping', 'flat', 'upsloping'],validators=[DataRequired()])
    ca = IntegerField('Number of Major vessels(ca)', validators=[DataRequired(),NumberRange(min=0,max=4)])
    thal = RadioField('Thalassemia',choices=['fixed defect','normal','reversable defect'],validators=[DataRequired()])
    submit = SubmitField('Submit')
    
# categorical mapping
mapping = {
            "sex":{"Male":1, "Female":0},
            "cp":{"typical angina":3,"asymptomatic":0,"non-anginal":2, "atypical angina":1},
            "fbs":{"True":1, "False":0},
            "restecg":{'lv hypertrophy':0, 'normal':1, 'st-t abnormality':2},
            "exang":{"False":0,  "True":1},
            "slope":{'downsloping':0, 'flat':1, 'upsloping':2},
            "thal":{'fixed defect':0,'normal':1,'reversable defect':2}
        }
# Load the model
model = joblib.load('heart_disease_model.pkl')

def home():
    return render_template('index.html')
@app.route('/', methods=['GET', 'POST'])

def predict():
    form = InputForm()
    converted_data = {}
    result=None
    prediction=None
    if form.validate_on_submit():
        try:
            age = form.age.data
            gender_val = form.sex.data
            cp_val =form.cp.data
            trestbps = form.trestbps.data
            chol = form.chol.data
            fbs_val = form.fbs.data
            restecg_val = form.restecg.data
            thalch =form.thalch.data
            exang_val = form.exang.data
            oldpeak =form.oldpeak.data
            slope_val = form.slope.data
            ca = form.ca.data
            thal_val=form.thal.data
            
            sex = mapping["sex"].get(gender_val,-1)
            cp = mapping["cp"].get(cp_val,-1)
            fbs = mapping["fbs"].get(fbs_val,-1)
            restecg = mapping["restecg"].get(restecg_val,-1)
            exang = mapping["exang"].get(exang_val,-1)
            slope = mapping["slope"].get(slope_val,-1)
            thal = mapping["thal"].get(thal_val,-1)
            
            converted_data ={
                "age ":age ,
                "sex":sex,
                "cp" :cp,
                "trestbps": trestbps,
                "chol" : chol,
                "fbs" : fbs,
                "restecg":restecg,
                "thalch" :thalch,
                "exang":exang,
                "oldpeak":oldpeak,
                "slope":slope,
                "ca" : ca,
                "thal":thal
            }
            # Get form data
            features = [float(x) for x in converted_data.values()]
            final_features = np.array(features).reshape(1, -1)
            # return f"<h3>Converted Data: {converted_data}</h3><h3>{final_features}</h3>"
            
            prediction = model.predict(final_features)

            result = "At risk of heart disease" if prediction[0] == 1 else "Not at risk"
            return f"<h3>Prediction: {prediction}</h3>"
        except Exception as e:
            return f"Error: {e}"
    return render_template('index.html',form=form,prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, render_template, redirect, url_for,request
import numpy as np
import joblib

model = joblib.load('dt_model.sav')
scaler = joblib.load('scaler.pkl')
app = Flask(__name__)

@app.route('/<result>')
def predict(result):
    return f"<h1> Stroke Status(0 for no and 1 for maybe):{result}</h1>"

@app.route('/',methods=['POST','GET'])
def home():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level =float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        content = np.array([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]])
        print(content)
        scaled_content = scaler.fit_transform(content)
        prediction = model.predict(scaled_content)
        print(prediction)

        return redirect(url_for("predict",result=int(prediction)))

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run()
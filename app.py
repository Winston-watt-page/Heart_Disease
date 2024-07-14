from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)
model_path =r'c:\Users\winst_fpd3d9v\OneDrive\Documents\Heart_Disease\Heart_Disease_model.pkl'
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(features)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

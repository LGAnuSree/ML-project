# app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Load the Naive Bayes model
model = GaussianNB()

# Load dataset
df = pd.read_csv("diabetes.csv")
print(df.shape)
X = df.drop(columns='diabetes')
y = df['diabetes']

# Get feature names
feature_names = X.columns.tolist()

# Fit the model with feature names
model.fit(X, y)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        feature_dict = {feature_names[i]: features[i] for i in range(len(features))}
        prediction = model.predict([features])[0]
        return render_template('result.html', prediction=prediction, feature_dict=feature_dict)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)

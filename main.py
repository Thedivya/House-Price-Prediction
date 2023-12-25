from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)
data = pd.read_csv('minor_cleaned_01.csv')
piperidge = pickle.load(open("Ridge.pkl",'rb'))
pipelasso = pickle.load(open("Lasso.pkl",'rb'))
pipelinear = pickle.load(open("LinearRegression.pkl",'rb'))




@app.route('/')
def index():

    return render_template('index.html')


@app.route('/linear_regression.html')
def linear_regression():
        locations = sorted(data['location'].unique())
        area_type = sorted(data['area_type'].unique())
        return render_template('linear_regression.html', locations=locations, area_type =area_type)

@app.route('/predict_linear', methods=['POST'])
def predict_linear():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    balcony = request.form.get('balcony')
    area_type = request.form.get('area_type')

    print(location, bhk, bath, sqft, balcony, area_type )
    input = pd.DataFrame([[location,sqft,bath,bhk,balcony,area_type]],columns=['location','total_sqft', 'bath', 'bhk', 'balcony', 'area_type'])
    
    prediction = pipelinear.predict(input)[0] * 1e5

    return str(np.round(prediction,2))


@app.route('/lasso.html')
def lasso():
        locations = sorted(data['location'].unique())
        area_type = sorted(data['area_type'].unique())
        return render_template('lasso.html', locations=locations, area_type =area_type)

@app.route('/predict_lasso', methods=['POST'])
def predict_lasso():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    balcony = request.form.get('balcony')
    area_type = request.form.get('area_type')

    print(location, bhk, bath, sqft, balcony, area_type )
    input = pd.DataFrame([[location,sqft,bath,bhk,balcony,area_type]],columns=['location','total_sqft', 'bath', 'bhk', 'balcony', 'area_type'])
    
    prediction = pipelasso.predict(input)[0] * 1e5

    return str(np.round(prediction,2))


@app.route('/ridge.html')
def ridge():  
        locations = sorted(data['location'].unique())
        area_type = sorted(data['area_type'].unique())
        return render_template('ridge.html', locations=locations, area_type =area_type)

@app.route('/predict_ridge', methods=['POST'])
def predict_ridge():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    balcony = request.form.get('balcony')
    area_type = request.form.get('area_type')

    print(location, bhk, bath, sqft, balcony, area_type )
    input = pd.DataFrame([[location,sqft,bath,bhk,balcony,area_type]],columns=['location','total_sqft', 'bath', 'bhk', 'balcony', 'area_type'])
    
    prediction = piperidge.predict(input)[0] * 1e5

    return str(np.round(prediction,2))
    




if __name__== "__main__":
    app.run(debug=True)
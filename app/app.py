import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('Salary_Estimater.pkl', 'rb'))

def get_dummies(value, possible_values):
    return [1 if val == value else 0 for val in possible_values]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    rating = float(request.form.get('rating'))
    job_title = request.form.get('job-title')
    state = request.form.get('state')

    # Get Dummies for the two categorical features
    job_title_encoded = get_dummies(job_title, ['analyst', 'data engineer', 'data scientist', 'deep learning engineer', 'director', 'machine learning engineer', 'manager', 'other', 'researcher'])
    state_encoded = get_dummies(state, ['AZ', 'CA', 'DC', 'DE', 'FL', 'GA', 'IL', 'IN', 'KY', 'MA', 'MD', 'ME', 'MI', 'NC', 'NH', 'NJ', 'NM', 'NY', 'OH', 'PA', 'TX', 'VA', 'WI'])


    # Combining all features
    final_features = [rating] + job_title_encoded + state_encoded
    final_features = [np.array(final_features)]

    prediction = model.predict(final_features)


    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {} K'.format(output))

if __name__ == '__main__':
    app.run(debug=True, port=5012)

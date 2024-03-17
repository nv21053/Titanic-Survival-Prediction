from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('titanic_model.pkl')

# Define a function to preprocess user inputs
def preprocess_input(sex, age, sibsp, parch, fare, embarked, cls, who, alone):
    # Convert categorical values to numeric
    sex = 0 if sex.lower() == 'female' else 1
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    embarked = embarked_mapping.get(embarked.upper(), -1)
    class_mapping = {'First': 0, 'Second': 1, 'Third': 2}
    cls = class_mapping.get(cls, -1)
    who_mapping = {'child': 0, 'woman': 1, 'man': 2}
    who = who_mapping.get(who.lower(), -1)
    alone = int(alone)

    # Create a numpy array with the preprocessed values
    inputs = np.array([sex, age, sibsp, parch, fare, embarked, cls, who, alone]).reshape(1, -1)
    return inputs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs from the form
        sex = request.form['sex']
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        cls = request.form['class']
        who = request.form['who']
        alone = request.form['alone']
        
        # Preprocess the user inputs
        inputs = preprocess_input(sex, age, sibsp, parch, fare, embarked, cls, who, alone)

        # Make prediction using the loaded model
        prediction = model.predict(inputs)
        survived = 'Yes' if prediction[0] == 1 else 'No'

        # Render the result page with the prediction
        return render_template('result.html', survived=survived)
    else:
        # Render the input form page
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
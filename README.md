# Titanic Survival Prediction

This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. It is implemented with Python, Flask, and scikit-learn.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/nv21053/titanic-survival-prediction.git
```

2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset files (`Titanic_preprocessed.csv` and `Titanic_model.pkl`) and place them in the project's root directory.

## Usage

1. Run the Flask web server:

```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`.

3. Fill in the passenger details in the provided form and click the "Predict" button.

4. The application will predict the survival of the passenger and display the result on the "Result" page.

## File Structure

- `app.py`: The Flask application script.
- `templates/index.html`: The HTML template for the web application's home page.
- `templates/result.html`: The HTML template for the result page.
- `static/style.css`: CSS file for styling the web application.
- `creating_model.py`: Python script for creating and training the machine learning model.
- `process.py`: Python script for preprocessing the Titanic dataset.
- `Titanic.csv`: Original dataset file containing passenger information.
- `Titanic_preprocessed.csv`: Preprocessed dataset file used for training the model.
- `Titanic_model.pkl`: Pickle file containing the trained machine learning model.

## Dependencies

- Python: Programming language used for the project.
- Flask: Web framework for Python.
- scikit-learn: Machine learning library for classification and regression tasks.
- pandas: Data manipulation and analysis library.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please create an issue or submit a pull request.

---

**Note:** This project is for educational purposes and is based on the famous Titanic dataset. The dataset can be found on [Kaggle](https://www.kaggle.com/c/titanic).

import pandas as pd

# Read the CSV file
data = pd.read_csv('Titanic.csv')

# Preprocess the data
data.dropna(inplace=True)  # Remove rows with missing values
data = data[['sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'alone', 'survived']]  # Select relevant columns
data['sex'] = data['sex'].map({'female': 0, 'male': 1})  # Convert 'sex' column to numeric values
data['embarked'] = data['embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # Convert 'embarked' column to numeric values
data['class'] = data['class'].map({'First': 0, 'Second': 1, 'Third': 2})  # Convert 'class' column to numeric values
data['who'] = data['who'].map({'child': 0, 'woman': 1, 'man': 2})  # Convert 'who' column to numeric values
data['alone'] = data['alone'].astype(int)  # Convert 'alone' column to integer values
data['survived'] = data['survived'].astype(int)  # Convert 'survived' column to integer values

# Save the preprocessed data to a new CSV file
data.to_csv('Titanic_preprocessed.csv', index=False)
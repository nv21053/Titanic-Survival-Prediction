import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the preprocessed data
data = pd.read_csv('Titanic_preprocessed.csv')

# Select specific columns for the feature matrix
selected_columns = ['sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'alone']
X = data[selected_columns]

# Select the target variable
y = data['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
model = RandomForestClassifier(n_estimators=100)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the trained model to a file
joblib.dump(model, 'titanic_model.pkl')
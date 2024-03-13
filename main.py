# Data Loading Code Hidden Here
import pandas as pd

# Load data
titanic_path = '../input/titanic/train.csv'
titanic_data = pd.read_csv(titanic_path)

titanic_data['Sex'] = titanic_data['Sex'].map({'male': 1, 'female': 0})
filt_titanic_data = titanic_data.dropna(axis=0)
# Filter rows with missing price values
# Choose target and features
y = filt_titanic_data.Survived
titanic_features = ['Pclass', 'Sex', 'Age', 'SibSp',
                        'Parch']
X = filt_titanic_data[titanic_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
titanic_model = DecisionTreeRegressor()
# Fit model
print(titanic_model.fit(X, y))
predictions = titanic_model.predict(X)

# Load test data
test_path = '../input/titanic/test.csv'
test_data = pd.read_csv(test_path)

# Apply the same transformation to the 'Sex' column in the test data
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})

# Fill missing values in the 'Age' column of the test data with the median age
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

# Select features for the test data
X_test = test_data[titanic_features]

# Make predictions using the trained model
test_predictions = titanic_model.predict(X_test)
test_predictions = test_predictions.astype(int)

# Create a DataFrame for submission
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})

# Save the submission to a CSV file
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
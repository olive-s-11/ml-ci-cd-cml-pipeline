import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Dummy data for training
data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1], 'target': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

X = df[['feature1', 'feature2']]
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.joblib')
print("Training complete!")

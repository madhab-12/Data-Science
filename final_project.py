import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create a new dataset for demonstration (Car Data)
data = {
    'Mileage': [25, 30, 22, 28, 20, 32, 27, 24, 18, 35],  # Miles per gallon
    'EngineSize': [2.0, 1.6, 2.5, 1.8, 3.0, 1.5, 2.2, 2.4, 3.5, 1.4],  # Engine size in liters
    'Age': [5, 3, 7, 2, 9, 1, 4, 6, 8, 2],  # Age of the car in years
    'Price': [15000, 12000, 10000, 20000, 8000, 17000, 13000, 11000, 7000, 19000]  # Price in USD
}

# Convert the dataset to a Pandas DataFrame
df = pd.DataFrame(data)

# Define features (X) and target variable (y)
X = df[['Mileage', 'EngineSize', 'Age']]  # Features: Mileage, Engine Size, Age
y = df['Price']  # Target variable: Price

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Visualize the predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Car Prices")

# Add a line for perfect predictions
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.legend()
plt.show()


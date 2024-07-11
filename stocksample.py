import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
file_path = 'C:/Users/asus/Downloads/BRITANNIA.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Prepare the data
data['Date'] = pd.to_datetime(data['Date'])
data['Timestamp'] = data['Date'].map(pd.Timestamp.timestamp)

# Additional feature engineering (e.g., day of the week, month)
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# Split the data into features and target
X = data[['Timestamp', 'DayOfWeek', 'Month']]
y = data['Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a more sophisticated model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(data['Timestamp'], data['Close'], color='blue', label='Actual Prices')
plt.scatter(X_test['Timestamp'], y_pred, color='red', label='Predicted Prices')
plt.xlabel('Timestamp')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()

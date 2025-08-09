import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load the dataset
data = pd.read_csv('/Users/edersonmarcellus/Downloads/home_dataset.csv')

# Extract features and target variable
house_size = data['HouseSize'].values
house_price = data['HousePrice'].values


# Visualize the data
plt.scatter(house_size, house_price, marker='o', color='blue')
plt.title('House Size vs Price')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($)')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(house_size.reshape(-1, 1), house_price, test_size=0.2, random_state=42)

# Reshape the data for Numpy
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Visualize the predictions
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, predictions, color='red', label='Predicted Prices')
plt.title('House Price Predictions')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()

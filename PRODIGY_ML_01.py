# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the train dataset
df = pd.read_csv('train.csv', usecols=['LotArea', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'SalePrice'])
df['TotalBathrooms'] = df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath']
X= df[['LotArea', 'BedroomAbvGr', 'TotalBathrooms']]
y = df['SalePrice']

# Concatencate X and Y in to remove outliers
data = pd.concat([X, y], axis = 1)

# A function to remove outliers using IQR method
def remove_IRQ_outliers(df, columns):
    Q1 = df[columns].quantile(0.25) # 25th percentile
    Q3 = df[columns].quantile(0.75) # 75th percentile
    IRQ = Q3 - Q1                   # Interquartile Range

    # Create a filter that removes rows with any outlier in the specified columns
    filter = ~((df[columns] < (Q1 - 1.5 * IRQ)) | (df[columns] > (Q3 + 1.5 * IRQ))).any(axis=1)
    return df[filter]

# Apply he IQR function to remove outliers
data_clean = remove_IRQ_outliers(data, ['LotArea', 'BedroomAbvGr', 'TotalBathrooms', 'SalePrice'])
X_clean = data_clean[['LotArea', 'BedroomAbvGr', 'TotalBathrooms']].values
X_clean = np.array(X_clean)
y_clean = data_clean['SalePrice'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size = 0.2, random_state = 1)

# Feature Scaling for the data
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Initialize and fit the linear regression model
linear = LinearRegression()
linear.fit(X_train, y_train)

# Predict the values for X_test
y_pred = linear.predict(X_test)

# Inverse transform the scaled predictions to get back the original scale
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

# Output the model coefficients and intercept
print(f"Model coefficients: {linear.coef_}")
print(f"Model intercept: {linear.intercept_}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2}")

# Scatter plot for actual vs predicted values
plt.scatter(y_test, y_pred, color='blue', label='Predicted')

# Plotting the identity line 
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Perfect Prediction Line')

# Add labels and title
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.legend()

# Show the plot
plt.show()
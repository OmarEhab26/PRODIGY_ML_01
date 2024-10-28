# House Price Prediction Using Linear Regression

## Overview
This project aims to predict house prices based on certain features using a linear regression model. The primary dataset used for training is `train.csv`, which contains various features related to houses, including the lot area, number of bedrooms, and bathrooms. The output variable is the sale price of the houses.

## Code Breakdown

### 1. Importing Libraries
The code begins by importing necessary libraries:
- `numpy`: For numerical operations and handling arrays.
- `pandas`: For data manipulation and analysis, particularly for handling data in tabular form.
- `matplotlib.pyplot`: For plotting graphs to visualize results.
- `sklearn.preprocessing`: To scale the features for better model performance.
- `sklearn.linear_model`: To implement the linear regression model.
- `sklearn.model_selection`: For splitting the dataset into training and testing sets.
- `sklearn.metrics`: To evaluate the performance of the regression model.

### 2. Loading the Dataset
The dataset is loaded using `pandas`:
- The relevant columns (`LotArea`, `BedroomAbvGr`, `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`, and `SalePrice`) are imported from `train.csv`.
- A new column, `TotalBathrooms`, is calculated by summing the bathroom-related columns.

### 3. Preparing the Features and Target Variable
The features (`X`) and target variable (`y`) are defined:
- `X` includes the selected features: `LotArea`, `BedroomAbvGr`, and `TotalBathrooms`.
- `y` is the `SalePrice` column.

### 4. Removing Outliers
Outliers can significantly affect the performance of a regression model. The code defines a function, `remove_IRQ_outliers`, which:
- Calculates the first quartile (Q1) and the third quartile (Q3) for the specified columns.
- Computes the interquartile range (IRQ) and uses it to filter out any rows containing outliers in the selected columns.
- The function is applied to the combined data (`data`) to clean it, resulting in `data_clean`.

### 5. Splitting the Data
The cleaned dataset is then split into training and testing sets:
- The data is split into `X_train`, `X_test`, `y_train`, and `y_test`, with 20% of the data reserved for testing.

### 6. Feature Scaling
To improve the performance of the regression model, feature scaling is performed:
- A `StandardScaler` is initialized to standardize the features by removing the mean and scaling to unit variance.
- The scaler is fit on the training data and then used to transform both the training and testing data.

### 7. Initializing and Fitting the Linear Regression Model
A linear regression model is created using `LinearRegression`:
- The model is fitted on the scaled training data (`X_train` and `y_train`).

### 8. Making Predictions
After the model is trained, it is used to make predictions:
- Predictions for the testing set (`X_test`) are generated, resulting in `y_pred`.

### 9. Evaluating the Model
The performance of the regression model is assessed using various metrics:
- The predicted and actual values of the sale prices are printed for comparison.
- The model coefficients and intercept are displayed, providing insights into the relationship between the features and the target variable.
- Metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) are calculated and printed to evaluate model accuracy.

### 10. Visualizing Results
Finally, a scatter plot is created to visualize the relationship between the actual sale prices and the predicted sale prices:
- The scatter plot displays predicted values against actual values, with a red identity line indicating where the predicted values would fall if the predictions were perfect.
- Labels and a title are added to the plot for clarity.

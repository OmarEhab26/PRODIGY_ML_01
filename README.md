# House Price Prediction

This project implements a linear regression model to predict house prices based on selected features from a housing dataset. The model uses the Lot Area, Number of Bedrooms, and Total Bathrooms as input features to predict the Sale Price of the house.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The aim of this project is to predict house prices using a linear regression model. The dataset is cleaned to remove outliers using the Interquartile Range (IQR) method. The features considered for prediction include:

- Lot Area
- Number of Bedrooms Above Ground
- Total Number of Bathrooms

The model's performance is evaluated using various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).

## Installation

To run this project, you will need to install the required Python libraries. You can install them using pip.

# Install the required libraries
pip install numpy pandas matplotlib scikit-learn


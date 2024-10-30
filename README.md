# House Price Prediction
This project focuses on predicting house prices based on various features such as size, location, and number of rooms using Linear Regression.<br><br>
![148332938-4e66d4ca-2d16-474f-8482-340aef6a48d0](https://github.com/user-attachments/assets/7e3be8b8-4d6c-45de-9962-5ba5865e2ebc)


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data-Processing](#data-processing)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Output](#output)
- [Conclusion](#conclusion)


## Introduction

The goal of this project is to predict house prices using a Linear Regression model. By leveraging historical data on various features, we aim to develop a model that can accurately estimate housing prices.

## Installation

To run this project, you need to have Python and the following libraries installed:
- pandas
- NumPy
- Matplotlib
- seaborn
- scikit-learn


You can install the required libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```


## Dataset

The dataset used in this project is the "USA_Housing.csv" file, which contains data on various housing features and prices.

## Data Processing
The data processing steps include:
1. Load Data: Read the dataset into a Pandas DataFrame.
2. Visualize Data: Use pair plots and distribution plots to understand the data distribution and relationships.
3. Handle Non-Numeric Data: Identify and drop non-numeric columns.
4. Correlation Heatmap: Generate a heatmap to visualize the correlation between features.

## Model
The model used in this project is a Linear Regression model, implemented using scikit-learn.


## Training
The training process involves the following steps:
1. Split Data: Split the dataset into training and testing sets (70% training, 30% testing).
2. Train Model: Fit the Linear Regression model on the training data.
3. Evaluate Model: Use the testing set to evaluate the model's performance.

## Evaluation
The model's performance is evaluated using the following metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The model achieved the following results:
- MAE: 82,288
- MSE: 10,460,958,907
- RMSE: 102,279
- R²: 0.918

## Output
The following visualizations are generated to better understand the model's performance:
<br>
- Scatter plot of actual vs predicted prices.<br><br>
![Screenshot 2024-10-20 052812](https://github.com/user-attachments/assets/100e64f4-e3ab-4aec-9db7-da5739c61a4f)<br><br>


- Distribution plot of residuals.<br><br>
![Screenshot 2024-10-20 053300](https://github.com/user-attachments/assets/05d266f4-f14a-492f-b216-f266e9132b41)



## Conclusion
This project successfully demonstrates the use of Linear Regression for predicting house prices. The model provides accurate predictions based on historical data, showcasing the potential of machine learning in real estate applications.

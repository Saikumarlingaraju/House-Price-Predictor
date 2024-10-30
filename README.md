# House-Price-Predictor🏡

![148332938-4e66d4ca-2d16-474f-8482-340aef6a48d0](https://github.com/user-attachments/assets/7e3be8b8-4d6c-45de-9962-5ba5865e2ebc)

## Overview

This project predicts U.S. house prices based on various factors like average area income, house age, number of rooms, number of bedrooms, and area population. A Linear Regression model is trained and evaluated to determine price predictions with metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R².

## Live Demo

Explore the House Price Predictor! 👉🏻 [![Experience It! 🌟](https://img.shields.io/badge/Experience%20It!-blue)](https://valuemyhouse.streamlit.app/)

<br>

_Below is a preview of the House Price Predictor in action. Input various housing factors to predict the price. Check out the user-friendly interface and data-driven predictions!_ 👇🏻
<p align="center">
  <img src="https://github.com/user-attachments/assets/cb2a1203-2da9-4c21-9ba4-a528b671730a" alt="house">
</p>

<br>


## Table of Contents

1. [Features](#features)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Technologies Used](#technologies-used)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [Contact](#contact)

<br>

## Features🌟

- Loads and preprocesses a dataset of U.S. housing prices.
- Cleans data, removing non-numeric columns.
- Trains a Linear Regression model on the dataset for price prediction.
- Saves the trained model and preprocessed data for future use.

<br>

## Dataset📊

- **Source**: The dataset (`USA_Housing.csv`) includes attributes like:
  - Avg. Area Income
  - Avg. Area House Age
  - Avg. Area Number of Rooms
  - Avg. Area Number of Bedrooms
  - Area Population
  - Price (target variable)

<br>

## Data Preprocessing🛠

1. **Data Cleaning**: Dropped non-numeric columns to focus on numeric predictors.
2. **Feature Scaling**: Applied where necessary for model consistency.

<br>

## Model Training🧠

- The model used for this project is **Linear Regression**.
- **Train-Test Split**: 60% of the data is used for training, and 40% for testing.

<br>

## Evaluation📈

Each model is evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

<br>

## Installation🛠

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hk-kumawat/housepricepredictor.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

<br>

## Usage🚀

1. **Train the model**: Run the main script to train and evaluate the model.
2. **Model Inference**:
   - Use `model.pkl` for predictions on test data.
   - Preprocessed data is saved in `df.pkl`.

<br>

## Technologies Used💻

- Python
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- Deployment: Streamlit for UI

<br>

## Results🏆

- The model achieved a reasonable prediction accuracy, with error metrics calculated as follows:
  - **MAE**: The average absolute difference between the predicted and actual prices.
  - **MSE** and **RMSE**: Measures of the model's accuracy, with RMSE providing insight into the spread of errors.
  - **R²**: Indicates the proportion of variance explained by the model.

<br>  

## Conclusion📚

The House Price Predictor project demonstrates the potential of Linear Regression in forecasting property prices based on key housing features. With a streamlined model and preprocessed dataset, this project emphasizes the importance of data cleaning and error metric analysis in real estate price predictions. Deploying the model on Streamlit provides users with easy access to housing predictions.

<br>

## Contact

### 📬 Get in Touch!
I’d love to hear from you! Feel free to reach out:

- [![GitHub](https://img.shields.io/badge/GitHub-hk--kumawat-blue?logo=github)](https://github.com/hk-kumawat) 💻 — Explore my projects and contributions.
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Harshal%20Kumawat-blue?logo=linkedin)](https://www.linkedin.com/in/harshal-kumawat/) 🌐 — Let’s connect professionally.
- [![Email](https://img.shields.io/badge/Email-harshalkumawat100@gmail.com-blue?logo=gmail)](mailto:harshalkumawat100@gmail.com) 📧 — Send me an email for any in-depth discussions.

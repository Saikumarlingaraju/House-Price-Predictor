<a id="readme-top"></a>

# House-Price-Predictor🏡

![148332938-4e66d4ca-2d16-474f-8482-340aef6a48d0](https://github.com/user-attachments/assets/7e3be8b8-4d6c-45de-9962-5ba5865e2ebc)

## Overview

This project is a house price prediction system built using **Streamlit** and **scikit-learn**. It uses a linear regression model trained on the **USA Housing** dataset to estimate house prices based on factors such as area income, house age, number of rooms, number of bedrooms, and area population. The interactive web app allows users to input these parameters and receive real-time price predictions with an intuitive and clean interface.

## Live Demo

Explore the House Price Predictor! 👉🏻 [![Experience It! 🌟](https://img.shields.io/badge/Experience%20It!-blue)](https://valuemyhouse.streamlit.app/)

<br>

_Below is a preview of the House Price Prediction System in action. Enter house details to get an instant price estimate! 👇🏻_
<p align="center">
  <img src="https://github.com/user-attachments/assets/bfff63d3-a2f3-4aa6-8da5-ad89bcab1c74" alt="house">
</p>

<br>

### Learning Journey 🗺️

This project represents my venture into real estate analytics and machine learning. Here's my journey:

- **Inspiration:**  
  The volatile housing market inspired me to create a tool that could help people make informed decisions about real estate investments using data science.

- **Why I Made It:**  
  I wanted to build a practical application that demonstrates how machine learning can solve real-world problems while making complex predictions accessible to everyone through a simple interface.

- **Challenges Faced:**  
  - **Data Quality:** Ensuring the dataset was clean and representative of real market conditions.  
  - **Feature Selection:** Identifying the most impactful features for accurate price prediction.  
  - **Model Selection:** Balancing model complexity with prediction accuracy.  
  - **UI/UX Design:** Creating an interface that's both informative and easy to use.

- **What I Learned:**  
  - **Data Analysis:** Advanced techniques in data preprocessing and visualization.  
  - **Machine Learning:** How to implement and evaluate linear regression models effectively.  
  - **Web Development:** Building interactive web applications using Streamlit.  
  - **Statistical Analysis:** Gaining insights into correlations and feature importance in the context of real estate.

This journey has not only improved my technical skills but also deepened my understanding of real estate analytics and its potential to drive smart investment decisions.


<br>


## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Training](#model-training)
8. [Results](#results)
10. [Future Enhancements](#future-enhancements)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contact](#contact)

    
<br>

## Features🌟

- **Accurate Predictions:**  
  Leverages a trained linear regression model for reliable house price estimation.
- **Interactive UI:**  
  A user-friendly Streamlit app where you can enter housing details and get instant price predictions.
- **Visual Data Insights:**  
  Notebooks include data visualizations such as pair plots, distribution plots, and heatmaps to understand feature relationships.
- **Model Persistence:**  
  The trained model and scaler are saved as pickle files, ensuring quick deployment and reusability.
- **Clean Design:**  
  A modern, gradient-themed interface with clear input descriptions and an informative disclaimer.


<br>

## Installation🛠

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/hk-kumawat/House-Price-Predictor.git
   cd hk-kumawat-house-price-predictor
   ```

2. **Create & Activate a Virtual Environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

5. **(Optional) Use Dev Container:**
   - Open the project in an IDE that supports Dev Containers using `.devcontainer/devcontainer.json`.

<br>


## Usage🚀

### Running the Streamlit App

Launch the prediction system with:
```bash
streamlit run app.py
```

**Features include:**
- Input various house parameters
- Get instant price predictions
  
### Running the Jupyter Notebook

Explore the data analysis and model training:
```bash
jupyter notebook "House_Price_Prediction.ipynb"
```

<br>


## Technologies Used💻

- **Programming Language:**  
  - Python

- **Machine Learning:**  
  - scikit-learn (Linear Regression)

- **Data Handling & Visualization:**  
  - Pandas, NumPy, Matplotlib, Seaborn

- **Web Framework:**  
  - Streamlit

- **Model Persistence:**  
  - pickle

<br>



## Dataset📊

The project uses the `USA_Housing.csv` dataset which includes:

- **5,000+ house records**
- **6 key features:**
  - Average Area Income
  - Average House Age
  - Average Number of Rooms
  - Average Number of Bedrooms
  - Area Population
  - House Price (target variable)

**Key Statistics:**
- **Total Records:** 5,000+
- **Features:** 6
- **Target Variable:** House Price
- **Data Format:** CSV

<br>


## Data Preprocessing🔄

1. **Data Cleaning:**
   - Removal of null values
   - Handling of non-numeric entries
   - Outlier detection and treatment

2. **Feature Engineering:**
   - Correlation analysis
   - Feature importance ranking
   - Feature scaling and normalization

3. **Data Splitting:**
   - 60% training data
   - 40% testing data 

<br>


## Model Training🧠

- **Training Process:**  
  - A linear regression model is trained on selected features from the dataset.
- **Evaluation:**  
  - Model performance is evaluated using MAE, MSE, RMSE, and R² metrics.
- **Model Persistence:**  
  - The trained model is saved as `model.pkl` and the processed DataFrame as `df.pkl` for deployment.
    
<br>


## Results🏆

- The model achieved a reasonable prediction accuracy, with error metrics calculated as follows:
  
  - **MAE**: The average absolute difference between the predicted and actual prices.
    
  - **MSE** and **RMSE**: Measures of the model's accuracy, with RMSE providing insight into the spread of errors.
    
  - **R²**: Indicates the proportion of variance explained by the model.

<br>  




## Directory Structure📁

```plaintext
hk-kumawat-house-price-predictor/
├── README.md                    # Project documentation
├── House_Price_Prediction.ipynb # Jupyter notebook for analysis
├── USA_Housing.csv             # Dataset file
├── app.py                      # Streamlit application
├── df.pkl                      # Pickled DataFrame
├── model.pkl                   # Trained model file
└── requirements.txt            # Dependencies list
```

<br>


## Future Enhancements🚀

- **Model Improvements:**  
  - Explore advanced regression techniques and ensemble methods for higher prediction accuracy.
- **Feature Engineering:**  
  - Incorporate additional features or perform more detailed feature selection.
- **UI Enhancements:**  
  - Improve the Streamlit interface with more dynamic visualizations and user-friendly design elements.
- **Data Updates:**  
  - Update the dataset with more recent housing market data for better model performance.
- **Mobile Responsiveness:**  
  - Develop a mobile-friendly version of the app.
 
<br>


## Contributing🤝

Contributions make the open source community such an amazing place to learn, inspire, and create. 🙌 Any contributions you make are greatly appreciated! 😊

Have an idea to improve this project? Go ahead and fork the repo to create a pull request, or open an issue with the tag **"enhancement"**. Don't forget to give the project a star! ⭐ Thanks again! 🙏

<br>

1. **Fork** the repository.

2. **Create** a new branch:
   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit** your changes with a descriptive message.

4. **Push** to your branch:
   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open** a Pull Request detailing your enhancements or bug fixes.

<br> 


## License📝

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

<br>


## Contact

### 📬 Get in Touch!
Feel free to reach out for collaborations or questions:

- [![GitHub](https://img.shields.io/badge/GitHub-hk--kumawat-blue?logo=github)](https://github.com/hk-kumawat) 💻 — Explore more projects by Harshal Kumawat.
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Harshal%20Kumawat-blue?logo=linkedin)](https://www.linkedin.com/in/harshal-kumawat/) 🌐 — Let's connect professionally.
- [![Email](https://img.shields.io/badge/Email-harshalkumawat100@gmail.com-blue?logo=gmail)](mailto:harshalkumawat100@gmail.com) 📧 — Reach out for inquiries or collaboration.

<br>


## Thanks for exploring—happy predicting! 🏡

> "A home is more than just a structure—it's where dreams are built. Let data help you build your future." – Anonymous

<p align="right">
  (<a href="#readme-top">back to top</a>)
</p>

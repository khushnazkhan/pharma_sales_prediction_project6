# Rossmann Pharmaceutical Sales Prediction

This project aims to predict daily sales for Rossmann pharmaceutical stores using machine learning and deep learning (LSTM) models. It includes EDA, feature engineering, model training, and a frontend interface built with Streamlit.

---

##  Project Structure

RoshmanProject/
├── app.py # Streamlit dashboard for prediction
├── requirements.txt # Project dependencies
├── model_utils.py # Model-related helpers
├── test.csv # Sample test input
└── README.md # This file

`ml_model.pkl` is not included in GitHub due to size (>100MB). It is loaded from Google Drive.

---

##  Features

-  Exploratory Data Analysis (EDA)
-  Feature Engineering on time-series & promotions
-  ML Models: XGBoost, Random Forest
-  LSTM Deep Learning model
-  Streamlit frontend to upload test file and get predictions
-  Deployed on Streamlit Cloud / Heroku

---

##  Tech Stack

- Python, Pandas, NumPy
- scikit-learn, XGBoost, TensorFlow
- Streamlit
- Git & GitHub

---

##  Challenges Faced

-  GitHub 100MB limit blocked `.pkl` upload → fixed by hosting on Google Drive  
-  Preprocessing large and complex time-series data  
- Memory handling with LSTM models  



---

## 🧾 How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/khushnazkhan/pharma_sales_prediction_project6.git
cd pharma_sales_prediction_project6
Model File
 Download ml_model.pkl from:  Google Drive Link : https://drive.google.com/file/d/1_BbIVN331nnYIJIJ8W-8F_qwJqKptnbo/view?usp=drive_link

Deployment
The app can be accessed live here

# Author
 Khushnaz Khan
 Email: khushnazkhan0786@gmail.com
 GitHub: khushnazkhan

# final deplyments after dashboard show 

![project im 2](https://github.com/user-attachments/assets/11322517-0276-4aa7-b08f-3cc2c6af5c9f)

![img2](https://github.com/user-attachments/assets/15e2ab04-af92-4645-949b-5b3ca5c6900e)

![img3](https://github.com/user-attachments/assets/398fb9c4-b25d-4ffb-a86f-22aebc159bb3)

# useing source:
 Useful Learning & Reference Links
 1. GeeksForGeeks (GFG)
Topic	Link
 Time Series Forecasting	https://www.geeksforgeeks.org/time-series-forecasting-using-python/
 XGBoost in ML	https://www.geeksforgeeks.org/xgboost-introduction-to-xgboost-algorithm/
EDA in Pandas	https://www.geeksforgeeks.org/exploratory-data-analysis-in-python/
 Train-Test Split and Evaluation	https://www.geeksforgeeks.org/ml-train-test-and-cross-validation/

 2. Analytics Vidhya
Topic	Link
 Time Series Analysis in Python	https://www.analyticsvidhya.com/blog/2021/07/a-comprehensive-guide-to-time-series-analysis/
LSTM for Time Series Forecasting	https://www.analyticsvidhya.com/blog/2021/10/lstm-for-time-series-forecasting/
 Feature Engineering	https://www.analyticsvidhya.com/blog/2020/04/what-is-feature-engineering/
 Deploy ML Models	https://www.analyticsvidhya.com/blog/2021/06/how-to-deploy-machine-learning-model-using-streamlit/

3. W3Schools
Topic	Link
Python Basics	https://www.w3schools.com/python/
Working with Files	https://www.w3schools.com/python/python_file_handling.asp
Pandas Tutorial	https://www.w3schools.com/python/pandas/default.asp Numpy Tutorial	https://www.w3schools.com/python/numpy/default.asp






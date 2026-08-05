#  Petrodollar & De-dollarization Analysis

##  Overview

This project explores the relationship between the global **Petrodollar System**, the growing trend of **de-dollarization**, and crude oil markets. It combines multiple economic and financial datasets to analyze historical trends, engineer predictive features, build a De-dollarization Index using Principal Component Analysis (PCA), and forecast Brent crude oil prices using machine learning models.

---

##  Objectives

* Analyze the evolution of the Petrodollar system.
* Study global de-dollarization trends.
* Examine the relationship between oil prices and currency movements.
* Build a De-dollarization Index using PCA.
* Forecast Brent crude oil prices using machine learning.

---

##  Dataset

**Dataset**                                                                                                                                      |
 **oil_production_trade.csv** 
Historical oil production, exports, imports, consumption, and global trade statistics.                                             
 **opec_quotas_events.csv**    
OPEC production quotas and significant geopolitical or economic events affecting oil markets.                                      
 **recycling_swf.csv**      

Petrodollar recycling, sovereign wealth fund investments, and oil-export revenue flows.                                            
 **dedollarization.csv**    
 Indicators related to reserve currency composition, BRICS initiatives, bilateral trade settlements, and de-dollarization.          
 **daily_prices_fx.csv**      
Daily Brent and WTI crude oil prices, exchange rates, and the US Dollar Index (DXY) used for time-series analysis and forecasting. 

---

##  Project Workflow

1. Data Collection and Integration
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Principal Component Analysis (PCA)
6. Machine Learning Model Development
7. Model Evaluation and Prediction

---

## Exploratory Data Analysis

The analysis includes:

* Brent and WTI crude oil price trends
* US Dollar Index (DXY) analysis
* Oil production and consumption trends
* OPEC production analysis
* Petroyuan trading trends
* Reserve currency composition
* Correlation analysis
* Time-series visualization

---

## ⚙️ Feature Engineering

The following features were created to improve model performance:

* Lag Features
* Rolling Mean
* Rolling Standard Deviation
* Exponential Moving Average (EMA)
* Momentum Indicators
* Percentage Change
* Revenue per Barrel

---

## De-dollarization Index

A composite **De-dollarization Index** was developed using **Principal Component Analysis (PCA)** by combining multiple economic indicators into a single measure representing the progress of de-dollarization.

---

##  Machine Learning Models

The following models were implemented for forecasting Brent crude oil prices:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* **Naive Model (Baseline Forecast using last observed value)**
* **Trend Extrapolation Model (Linear trend-based forecasting)**
* **Random Forest Regressor (Enhanced tuned version for non-linear relationships)**

---

##  Model Evaluation

All models were evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

The inclusion of baseline and trend-based models provides a strong benchmark for comparing machine learning performance against simple forecasting approaches.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* Jupyter Notebook



---

##  Future Improvements

* Integrate live financial and commodity market APIs.
* Include geopolitical risk indicators.
* Add news sentiment analysis using NLP.
* Develop LSTM and Transformer-based forecasting models.
* Build an interactive Streamlit application.
* Create a Power BI dashboard for business users.
* Automate periodic data refresh and model retraining.

#

---

##  Skills Demonstrated

* Data Cleaning & Preprocessing
* Data Integration
* Exploratory Data Analysis
* Time-Series Analysis
* Feature Engineering
* Principal Component Analysis (PCA)
* Machine Learning
* Predictive Analytics
* Financial Data Analysis
* Data Visualization

---

##  Author

**Francis Infant**



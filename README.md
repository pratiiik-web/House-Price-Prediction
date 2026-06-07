# House Price Prediction

Predict house prices based on physical and contextual property features
using regression models.

## Problem
Buyers and sellers struggle to estimate fair house prices without expert
knowledge. This model predicts price from features like area, bedrooms,
furnishing status, and location preferences.

## Dataset
Housing Prices Dataset — 545 houses, 13 features  
Source: Kaggle (yasserh/housing-prices-dataset)  
Price range: ₹17.5L to ₹1.33Cr

## Key Findings
- Area and bathrooms are the strongest price predictors (correlation 0.54, 0.52)
- Furnished homes cost significantly more than unfurnished ones
- Bathrooms correlate with price almost as strongly as area — unexpected insight
- 5-bedroom houses command premium prices; 6-bedroom drops (older large houses)

## The Feature Engineering Breakthrough

| Model | R² | RMSE |
|---|---|---|
| Linear Regression (baseline) | 0.649 | ₹13.3L |
| Decision Tree | 0.416 | ₹17.2L |
| Random Forest | 0.613 | ₹14.0L |
| **Linear Regression + features** | **0.883** | **₹7.7L** |

Feature engineering improved R² from 0.649 to 0.883 — a bigger gain
than switching models. Three new columns were created:
- `price_per_sqft` — value density of the property
- `bath_per_bed` — luxury ratio
- `total_rooms` — overall size signal

## How to Run
pip install -r requirements.txt
streamlit run apps/app.py

## Tech Stack
Python · Pandas · Seaborn · Scikit-learn · Streamlit

## Project Structure
House Price Prediction/
├── Data/
│   └── Housing.csv
├── Models/
│   ├── model.pkl
│   └── scaler.pkl
├── Notebooks/
│   └── 01_eda.ipynb
└── apps/
    └── app.py

## Key Lesson
Feature engineering matters more than model choice.
A simple Linear Regression with engineered features outperformed
Random Forest on raw data by 27% in R² score.

## Live Demo
https://house-price-prediction-aaz3ta8edkwz2krzdpu7wn.streamlit.app/


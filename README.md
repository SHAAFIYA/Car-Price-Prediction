CAR PRICE PREDICTION

OVERVIEW

This project aims to predict used car prices using machine learning techniques. It analyzes various factors affecting car pricing, such as year, manufacturer, model, fuel type, odometer reading, and title status to generate accurate price predictions.

DATASET

•	Source: OASIS INFOBYTE (8,200 car listings)

•	Features Used: Year, Manufacturer, Model, Fuel Type, Odometer, Title Status, and Price (Target Variable)

•	Preprocessing Steps:

o	Removed non-car listings and unnecessary features

o	Filtered out zero-valued prices

o	Handled missing values using mean imputation

o	Applied outlier detection and label encoding for categorical variables

MACHINE LEARNING MODELS USED

1.	Random Forest Regressor (Best Performing Model)

2.	Gradient Boosting Regressor

3.	Extra Trees Regressor

MODEL EVALUATION METRICS

•	R-Squared (R²): Measures model accuracy (best model achieved 0.8341)

•	Mean Squared Error (MSE): Measures overall error magnitude

•	Mean Absolute Error (MAE): Evaluates absolute error magnitude

KEY FINDINGS

1. Model Accuracy:

   •	Random Forest Regressor achieved the highest R² score (0.8341), making it the most accurate model.

2. Data Preprocessing Impact:
   
   •	Feature selection, data cleaning, and preprocessing significantly improved prediction accuracy.

3. Model Robustness:
   
   •	The model performed well across various data distributions, demonstrating its robustness in price prediction.

4. Generalization Capability:
   
   •	Ensemble methods like Random Forest and Extra Trees provided better generalization compared to single regression models.

5. Exploratory Data Analysis (EDA) Insights:
    
   •	Exploratory Data Analysis (EDA) revealed key insights into manufacturer trends and fuel type distributions.

6. Market Alignment:
    
   •	The pricing predictions aligned well with market trends, demonstrating the model's practical applicability.

7. Performance Optimization:

   •	Hyperparameter tuning further optimized model performance, reducing errors and improving accuracy.



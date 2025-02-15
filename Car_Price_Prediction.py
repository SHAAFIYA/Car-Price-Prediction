# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("car.csv")

# Exploratory Data Analysis
print(data.head())
print(data.describe())

# Fuel Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, y='fuel', palette='viridis')
plt.title('Fuel Distribution')
plt.xlabel('Count')
plt.ylabel('Fuel')
plt.show()

# Manufacturer Distribution
plt.figure(figsize=(40, 30))
sns.countplot(data=data, x='manufacturer', palette='viridis')
plt.title('Bar Plot: Manufacturer Distribution')
plt.ylabel('Count')
plt.xticks(rotation=70, fontsize=6)
plt.show()

# Correlation Analysis
correlation_matrix = data.corr()
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True)
plt.show()

# Data Preprocessing
# Removing non-car listings & unnecessary features
data['type'].unique() 
not_cars = data[data['type'].isin(['other', 'van', 'bus'])] 
data.drop(not_cars.index, inplace=True) 
drop_col = ['lat', 'long'] 
data = data.drop(columns=drop_col)

# Filtering zero-valued prices
zero_price_cars = data[data['price'] == 0] 
count_zero_price_cars = zero_price_cars.shape[0] 
print("Number of cars listed with a price of '0':", count_zero_price_cars) 
data = data[data['price'] > 0]

# Handling Missing Values
missing_values = data.isna() 
num_missing_values = missing_values.sum() 
print("Number of missing values in each column:") 
print(num_missing_values)

null_threshold = 1500
columns_to_remove = data.columns[data.isnull().sum() > null_threshold] 
data = data.drop(columns=columns_to_remove)
numerical_cols = data.select_dtypes(include=[np.number]).columns 
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# Outlier Detection & Handling
plt.figure(figsize=(12, 8)) 
for i, column in enumerate(numerical_cols):
    plt.subplot(2, 2, i + 1)  
    sns.boxplot(x=data[column], color='skyblue')
    plt.title(f'Boxplot for {column} (Outliers)')
    plt.xlabel(column) 
plt.tight_layout() 
plt.show()


Q1 = data.quantile(0.25) 
Q3 = data.quantile(0.75) 
IQR = Q3 ‐ Q1 
 
def handle_outliers(column):
    lower_bound = Q1[column] ‐ 1.5 * IQR[column]     
    upper_bound = Q3[column] + 1.5 * IQR[column] 
    data[column] = data[column].apply(lambda x: max(lower_bound, min(upper_bound, x))) 
for column in numerical_cols:
    handle_outliers(column) 
 
plt.figure(figsize=(12, 8)) 
for i, column in enumerate(numerical_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x=data[column], color='skyblue') 
    plt.title(f'Boxplot for {column} (After Handling Outliers)')     
    plt.xlabel(column) 
plt.tight_layout() 
plt.show()

# Label Encoding for categorical features
categorical_cols = data.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
for column in categorical_cols:
    data[column] = le.fit_transform(data[column])

# Splitting Data
selected_features = ['year', 'manufacturer', 'model', 'fuel', 'odometer', 'title_status']
X = data[selected_features]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Model Training
rf = RandomForestRegressor(random_state=32)
gb = GradientBoostingRegressor(random_state=32)
et = ExtraTreesRegressor(random_state=32)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
et.fit(X_train, y_train)

# Prediction
rf_y_pred = rf.predict(X_test)
gb_y_pred = gb.predict(X_test)
et_y_pred = et.predict(X_test)

# Model Evaluation
# Random Forest Regressor
print("r2_score of Random Forest Regressor:", r2_score(y_test, rf_y_pred))
mse = mean_squared_error(y_test, rf_y_pred) 
print("Mean Squared Error (MSE) of Random Forest Regressor: ",mse) 
mae = mean_absolute_error(y_test, rf_y_pred) 
print("Mean Absolute Error (MAE)of Random Forest Regressor: ",mae)

# Gradient Boosting Regressor
print("r2_score of Gradient Boosting Regressor :", r2_score(y_test, gb_y_pred))
mse = mean_squared_error(y_test, gb_y_pred) 
print("Mean Squared Error (MSE) of Gradient Boosting Regressor: ",mse) 
mae = mean_absolute_error(y_test, gb_y_pred) 
print("Mean Absolute Error (MAE) of Gradient Boosting Regressor: ",mae)

# Extra Trees Regressor
print("r2_score of Extra Trees Regressor:", r2_score(y_test, et_y_pred)) 
mse = mean_squared_error(y_test, et_y_pred) 
print("Mean Squared Error (MSE) of Extra Trees Regressor: ",mse) 
mae = mean_absolute_error(y_test, et_y_pred) 
print("Mean Absolute Error (MAE)of Extra Trees Regressor: ",mae

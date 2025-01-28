#Dataset used on https://www.kaggle.com/datasets/shubham47/salary-data-dataset-for-linear-regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data loading
file_path = 'Salary_Data.csv'  
data = pd.read_csv(file_path)

# Descriptive data analysis 
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset summary:")
print(data.info())

print("\nDescriptive statistics:")
print(data.describe())

# Missing data analysis 
print("\nMissing data analysis:")
print(data.isnull().sum())

# Correlation graph 
print("\nCorrelation analysis:")
correlation_matrix = data.corr()
print(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Graph")
plt.show()


# Linear Regression Model

X = data.iloc[:, :-1]  # All columns except salary
y = data.iloc[:, -1]   # Salary column

# Splitting into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Visualizing actual vs predicted salaries
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Salaries")
plt.ylabel("Predicted Salaries")
plt.title("Actual vs Predicted Salaries")
plt.show()

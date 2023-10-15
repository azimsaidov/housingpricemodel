import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('housing.csv') #You enter any csv of the data set you want to train with

X = data.drop('price', axis=1)  # Features
y = data['price']              # Target variable

model = LinearRegression()
model.fit(X, y)

# Get user input for feature values
input_values = []
for feature in X.columns:
    value = input(f"Enter value for {feature}: ")
    input_values.append(value)

# Create a new DataFrame with the user input values
user_input = pd.DataFrame([input_values], columns=X.columns)

# Use the model to predict the price based on the user input
predicted_price = model.predict(user_input)

print(f"Predicted price: {predicted_price[0]}")

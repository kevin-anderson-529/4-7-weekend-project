# https://www.kaggle.com/datasets/aadhavvignesh/calories-burned-during-exercise-and-activities

#import bcrypt
#password = ''
#hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
#print(hashed_password)

#df = pd.read_csv(r'C:\Users\kevin\OneDrive\Desktop\Coding Temple\Week 5 (SQL)\sql project\exercise_dataset.csv')

# Clean data by dropping rows with missing values
#df.dropna(inplace=True)

#engine = create_engine('postgresql://postgres@localhost/exercise_dataset')
#df.to_sql('exercise_dataset', engine, if_exists='replace')

'''Use calories per kg as the predictor variable 
to try and predict the number of calories burned per kg of body weight at each of the four weight categories: 130, 155, 180, 205'''

import pandas as pd
import psycopg2
from sklearn.linear_model import LinearRegression

# Create a connection to the database
conn = psycopg2.connect(database="exercise_dataset", user="postgres", password="password", host="localhost", port="5432")

# Read the data from the database using a SQL query
data = pd.read_sql_query('SELECT * FROM exercise_dataset', conn)

# Close the connection
conn.close()

# Split columns based on what's being predicted
X = data[['Calories per kg']]  # select the predictor variable
y = data[['130 lb', '155 lb', '180 lb', '205 lb']]  # select the target variables

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Calculate the R-squared score for the model
score = model.score(X, y)
print('R-squared score:', score)

# R-squared score: 0.9999991572104665 - The high score indicates that the linear model fits the data well.

# Input data to see predictions
input_data = [[1.0], [1.5], [2.0]]  # Calories per kg

# Make predictions using the model
predictions = model.predict(input_data)

# Print the predictions
print("Predictions for calories burned at each weight category:")
print("130 lb:", predictions[:, 0])
print("155 lb:", predictions[:, 1])
print("180 lb:", predictions[:, 2])
print("205 lb:", predictions[:, 3])

#Predictions for calories burned at each weight category:
#130 lb: [286.66436523 429.87240525 573.08044527]
#155 lb: [341.6720982  512.56297231 683.45384642]
#180 lb: [396.83264742 595.27809118 793.72353493]
#205 lb: [451.92925067 677.93407348 903.93889629]

'''This shows that each weight is going to burn a different number of calories per hour 

For 1.0 Calories per kg:

130 lb: 286.66 calories
155 lb: 341.67 calories
180 lb: 396.83 calories
205 lb: 451.93 calories

For 2.0. Calories per kg:

130 lb: 429.87 calories
155 lb: 512.56 calories
180 lb: 595.28 calories
205 lb: 677.93 calories

For 3.0 Calories per kg:

130 lb: 573.08 calories
155 lb: 683.45 calories
180 lb: 793.72 calories
205 lb: 903.94 calories'''
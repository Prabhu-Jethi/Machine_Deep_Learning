import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


####### Collect
df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\CSV\advertising.csv")
print("\nTotal DataFrame:\n", df)

#### Cleaning data
print("\nNull counts:\n", df.isnull().sum())


##### Encoding (Assigns unique integers to each columns also converts string or categorical type to numeric)

### Here In this case we are using complete numerical data so we don't require any encoding


#### Spliting -----> [Removing the target column]
X = df.drop('Sales', axis=1)
y = df['Sales']
print("\nIndependent_variables:\n", X)
print("\nTarget_column:\n", y)


######### Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\nX-Train data:\n", X_train)
print("\nX-Test data:\n", X_test)
print("\nY-Train data:\n", y_train)
print("\nY-Test data:\n", y_test)


########## Feature Scaling or Standard Scaling (Transform data so that every feature has mean = 0, standard division = 1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("\nScaled X_train data:\n", X_train)
print("\nScaled X_test data:\n", X_test)


########## Model training -----> (Here Linear regression model is used as linear model supports continuous data in its target column)
## CV provides best suitable alpha value according to the dataset, to avoid making coefficients zero and tuning hyperparameters perfectly
model = RidgeCV(cv=5)
model.fit(X_train, y_train)
print("\nRidge_Regression:\n", model)
print("\nAlpha Value:\n", model.alpha_)
print("\nRidge :", model.coef_)

######### Predict
ypred = model.predict(X_test)
print("\nPrediction of Test data:\n", ypred)


######## Evaluation of Metrics like ----> MSE, MAE, r2 score and RMSE
mse = mean_squared_error(y_test, ypred)
mae = mean_absolute_error(y_test, ypred)
r2 = r2_score(y_test, ypred)
rmse = root_mean_squared_error(y_test, ypred)
print("\nMean Squarred Error:\n", mse)
print("\nMean Absolute Error:\n", mae)
print("\nR2 Score:\n", r2)
print("\nRoot Mean Squarred Error:\n", rmse)
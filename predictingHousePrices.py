from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

# https://towardsdatascience.com/keras-101-a-simple-and-interpretable-neural-network-model-for-house-pricing-regression-31b1a77f05ae

print("helloML101")

boston_dataset = load_boston()
df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
df.head(n=10)
# - CRIM per capita crime rate by town
# - ZN proportion of residential land zoned for lots over 25,000 sq.ft.
# - INDUS proportion of non-retail business acres per town
# - CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# - NOX nitric oxides concentration (parts per 10 million)
# - RM average number of rooms per dwelling
# - AGE proportion of owner-occupied units built prior to 1940
# - DIS weighted distances to five Boston employment centres
# - RAD index of accessibility to radial highways
# - TAX full-value property-tax rate per $10,000
# - PTRATIO pupil-teacher ratio by town
# - B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# - LSTAT % lower status of the population
# - MEDV Median value of owner-occupied homes in $1000's

target_var = 'MEDV'
df[target_var] = boston_dataset.target


var = 'RM'
df.sort_values(by=var, inplace=True)
plt.scatter(df[var], df[target_var])
plt.xlabel(var)
plt.ylabel("price")

# f(x) = a + b*x
coef = np.polyfit(df[var], df[target_var], 1)
poly1d_fn = np.poly1d(coef)

plt.plot(df[var], poly1d_fn(df[var]), '--k')

var = 'CRIM'
df.sort_values(by=var, inplace=True)
plt.scatter(df[var], df[target_var])
plt.xlabel(var)
plt.ylabel("price")


# Data Preprocessing #

# target
Y = df[target_var].values
# input
X = df.drop(target_var, 1).values


model = keras.Sequential()
model.add(Dense(32, input_shape=(13, ), activation='relu', name='dense_1'))
model.add(Dense(16, activation='relu', name='dense_2'))
model.add(Dense(1, activation='linear', name='dense_output'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

model.fit(x=X, y=Y, batch_size=16, epochs=10, validation_split=0.7)

Y[0]
X[0]

df_compare = pd.DataFrame(Y, columns=['truth'])
df_compare['predict'] = model.predict(X)

plt.scatter(df_compare.index[:20], df_compare['truth'][:20])
plt.scatter(df_compare.index[:20], df_compare['truth'][:20])


# normalisation #
normalisation_min_value = df.min()
normalisation_scalar = (df.max()-df.min())
df=(df-normalisation_min_value)/normalisation_scalar

original_target = Y * normalisation_scalar[target_var] + normalisation_min_value[target_var]

# target
Y = df[target_var].values
# input
X = df.drop(target_var, 1).values

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x=X[20:], y=Y[20:], batch_size=16, epochs=10)

test_prediction = model.predict(X[:20])

test_prediction = test_prediction * normalisation_scalar[target_var] + normalisation_min_value[target_var]

plt.scatter(np.arange(0, 20), df[target_var][:20])
plt.scatter(np.arange(0, 20), test_prediction)

# TODO: remove outlieres, handle missing data/null values, encoding of categorical variables,
#       normalization, model architecture and size, remove/add data

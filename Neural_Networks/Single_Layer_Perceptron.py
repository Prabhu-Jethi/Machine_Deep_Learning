import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


x = np.array([3,6,8,5,9,3,4])
y = np.array([9,7,3,1,2,6,5])

plt.scatter(x,y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()

# SLP
model = Sequential()

# input 
model.add(Dense(1, input_dim=1, activation='linear'))

# output
model.add(Dense(1, activation='linear'))

# compiler ADAM
optimizer = Adam(learning_rate=0.1)

model.compile(
    optimizer=optimizer,
    loss='mse',
)


model.summary()

fit = model.fit(
    x,
    y, 
    batch_size=16,
    epochs=200, 
    verbose=1
)
print(fit)

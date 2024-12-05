import pandas as pd
import random
import keras
from keras.layers import Dense
from keras import Sequential
from keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
print(style.available)


def create_data(num=10):
    df = pd.DataFrame(columns=['a', 'b', 'sum'])
    for i in range(num):
        a = random.randrange(0, 100, 1)
        b = random.randrange(0, 100, 1)
        c = 3*a+2*b + 5
        df.loc[i] = [a, b, c]
    return df


num = 10000
filename = 'data.csv'
if os.path.exists(filename):
    data = pd.read_csv(filename)
else :
    data = create_data(num=num)
    data.to_csv(filename, index=False)
print(data.head())
train, test = data[:int(0.7*num)].values, data[int(0.7*num):].values

model = Sequential()
model.add(Dense(1, input_shape=(2,)))

opt=Adam(lr=0.001, decay=1e-5)
model.compile(optimizer=opt, loss='mse', metrics=['mse', 'acc'])

print(train[:, :2].shape)
history = model.fit(train[:, :2], train[:, 2:],
                    epochs=100,
                    validation_data=(test[:, :2], test[:, 2:]))

plt.plot(history.history['loss'])
plt.show()

plt.clf()
print(np.vstack(model.get_weights()))
weights = np.vstack(model.get_weights()).flatten()
print(weights)
print(np.arange(len(weights)))
plt.barh(np.arange(len(weights)), weights, align='center')
plt.xlabel('Value')
plt.ylabel('Neuron')
plt.yticks(np.arange(len(weights)))
# plt.legend()
# show window but continue code
plt.draw()
a = 150
b= 224
print(f"3*{a} + 2*{b} =", model.predict(np.array([[a, b],]))[0][0])
print(f"3*{a} + 2*{b} =", 3*a + 2*b)

# prevent window from being closed
plt.show()

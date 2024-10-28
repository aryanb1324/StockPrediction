# Stock Prediction App
(WIP)

Created this project in order to learn more about RNNs and GRUs

Here is an explanation about the project and the _science_ behind it!



## What are GRUs?
**Gated Recurrent Units** or GRUs (NOT THE ONE FROM MINIONS) are a type of recurrent neural network archietcture that are designed to process and predict sequences of data (like stocks. GRUs were introuduced to improve traditional RNNs and they address issues with RNNs like vanishing gradient.

### wait, what is the vanishing gradient problem?
**Vanishing gradient** is a really big issue that occurs when creating machine learning models. when you try and decrease error on a model, you can model it as a graph. the x value is some 'parameter' and the y value is the error. The issue is that the values or the 'slope' between the error decreases as learning goes on. Basically the gradients (or the changes to weights) become veyr small during training adn the model's don't update properly which leads to slow learning. vanishing gradient usually happens duirng backpropagation adn the usage of activation functions. but thats not really that important tbh, just know that GRUs solve for this

## How GRUs work
grus introduce two types of gates: **update gates and reset gates**
1. update gates decide hwo much of the past inofo the keep
2. reset gates decide how much of past info to forget.
<!-- end of the list -->
these gates help keep new information in the loop and discard non important info while reducing the issue from the vanishing gradient problem.
each gru cell has its own components, heres a quick description of them.
1.** hidden states**: the current state of the sequence
2. **update gate**: decides how much of previous state to carry forward
3. **reset gate**: controls how much to forget
4. **candidate hidden state** and **final hidden state** are both pretty difficult to explain _(google for more info about these)_
<!-- end of the list -->

## so how are grus and lstms diff?
**LSTMs or long short-term memory** have their own fuctions and differ, though they both try and solve the issue with regular rnns
GRUs:
1. Simpler architecture with two gates (update and reset).
2. Fewer parameters, making them faster to train.
3. Often perform well on simpler sequences.
<!-- end of the list -->
LSTMs:
1. More complex architecture with three gates (input, forget, and output).
2. Tend to perform better on more complex sequences but are slower to train.
<!-- end of the list -->
Considering that this is just 'number' prediction, we use grus (also cuz idrk how they work and i want to learn lol)
BUT, grus also have their own benefits
Grus can...
1. reatain patterns from past stock prices and discard uninportant info
2. adapt to changes because of their simple architecture
3. handle long sequences without vanishing gradient issues
<!-- end of the list -->

in a coding context, grus look like this:
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(GRU(units=50, return_sequences=False))
model.add(Dense(units=1))
```
heres what each piece of code means
```
units = 50
```
this says how many gru neurans in each layer, it determines the capacity of the layer to capture patterns
```
return_sequences=True
```
what this does is it determines if we should provide the enture output to the next layer, we choose false when it is our last _output_ layer. 
```
model.add(...)
```
can't forget the method, model.add adds layers
and finally, the output.
```
model.add(Dense(units=1))
```
this code outputs the final value, since we want just one stock price, we only output one item, hence units = 1

NOW lets get started with the actual project!

## data collection and preprocessing
import needed libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
```
1. Numpy and Pandas: Used for data manipulation.
2. Matplotlib: For data visualization.
3. MinMaxScaler: Scales data to a specific range, improving model convergence.
4. GRU: A type of neural network layer suited for sequential data.
5. RandomizedSearchCV: Helps in tuning model hyperparameters. (not really implemented lol)
<!-- end of the list -->
```
data = yf.download('AAPL', start='2010-01-01', end='2024-10-25')
```
gets stock info from yahoo finance
```
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])
```
scales down stock pricess to 0 to 1
```
def create_sequences(data, seq_length=60):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Split into training and testing data
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
x_train, y_train = create_sequences(train_data)
x_test, y_test = create_sequences(test_data)
```
creates sequences so the grus can understandpatterns
x: Contains sequences of 60 days of past stock prices.
y: Contains the actual next day's stock price.
```
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
```
reshape
## build gru model
```
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(GRU(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
```
define model thru keras
```
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2)
```
compile and train model and we r done!

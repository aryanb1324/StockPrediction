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

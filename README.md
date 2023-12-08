# ANN-Binary-Classification

I'm randomly generating data that can be linearly separated into class1 (0) and class2 (1). 
 A simple ANN model with 2 neurons in the input layer (1 neuron for the x-coordinate and 1 for the y-coordinate), ReLU as the activation function for the input layer, and 1 neuron in the output layer with Sigmoid as the activation function is trained.
 There are no hidden layers. Binary Cross-Entropy (BCE) is used as a loss function. 

## Meta Parametric Experiment
The Accuracy is plotted as a function of learning rate. The experiment is repeated 50 times to tackle randomness in results caused by random weights and bias initialization. mean of 50 accuracy values respective to a particular learning rate in the range (0.0001 - 0.1) is plotted, and it can be deduced that accuracy is directly proportional to the learning rate.   

![meta parametric experiment result](https://github.com/nishit3/ANN-Binary-Classification/assets/90385616/058ac05c-98e3-43d4-9cf9-856d4941dfae)

## Good/Bad Wine Quality Classification on Red Wine Dataset by UCI

The model predicts wine quality (0=bad, 1=good). I have compared the performance of model on variable batch sizes. It can be inferred that at batchsize=4 model is inconsistent but also throws relatively high accuracies, other batch size performance is pretty stable around 75%.

![Figure_1](https://github.com/nishit3/ANN-Binary-Classification/assets/90385616/0eba27ae-2e3c-4d52-82b4-f3781b7901f7)

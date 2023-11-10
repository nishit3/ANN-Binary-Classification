# ANN-Binary-Classification

I'm randomly generating data that can be linearly separated into class1 (0) and class2 (1). 
 A simple ANN model with 2 neurons in the input layer (1 neuron for the x-coordinate and 1 for the y-coordinate), ReLU as the activation function for the input layer, and 1 neuron in the output layer with Sigmoid as the activation function is trained.
 There are no hidden layers. Binary Cross-Entropy (BCE) is used as a loss function. 

## Meta Parametric Experiment
The Accuracy is plotted as a function of learning rate. The experiment is repeated 50 times to tackle randomness in results caused by random weight and bias initialization. mean of 50 accuracy values respective to a particular learning rate in the range (0.0001 - 0.1) is plotted, and it can be deduced that accuracy is directly proportional to the learning rate.   


![Alt text](/meta parametric experiment result.jpeg?raw=true "Optional Title")

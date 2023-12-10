import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy

data = pd.read_csv('winequality-red.csv', sep=';')
data = data[data['total sulfur dioxide'] < 200]
data['isGood'] = (data['quality'] > 5)

data = data.drop(columns=["quality"])

indpndt_vars = data.keys()
indpndt_vars = indpndt_vars.drop('isGood')

data[indpndt_vars] = data[indpndt_vars].apply(scipy.stats.zscore)

feature_matrix = data.iloc[:, :-1].values
target = data.iloc[:, -1].values
target = np.reshape(target, (-1, 1))


train_X, test_X, train_y, test_y = train_test_split(feature_matrix, target, train_size=0.8)
train_X = torch.tensor(train_X).float()
test_X = torch.tensor(test_X).float()
train_y = torch.tensor(train_y).float()
test_y = torch.tensor(test_y).float()


train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
activation_functions = [nn.ReLU(), nn.LeakyReLU(), nn.ReLU6()]
af_names = ['ReLU', 'LeakyReLU', 'ReLU6']
epochs = np.linspace(1, 1000, num=1000)
results = list(np.zeros((3, len(epochs))))
loss_func = nn.BCEWithLogitsLoss()
train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=len(torch.detach(test_X)))


for af_i, ActivationFunction in enumerate(activation_functions):
    classifier = nn.Sequential(
        nn.Linear(len(indpndt_vars), 16),
        ActivationFunction,
        nn.Linear(16, 32),
        ActivationFunction,
        nn.Linear(32, 32),
        ActivationFunction,
        nn.Linear(32, 1),
        # nn.Sigmoid()
    )

    optimizer = torch.optim.SGD(classifier.parameters(), lr=.01)

    for epoch_i, epoch in enumerate(epochs):
        for X, y in train_loader:
            predictions = classifier(X)
            loss = loss_func(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_X, test_y = next(iter(test_loader))
        pred = classifier(test_X)
        test_loss = loss_func(pred, test_y)
        results[af_i][epoch_i] = torch.mean(((pred > 0) == test_y).float()).item()*100.00

plt.title("Activation Function's Performance")
for batch_i in range(3):
    plt.plot(epochs, results[batch_i], label=str(af_names[batch_i]))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.legend()
plt.show()

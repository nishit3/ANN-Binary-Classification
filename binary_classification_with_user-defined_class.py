import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import sys

# Creating data
total_elements_per_class = 100

starting_xandy_coordinates_class1 = [1, 1]
starting_xandy_coordinates_class2 = [5, 1]

class1 = [starting_xandy_coordinates_class1[0] + np.random.randn(total_elements_per_class),
          starting_xandy_coordinates_class1[1] + np.random.randn(total_elements_per_class)]
class2 = [starting_xandy_coordinates_class2[0] + np.random.randn(total_elements_per_class),
          starting_xandy_coordinates_class2[1] + np.random.randn(total_elements_per_class)]

data_np = np.hstack((class1, class2)).T
labels_np = np.vstack((np.zeros((total_elements_per_class, 1)), np.ones((total_elements_per_class, 1))))

data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

print(np.shape(data))
print(np.shape(labels))


# training model
class BinaryClassifierAnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 1)
        self.output = nn.Linear(1, 1)

    def forward(self, x):
        x = self.input(x)
        x = f.relu(x)
        x = self.output(x)
        return f.sigmoid(x)


ANN_classifier = BinaryClassifierAnn()
lossFun = nn.BCELoss()
optimizer = torch.optim.SGD(ANN_classifier.parameters(), 0.01)
epochs = 1500

for epoch in range(epochs):
    class_predictions = ANN_classifier(data)
    loss = lossFun(class_predictions, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

outputs_for_data = ANN_classifier(data)
predicted_classes = (
            outputs_for_data > 0.5)  # >0.5 output is classified as class2, <=0.5 output is classified as class 1

# evaluating model
misclassified_indices = np.where(predicted_classes != labels)[0]
total_misclassified_elements = len(misclassified_indices)
percent_misclassified = total_misclassified_elements / len(labels) * 100
accuracy = 100 - percent_misclassified

sys.stdout.write("\n\nAccuracy = ")
sys.stdout.write(str(accuracy))
sys.stdout.write(" %")

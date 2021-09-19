# PyTorch implementation
import os

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

#read csv to dataframe
training_dataset = pd.read_csv("Data/crx-data.csv", header = None)
#remove column with names (letters)
training_dataset = training_dataset.drop(training_dataset.columns[0],axis=1).values.tolist()
print(training_dataset)
#convert cp to 0, im to 1
for a in training_dataset:
    if a[7]=="cp":
        a[7]=0
    elif a[7]=="im":
        a[7]=1

#print(training_dataset)
X = torch.Tensor([i[0:6] for i in training_dataset])
Y = torch.Tensor([i[7] for i in training_dataset])

# Class for the network
class Net(nn.Module):
    def __init__(self): # constructor, defines the different layers and how many neurons are in each
        super(Net,self).__init__()
        self.fc1 = nn.Linear(6,3)
        self.fc2 = nn.Linear(3,1)

    # propagates value x through network to give result
    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
# creates a model
model = Net()
print(model)

# measure error rate
criterion = nn.MSELoss()
# define optimization method
optimizer = optim.SGD(model.parameters(), lr=0.001)
# array to hold erroneous predictions
allloss = []

# repeat predictions 100x
for epoch in range(100):
    print(epoch)
    outputs = model(X)
    loss = criterion(outputs,Y)
    loss.backward()
    optimizer.step()
    allloss.append(loss.item())

import pdb;pdb.set_trace()


import matplotlib.pyplot as plt
plt.plot(allloss)
plt.show()

print(list(model.parameters()))



# Native python implementation
import math

# weights for the network; 7 inputs, grouping in 3 based on descriptions
# 7 inputs, 3 biases
#   group 1:
#   0: bias for 1st neuron
#   1: mcg: McGeoch's method for signal sequence recognition.
#   2.  gvh: von Heijne's method for signal sequence recognition.
#   group 2:
#   3. bias for 2nd neuron
#   4.  lip: von Heijne's Signal Peptidase II consensus sequence score.
#            Binary attribute.
#   5.  chg: Presence of charge on N-terminus of predicted lipoproteins.
# 	   Binary attribute.
#   6.  aac: score of discriminant analysis of the amino acid content of
# 	   outer membrane and periplasmic proteins.
#   group 3:
#   7. bias for 3rd neuron
#   8. alm1: score of the ALOM membrane spanning region prediction program.
#   9. alm2: score of ALOM program after excluding putative cleavable signal
# 	   regions from the sequence.
#   10. bias for 2nd layer neuron
#   11. weight for 1st neuron in second layer
#   12. weight for 2nd neuron in second layer
#   13. weight for 3rd neuron in second layer
weights = [0.3299,  0.2785,  0.0619 ,-0.1197 ,0.3677, -0.1072,  0.1614 ,0.1732, -0.3240, -0.3326 ,0.31 ,0.14 ,0.167 , -0.3815]

# sigmoid function. Since output value is 0 or 1, we simply return Z

def sigmoid(z):
    if z>=1:
        return 1
    else:
        return 0

# computing output of first layer
def firstLayer(row, weights):
    # first neuron and its bias
    activation_1 = weights[0]*1
    activation_1 += weights[1]*row[0]
    activation_1 += weights[2]*row[1]
    # second neuron and its bias
    activation_2 = weights[3]*1
    activation_2 += weights[4]*row[2]
    activation_2 += weights[5]*row[3]
    activation_2 += weights[6] * row[4]
    # third neuron and its bias
    activation_3 = weights[7]*1
    activation_3 += weights[8] * row[5]
    activation_3 += weights[9] * row[6]

    #return sigmoid result from the three activation functions:
    return sigmoid(activation_1),sigmoid(activation_2),sigmoid(activation_3)

# computes output from second layer
def secondLayer(row,weights):
    activation_4 = weights[10] * 1
    activation_4 += weights[11] * row[0]
    activation_4 += weights[12] * row[1]
    activation_4 += weights[13] * row[2]
    return sigmoid(activation_4)

# computes predictions
def predict(row,weights):
    input_layer = row
    first_layer = firstLayer(input_layer,weights)
    second_layer = secondLayer(first_layer,weights)
    return second_layer,first_layer

#print predictions and real values of y
correct = 0
trials = 0
for d in training_dataset:
    print(predict(d,weights)[0],d[-1])   # Prints y_hat and y
    if (predict(d,weights)[0] == d[-1]):
        correct+=1
    trials+=1

accuracy = correct/trials

print("Accuracy: "+str(accuracy))
# PyTorch implementation
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math

### Data import and handling
## loop through each folder, extract files into array
## loop through with integer iterator, check if file with that id exists, if it does, paste as array to array
# each file is a bunch of rows, put second value as array elements per file
## normal first
def loadHeartbeatData(folder):
    array = []
    filepath = "Data/ecg/"+folder+"/801_"
    for it in range(0, 100000): # max number is 6 digits, so 999999. This loop goes up to that.
        heartbeat = [] # array to hold a given heartbeat
        it_as_string = str(it) # gets the current index as a string
        if (len(it_as_string)<6): # if the current index is not 6 digits, add a leading 0.
            while(len(it_as_string)<6):
                it_as_string = "0"+it_as_string
        for fileExt in [".0",".1"]: # open both .0 and .1 files for each heartbeat
            fileArray = []  # array to hold a given file (.0 or .1)
            if os.path.isfile(filepath+it_as_string+str(fileExt)): # if file exists, open it
                file = open(filepath+it_as_string+str(fileExt))
                for line_in_file in range(0,39): # read 39 lines (all files have at least 39 lines, taking first 39 from each)
                    line_content = file.readline() # read line
                    if len(line_content)>0: # make sure line was not empty
                        line_content = line_content.split(" ") # split by spaces
                        line_content = line_content[len(line_content)-1] # get the last element
                        line_content = line_content.strip() # remove all \n
                        new_line_content = int(line_content) # cast to int
                        fileArray.append(new_line_content) # add to array containing data for this file
                file.close() # close file
                heartbeat.append(fileArray) # add the content for this file to the current heartbeat
        if len(heartbeat) > 0: # if there was data for the given hearbeat, add it to the array of heartbeats
            full_heartbeat = []
            for a in heartbeat: # add all entries, for both electrodes, to a single array to send out
                for b in a:
                    full_heartbeat.append(b)
            array.append(full_heartbeat)
    return array # return array of heartbeats, each with an array for each .0 and .1 file
print("Loading Normal heartbeats")
normals = loadHeartbeatData("normal")
for a in normals:
    a.append(0)

print("Loading Abnormal heartbeats")
abnormals = loadHeartbeatData("abnormal")
for a in abnormals:
    a.append(1)

all_ECG = [normals,abnormals] # put both sets in one array for index-based access
# select 100 unique samples randomly from either dataset to use as training data
training_dataset = []
used_indices = [[],[]]
print("Gathering training data")
for a in range(200):
    valid = False
    while valid==False:
        type = random.choice([0, 1])  # 0 = normals, 1 = abnormals
        choice = math.floor(random.random() * len(all_ECG[type])) # get random number between 0 and length of chosen data
        if choice not in used_indices[type]:
            training_dataset.append(all_ECG[type][choice])
            used_indices[type].append(choice)
            valid=True

# select 50 unique samples randomly from either dataset to use as testing data
# testing_dataset = []
# print("Gathering testing data")
# for a in range(50):
#     valid = False
#     while valid==False:
#         type = random.choice([0, 1])  # 0 = normals, 1 = abnormals
#         choice = math.floor(random.random() * len(all_ECG[type])) # get random number between 0 and length of chosen data
#         if choice not in used_indices[type]:
#             testing_dataset.append(all_ECG[type][choice])
#             used_indices[type].append(choice)
#             valid=True


X = torch.Tensor([i[0:78] for i in training_dataset])
Y = torch.Tensor([i[78] for i in training_dataset])

# Class for the network
class Net(nn.Module):
    def __init__(self): # constructor, defines the different layers and how many neurons are in each
        super(Net,self).__init__()
        self.fc1 = nn.Linear(78,2)
        self.fc2 = nn.Linear(2,1)

    # propagates value x through network to give result
    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
# creates a model
model = Net()
#print(model)

# measure error rate
criterion = nn.MSELoss()
# define optimization method
optimizer = optim.SGD(model.parameters(), lr=0.001)
# array to hold erroneous predictions
allloss = []
correctness = 0

# repeat predictions 100x
for epoch in range(100):
    print("Epoch: "+str(epoch))
    outputs = model(X)
    loss = criterion(outputs,Y)
    loss.backward()
    optimizer.step()
    allloss.append(loss.item())
    correctness = 1-loss.item()

#import pdb;pdb.set_trace()
print("Accuracy = "+str(correctness))

import matplotlib.pyplot as plt
plt.plot(allloss)
plt.show()

#print(list(model.parameters()))

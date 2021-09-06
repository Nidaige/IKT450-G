import numpy
import statistics
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8

# split into input (X) and output (Y) variables
X_train = dataset[:int(len(dataset)*splitratio),0:8]
X_val = dataset[int(len(dataset)*splitratio):,0:8]
Y_train = dataset[:int(len(dataset)*splitratio),8]
Y_val = dataset[int(len(dataset)*splitratio):,8]
print(X_train)
print(Y_train)

def distance(one,two):
    return numpy.linalg.norm(one-two)

def shortestDistance(x,x_rest,y_rest, K_val):
    k_nearest_neightbours = [] # x value of k nearest neighbours
    removeIndex = []  # indexes of the saved nearest neighbours
    for k in range(0, K_val): # loop through k times
        shortest = distance(x,x_rest[0]) # start-verdi - avstand til første element i trening
        predicted = y_rest[0] # startverdi -  tilhøyrende y til første element i trenings-sett
        currentSkipIndex = 0 # index til nåværende nærmeste element, eksporteres for hver nabo
        for i in range(len(x_rest)): # loop gjennom treningssett
            if ((distance(x,x_rest[i])<=shortest) & (i not in removeIndex)):  # hvis avstand fra x til treningssett[i] er kortere enn nåværende korteste:
                shortest = distance(x,x_rest[i])  # redefiner korteste
                predicted = y_rest[i] # redefiner predicted til å passe nye korteste
                currentSkipIndex = i  # oppdaterer hvilken index
        k_nearest_neightbours.append(predicted) # har funnet nærmeste element, setter i array
        removeIndex.append(currentSkipIndex) # setter tilhørende index i array som skal ignoreres
    return statistics.mode(k_nearest_neightbours)

TP = 0
TN = 0
FP = 0
FN = 0
K = int(input("Please input number of neighbours: "))
for i in range(len(X_val)):
    x = X_val[i]
    y = Y_val[i]
    pred = shortestDistance(x,X_train,Y_train, K)
    if(y==1 and pred ==1):
        TP += 1

    if(y==0 and pred ==0):
        TN += 1

    if(y==1 and pred ==0):
        FN += 1

    if(y==0 and pred ==1):
        FP += 1

print("Accuracy:",(TP+TN)/(TP+TN+FP+FN))
print("Recall",TP/(TP+FN))
print("Precision",TP/(TP+FP))
print("F1",(2*TP)/(2*TP+FP+FN))
print("K: "+str(K))

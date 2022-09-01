import numpy as np
import pandas as pd
import nnlib as nn
from matplotlib import pyplot as plt

# Constants for iteration and learning rate.
# Adjust these parameters to fine-tune the NN.
ITERATIONS = 200000
LEARNING_RATE = 0.2


# Printing a confirmation that calculations started.
# The more iterations, the more time it takes.
print('Calculating performace over multiple iterations. Please wait......\n')

# Reading the data from the file and splitting it into columns.
fullDataSet = pd.read_csv("data3.txt").iloc[:, 0].str.split(" ", expand=True)
fullDataSet.columns = ["i1", "i2", "i3", "i4", "i5", "i6", "o"]

# Splitting data set into train and test data sets
dataSet = fullDataSet.to_numpy()
indices = np.random.permutation(dataSet.shape[0])
trainingDataIndex, testDataIndex = indices[:80], indices[80:]
trainingDataSet, testDataSet = dataSet[trainingDataIndex, :], dataSet[testDataIndex, :]

# Creating a neural network and then training the neural network.
neuralNetwork = nn.NeuralNetwork(LEARNING_RATE)
outputErrors = neuralNetwork.train(
    (trainingDataSet[:, :6].astype(float)),
    (trainingDataSet[:, 6:].astype(float)),
    ITERATIONS)

# Shows a confirmation after calculations are completed.
print('Calculations completed and plotting inprogress. Please wait......\n')

# Plotting the training error for each iteration.
plt.plot(outputErrors, color="red", linestyle='solid', linewidth=1)
plt.xlabel("Iterations (x100)")
plt.ylabel("Total Errors")
plt.title("Performance of Neural Network over Iterations \nLearning Rate: "+str(LEARNING_RATE))
plt.show()

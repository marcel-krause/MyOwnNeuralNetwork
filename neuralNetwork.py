#!/usr/bin/env python
#Filename: neuralNetwork.py
# coding: utf-8

 ################################################################################################################
#                                                                                                                #
#                           Example of a neural network for handwriting recognition                              #
#                                                                                                                #
#   Author:      Dr. Marcel Krause                                                                               #
#   Date:        22.02.2020                                                                                      #
#   Description: This program provides a simple neural network consisting of one input layer, one hidden layer   #
#                and one output layer, each with an arbitrary amount of nodes. The network supports training     #
#                from the MNIST database and testing against it as well as saving the trained weight matrices    #
#                to CSV files or loading previously trained weight matrices, saved as CSV files, into the neural #
#                network. Finally, it provides a simple plot function to plot the characters from the MNIST      #
#                database and a back-query to plot reconstructed pseudo-input data by backfeeding expected       #
#                output through the trained network.                                                             #
#   Copyright:   Copyright (C) 2020, Dr. Marcel Krause                                                           #
#   License:     GNU General Public License (GNU GPL-3.0-or-later)                                               #
#                                                                                                                #
#                This program is released under GNU General Public License (GNU GPL-3.0-or-later).               #
#                This program is free software: you can redistribute it and/or modify it under the terms of the  #
#                GNU General Public License as published by the Free Software Foundation, either version 3 of    #
#                the License, or any later version.                                                              #
#                                                                                                                #
#                This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;       #
#                without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.       #
#                See the GNU General Public License for more details.                                            #
#                                                                                                                #
#                You have received a copy LICENSE.md of the GNU General Public License along with this program.  #
#                                                                                                                #
 ################################################################################################################


# Import all modules
import numpy
import math
import os
import matplotlib.pyplot as plt

# Configuration
pathToTrainingFile = "mnist_data" + os.sep + "mnist_train.csv"      # Path to the MNIST training data file
pathToTestFile = "mnist_data" + os.sep + "mnist_test.csv"           # Path to the MNIST test data file
pathToVisualizationFile = pathToTrainingFile        # Path to the MNIST file which contains the characters that shall be visualized
charToVisualize = 0     # Which character of the data set specified in pathToVisualizationFile shall be visualized
numberToBackquery = 0   # Which integer of the MNIST character set shall be visualized in the back-query
input_nodes = 28*28     # Number of input nodes of the network
output_nodes = 10       # Number of output nodes of the network
hidden_nodes = 100      # Number of hidden nodes of the network
learning_rate = 0.2     # Learning rate of the weights
epochs = 4              # Number of epochs (i.e. repetitions of learning from the training data set)

# Settings
wantTrainNetwork = False    # Whether the network shall be trained
wantSaveWeights = False     # Whether the current network's weight matrices shall be saved in CSV files
wantLoadWeights = True      # Whether the curret network's weight matrices shall be overwritten by loaded ones from CSV files
wantTestNetwork = True      # Whether the network's performance shall be tested against a test data set
wantVisualize = False       # Whether the MNIST characters shall be visualized
wantBackquery = False       # Whether a network back-query shall be performed (i.e. peaking into the mind of the network)

# Neural network class
class neuralNetwork:
    # Initialization
    def __init__(self, inputnodes, outputnodes, hiddennodes, learningrate):
        # Initialize the nodes and the learning rate
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes
        self.lr = learningrate
        
        # Initialize the weight matrices (wih: input->hidden, who: hidden->output)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
    # Define the activation function (we choose a sigmoid)
    def activationFunction(self, arr):
        currArr = arr
        with numpy.nditer(currArr, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = 1./(1. + math.exp(-x))
        return(currArr)
        
    # Define the inverse activation function (in our case, the logit function)
    def inverseActivationFunction(self, arr):
        currArr = arr
        with numpy.nditer(currArr, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = math.log(x/(1. - x))
        return(currArr)
    
    # Print the configuration of the network
    def printConfig(self):
        print("Configuration of the neural network:")
        print("Input nodes: " + str(self.inodes))
        print("Hidden nodes: " + str(self.hnodes))
        print("Output nodes: " + str(self.onodes))
        print("Learning rate: " + str(self.lr))
    
    # Print the weight matrices of the network
    def printWeights(self):
        print("Current weight matrices:")
        print("Weight matrix W(h,i): " + str(self.wih))
        print("Weight matrix W(o,h): " + str(self.who))

    # Save the weights to CSV files
    def saveWeights(self):
        numpy.savetxt("wih.csv", self.wih, delimiter=',')
        numpy.savetxt("who.csv", self.who, delimiter=',')
        print("The weight matrices were saved to the files wih.csv and who.csv.\n")

    # Load the weights from CSV files
    def loadWeights(self):
        self.wih = numpy.loadtxt("wih.csv", delimiter=',')
        self.who = numpy.loadtxt("who.csv", delimiter=',')
        print("The weight matrices were loaded from the files wih.csv and who.csv.\n")
    
    # Train the network (i.e. update the weights)
    def train(self, inputValues, targetValues):
        # Calculate the current output and hidden output for the given input
        outputOutputs = self.query(inputValues)[0]
        hiddenOutputs = self.query(inputValues)[1]
        
        # Convert the list of inputValues and targetValues to a numpy array
        inputs = numpy.array(inputValues, ndmin=2).T
        targets = numpy.array(targetValues, ndmin=2).T
        
        # Compute the errors of the output layer
        outputErrors = targets - outputOutputs
        
        # Back-propagate the errors to get the hidden layer errors
        hiddenErrors = numpy.dot(self.who.T, outputErrors)
        
        # Update the weight matrices
        self.who += self.lr * numpy.dot( (outputErrors*outputOutputs*(1.0 - outputOutputs)),  hiddenOutputs.T )
        self.wih += self.lr * numpy.dot( (hiddenErrors*hiddenOutputs*(1.0 - hiddenOutputs)),  inputs.T )
    
    # Query the network (i.e. provide input and get an output)
    def query(self, inputValues):
        # Convert the list of inputValues to a numpy array
        inputs = numpy.array(inputValues, ndmin=2).T
        
        # Calculate the values of the hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)
        
        # Calculate the values of the output layer
        outputInputs = numpy.dot(self.who, hiddenOutputs)
        outputOutputs = self.activationFunction(outputInputs)
        
        # Return the output values
        return outputOutputs, hiddenOutputs
    
    # Backquery the network (i.e. provide an input to the output layer and receive an output at the input layer); this in essence is just the full inverse (including inverted matrices) of the query method
    def backquery(self, targetValues):
        # Convert the list of targetValues to a numpy array
        outputOutputs = numpy.array(targetValues, ndmin=2).T

        # Calculate the "input" values of the output layer
        outputInputs = self.inverseActivationFunction(outputOutputs)

        # Calculate the "output" values of the hidden layer
        hiddenOutputs = numpy.dot(self.who.T, outputInputs)

        # Rescale the values of the hidden layer to the domain (0., 1.) since this is the domain of the logit function
        hiddenOutputs -= numpy.min(hiddenOutputs)   # Subtract the lowest value to reset the minimum to 0
        hiddenOutputs /= numpy.max(hiddenOutputs)   # Divide by the largest value to rescale all entries to the domain [0., 1.]
        hiddenOutputs *= 0.98                       # Rescale the domain to [0., 0.98]
        hiddenOutputs += 0.01                       # Translate the domain to [0.01, 0.99]

        # Calculate the "input" values of the hidden layer
        hiddenInputs = self.inverseActivationFunction(hiddenOutputs)

        # Calculate the "output" values of the input layer
        inputs = numpy.dot(self.wih.T, hiddenInputs)

        # Rescale the values of the hidden layer to the domain (0., 1.) since this is the domain of the logit function
        inputs -= numpy.min(inputs)   # Subtract the lowest value to reset the minimum to 0
        inputs /= numpy.max(inputs)   # Divide by the largest value to rescale all entries to the domain [0., 1.]
        inputs *= 255.                # Rescale the domain to [0., 255.]

        # Return the "input" values, i.e. the back-queried output
        return inputs

# Visualize the MNIST characters
if (wantVisualize):
    # Load the data file
    with open(pathToVisualizationFile, 'r') as f:
        data_values = f.readlines()
    
    # Convert the values to a matrix
    all_values = data_values[charToVisualize].split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape(28,28)

    # Plot the matrix
    plt.imshow(image_array, cmap='Greys',interpolation='None')
    print("Current number: " + all_values[:1][0])
    plt.show()

# Create an instance of the network
n = neuralNetwork(input_nodes, output_nodes, hidden_nodes, learning_rate)

# Training of the neural network
if (wantTrainNetwork):
    # Load the MNIST training data file
    with open(pathToTrainingFile, 'r') as f:
        training_data_values = f.readlines()

    # Train the neural network over the specified epochs
    print("Start training the network...")
    for e in range(epochs):
        for currData in training_data_values:
            # Get the current target and define the target array
            currTarget = int(currData.split(',')[0])
            currTargetValues = numpy.zeros(output_nodes) + 0.01
            currTargetValues[currTarget] = 0.99

            # Prepare the data for input
            currInputValues = numpy.asfarray(currData.split(',')[1:])/255*0.99 + 0.01

            # Train the network
            n.train(currInputValues, currTargetValues)
        print("Epoch " + str(e) + " finished.")
    print("Training of the network finished.\n")

# Save the weight matrices to files
if (wantSaveWeights):
    n.saveWeights()

# Load the weight matrices from files
if (wantLoadWeights):
    n.loadWeights()

# Testing the network
if (wantTestNetwork):
    # Load the MNIST test data file
    with open(pathToTestFile, 'r') as f:
        test_data_values = f.readlines()

    # Define the scorecard to keep track of (in)correct matches
    scorecard = []

    # Test the network with the test data
    print("Testing the network...")
    for currTest in test_data_values:
        # Get the actual correct value
        actualValue = int(currTest.split(',')[0])

        # Prepare the test data for input
        currTestData = numpy.asfarray(currTest.split(',')[1:])/255*0.99 + 0.01

        # Query the network with the test data
        identifiedLabel = numpy.argmax(n.query(currTestData)[0])

        # Compare the test value to the actual correct value and save the result in the scorecard
        if actualValue == identifiedLabel:
            scorecard.append(1)
        else:
            scorecard.append(0)
    print("Test of the network done.")

    # Calculate and print the performance of the network
    scorecard_array = numpy.asarray(scorecard)
    performance = scorecard_array.sum() / scorecard_array.size
    print("Performance = " + str(performance))

# Back-query the network
if (wantBackquery):
    # Define the target values which are used as input for the network
    currTargetValues = numpy.zeros(output_nodes) + 0.01
    currTargetValues[numberToBackquery] = 0.99

    # Back-query the target values through the network and reshape the result to a matrix
    backqueryValues = n.backquery(currTargetValues).reshape(28,28)

    # Plot the matrix
    plt.imshow(backqueryValues, cmap='Greys',interpolation='None')
    print("Current number: " + str(numberToBackquery))
    plt.show()
# My Own Neural Network

This program provides a class for a simple neural network consisting of one input layer, one hidden layer and one output layer, each with an arbitrary amount of nodes. As a sample implementation, the network supports training from the MNIST database and testing against it as well as saving the trained weight matrices to CSV files or loading previously trained weight matrices, saved as CSV files, into the neural network. Finally, it provides a simple plot function to plot the characters from the MNIST database and a back-query to plot reconstructed pseudo-input data by backfeeding expected output through the trained network.

The development of this program follows the book [Make Your Own Neural Network](https://books.google.de/books/about/Make_Your_Own_Neural_Network.html?id=Zli_jwEACAAJ) by [Tariq Rashid](https://github.com/makeyourownneuralnetwork).

## Program information

**Copyright** Copyright (C) 2020, Dr. Marcel Krause

**License** GNU General Public License (GNU GPL-3.0-or-later). This work is released under GNU General Public License (GNU GPL-3.0-or-later). This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You have received a copy ([LICENSE.md](LICENSE.md)) of the GNU General Public License along with this program.

**Changelog** For a documentation about the changes made in this program, check the [Changelog.md](Changelog.md) file.

**MNIST data sets** The MNIST data sets required for the training and performance check of this neural network are **not** part of this repository but can be obtained in a convenient CSV format from [https://pjreddie.com/projects/mnist-in-csv/](https://pjreddie.com/projects/mnist-in-csv/).

## Short General Description of the Artificial Neural Network
### General Description of the Network and Network Queries

![Image of the neural network](neuralNetwork.png)

The artificial neural network provided by this program consists of one input layer with ![](https://latex.codecogs.com/gif.latex?d%280%29 "d(0)") nodes, one hidden layer with ![](https://latex.codecogs.com/gif.latex?d%281%29 "d(1)") nodes and one output layer with ![](https://latex.codecogs.com/gif.latex?d%282%29 "d(2)") nodes. The following description is valid for networks with an arbitrary amount of hidden layers ![](https://latex.codecogs.com/gif.latex?n%3D1%2C2%2C3%2C%5Cdots "n=1,2,3,\dots"), however. Moreover, the neural network consists of weight matrices ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BW%7D%5E%7B%28n%2Cn-1%29%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%28n%29%20%5Ctimes%20d%28n-1%29%7D "\textbf{W}^{(n,n-1)} \in \mathbb{R}^{d(n) \times d(n-1)}") which connect all nodes from the ![](https://latex.codecogs.com/gif.latex?n%5E%5Ctext%7Bth%7D "n^\text{th}") layer with those from the ![](https://latex.codecogs.com/gif.latex?%28n-1%29%5E%5Ctext%7Bth%7D "(n-1)^\text{th}") layer. With ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BX%7D%5E%7B%28n%29%7D%20%5Cin%20%280%2C1%29%5E%7Bd%28n%29%7D "\textbf{X}^{(n)} \in (0,1)^{d(n)}"), we denote the ![](https://latex.codecogs.com/gif.latex?d%28n%29 "d(n)")-dimensional vector which contains the ![](https://latex.codecogs.com/gif.latex?d%28n%29 "d(n)") values of all nodes of the ![](https://latex.codecogs.com/gif.latex?n%5E%5Ctext%7Bth%7D "n^\text{th}") layer. The domain ![](https://latex.codecogs.com/gif.latex?%280%2C1%29 "(0,1)") for all nodes is chosen for convenience in order to avoid a saturation of the network.

The arbitrary input values are provided as an input vector ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BI%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%280%29%7D "\textbf{I}\in \mathbb{R}^{d(0)}"). This input vector is rescaled component-wise to the domain which is used throughout all nodes by the component-wise function ![](https://latex.codecogs.com/gif.latex?f_i%3A%20%5Cmathbb%7BR%7D%20%5Crightarrow%20%280%2C1%29 "f_i: \mathbb{R} \rightarrow (0,1)") as ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BX%7D%5E%7B%280%29%7D%20%3D%20f_i%28%5Ctextbf%7BI%7D%29 "\textbf{X}^{(0)} = f_i(\textbf{I})"), which specifies the values of all nodes in the input layer.

The values of the nodes in all subsequent layers are calculated by the matrix product ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BX%7D%5E%7B%28n%29%7D%20%3D%20%5Csigma%20%5Cleft%28%20%5Ctextbf%7BW%7D%5E%7B%28n%2Cn-1%29%7D%20%5Ctextbf%7BX%7D%20%5E%7B%28n-1%29%7D%20%5Cright%29 "\textbf{X}^{(n)} = \sigma \left( \textbf{W}^{(n,n-1)} \textbf{X} ^{(n-1)} \right)"), where ![](https://latex.codecogs.com/gif.latex?%5Csigma%20%3A%20%5Cmathbb%7BR%7D%20%5Crightarrow%20%280%2C1%29 "\sigma : \mathbb{R} \rightarrow (0,1)") denotes the activation function. In our case, we choose the sigmoid function ![](https://latex.codecogs.com/gif.latex?%5Csigma%20%28x%29%20%3D%20%5Cleft%28%201%20&plus;%20e%5E%7B-x%7D%20%5Cright%29%20%5E%7B-1%7D "\sigma (x) = \left( 1 + e^{-x} \right) ^{-1}") as the activation function.

The values of the nodes in the last layer ![](https://latex.codecogs.com/gif.latex?k "k") are finally rescaled to the target domain, e.g. ![](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D "\mathbb{R}"), by the component-wise function ![](https://latex.codecogs.com/gif.latex?f_o%3A%20%280%2C1%29%20%5Crightarrow%20%5Cmathbb%7BR%7D "f_o: (0,1) \rightarrow \mathbb{R}") as ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BO%7D%20%3D%20f_o%28%5Ctextbf%7BX%7D%5E%7B%28k%29%7D%29 "\textbf{O} = f_o(\textbf{X}^{(k)})").

A full network query is performed by providing the input vector ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BI%7D "\textbf{I}"), feeding it through the network and by rescaling the node values of the last layer to the target domain which yields the output vector ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BO%7D "\textbf{O}"), as described above.

### Back-Query of the Network
In order to peek into the mind of the neural network, this program provides a back-query of the (trained) network by inverting all relations described in the previous section. By providing a desired output value as input, the back-query allows to reconstruct the image of the output value which the network has through the process of training.

The input values are provided as the output vector ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BO%7D "\textbf{O}") which is rescaled to the node domain of the final layer ![](https://latex.codecogs.com/gif.latex?k "k") by the function ![](https://latex.codecogs.com/gif.latex?f_i%3A%20%5Cmathbb%7BR%7D%20%5Crightarrow%20%280%2C1%29 "f_i: \mathbb{R} \rightarrow (0,1)") as ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BX%7D%5E%7B%28k%29%7D%20%3D%20f_i%28%5Ctextbf%7BO%7D%29 "\textbf{X}^{(k)} = f_i(\textbf{O})").

The values of all nodes in the previous layers are computed by the matrix product 

![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BX%7D%20%5E%7B%28n-1%29%7D%20%3D%20f_i%20%5Cleft%28%20%5Cleft%28%20%5Ctextbf%7BW%7D%5E%7B%28n%2Cn-1%29%7D%20%5Cright%29%20%5ET%20%5Csigma%20%5E%7B-1%7D%20%5Cleft%28%20%5Ctextbf%7BX%7D%5E%7B%28n%29%7D%20%5Cright%29%20%5Cright%29 "\textbf{X} ^{(n-1)} = f_i \left( \left( \textbf{W}^{(n,n-1)} \right) ^T \sigma  ^{-1} \left( \textbf{X}^{(n)} \right)  \right)")

where ![](https://latex.codecogs.com/gif.latex?%5Csigma%20%5E%7B-1%7D%20%3A%20%280%2C1%29%20%5Crightarrow%20%5Cmathbb%7BR%7D "\sigma ^{-1} : (0,1) \rightarrow \mathbb{R}") denotes the inverse activation function which in our case is given by the logit function
![](https://latex.codecogs.com/gif.latex?%5Csigma%20%5E%7B-1%7D%20%28x%29%20%3D%20%5Cln%20%5Cleft%28%20%5Cfrac%7Bx%7D%7B1-x%7D%20%5Cright%29 "\sigma ^{-1} (x) = \ln \left( \frac{x}{1-x} \right)").
Since the hereby calculated values of the logit function may lie outside the node domain, we additionally perform a rescaling back to the node domain after each step.

The output values ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BI%7D "\textbf{I}") calculated after the input layer are finally given by rescaling the values of the nodes at the input layer to the target domain by the component-wise function ![](https://latex.codecogs.com/gif.latex?f_o%3A%20%5Cmathbb%7BR%7D%20%5Crightarrow%20%5Cmathbb%7BR%7D "f_o: \mathbb{R} \rightarrow \mathbb{R}") as ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BI%7D%20%3D%20f_o%28%5Ctextbf%7BX%7D%5E%7B%280%29%7D%29 "\textbf{I} = f_o(\textbf{X}^{(0)})").

The following image is an example of the back-query created by this code after having trained the network against the MNIST data set:

![Back-query of the neural network](backquery.png)

### Initialization of the Network
During the initialization of the network, the values of all weight matrices ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BW%7D%5E%7B%28n%2Cn-1%29%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%28n%29%20%5Ctimes%20d%28n-1%29%7D "\textbf{W}^{(n,n-1)} \in \mathbb{R}^{d(n) \times d(n-1)}") are randomly determined from a normal distribution with mean value at zero and a standard deviation of ![](https://latex.codecogs.com/gif.latex?1%20/%20%5Csqrt%7Bd%28n%29%7D "1 / \sqrt{d(n)}") in order to avoid a saturation of the network.

### Training the Network
For training the network, pairs of input and target output values (i.e. data known to be true) are used. For example, given the input vector ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BI%7D "\textbf{I}"), the input is connected to an expected target output ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BT%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%28k%29%7D "\textbf{T}\in \mathbb{R}^{d(k)}") at the ![](https://latex.codecogs.com/gif.latex?k%5E%5Ctext%7Bth%7D "k^\text{th}") layer, i.e. the desired value which the network shall output when the input ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BI%7D "\textbf{I}") is provided. 

In a first step, the input ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BI%7D "\textbf{I}") is given and fed through the network, yielding the output ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BO%7D "\textbf{O}"). The error is then given by ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BE%7D%20%5E%7B%28k%29%7D%20%3D%20%5Ctextbf%7BT%7D%20-%20%5Ctextbf%7BO%7D "\textbf{E}^{(k)} = \textbf{T} - \textbf{O}"). This error is then back-propagated through all layers of the network in subsequent steps as ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BE%7D%20%5E%7B%28n-1%29%7D%20%3D%20%5Cleft%28%5Ctextbf%7BW%7D%5E%7B%28n%2Cn-1%29%7D%20%5Cright%29%20%5ET%20%5Ctextbf%7BE%7D%20%5E%7B%28n%29%7D "\textbf{E} ^{(n-1)} = \left(\textbf{W}^{(n,n-1)} \right) ^T \textbf{E} ^{(n)}").

The weights are updated by minimizing the error function at each layer. This leads to the update of the weight matrices as ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BW%7D%5E%7B%28n%2Cn-1%29%7D%20%5Crightarrow%20%5Ctextbf%7BW%7D%5E%7B%28n%2Cn-1%29%7D%20&plus;%20%5CDelta%20%5Ctextbf%7BW%7D%5E%7B%28n%2Cn-1%29%7D "\textbf{W}^{(n,n-1)} \rightarrow \textbf{W}^{(n,n-1)} + \Delta \textbf{W}^{(n,n-1)}"), where the weight matrix update is given by

![](https://latex.codecogs.com/gif.latex?%5Calpha%20%5Cleft%28%20%5Ctextbf%7BE%7D%5E%7B%28n%29%7D%20%5Ccirc%20%5Csigma%20%28%5Ctextbf%7BX%7D%5E%7B%28n%29%7D%29%20%5Ccirc%20%5Cleft%28%201%20-%20%5Csigma%20%28%5Ctextbf%7BX%7D%5E%7B%28n%29%7D%29%20%5Cright%29%20%5Cright%29%20%5Cleft%28%20%5Csigma%20%28%5Ctextbf%7BX%7D%5E%7B%28n%29%7D%29%20%5Cright%29%20%5ET "\alpha \left( \textbf{E}^{(n)} \circ \sigma (\textbf{X}^{(n)}) \circ \left( 1 - \sigma (\textbf{X}^{(n)}) \right) \right) \left( \sigma (\textbf{X}^{(n)}) \right) ^T")

where ![](https://latex.codecogs.com/gif.latex?%5Calpha "\alpha") is the learning rate of the network and ![](https://latex.codecogs.com/gif.latex?%5Ccirc "\circ") denotes the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)).

In order to improve the performance of the network, the aforementioned steps can be repeated one or more times, with each repetition being referred to as an epoch.
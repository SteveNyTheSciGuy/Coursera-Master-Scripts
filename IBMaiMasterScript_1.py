# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:13:53 2020

@author: Steven

Copied, augmented, tweaked, created code from Coursera courses: 
    IBM AI Engineering
        ML w/ Python
            cor_cov_1()
            lin_reg_1()
            reg_1()
            ann_1()
            clustering_1()
            class_1()
            recc_1()
        Apache Spark
            spark_1()
        Keras
            cement_strength_1()
            other_keras_1
        PyTorch
        TF
            cement_images()
        Capstone
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




###  ###
def cor_cov_1():
    x=[1,2,3,4,5,6,7,8,9,10]
    y=[7,6,5,4,5,6,7,8,9,10]
    
    x_avg=np.mean(x)
    y_avg=np.mean(y)
    n=len(x)
    cor_val=0
    
    for i in range(n):
        cor_val=cor_val+(x[i]-x_avg)*(y[i]-y_avg)
    
    cov=cor_val/n
    print(cov)
    
    x_std=np.std(x)
    y_std=np.std(y)
    cor=cov/(x_std*y_std)
    print(cor)
#cor_cov_1()

###  ###
def lin_reg_1():
    df = pd.read_csv("Data/FuelConsumptionCo2.csv")

    # take a look at the dataset
    print(df.head())
    
    # summarize the data
    print(df.describe())
    
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    cdf.head(9)
    
    # Plotting pre-processed data for trends
    viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
    viz.hist()
    plt.show()
    
    plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
    plt.xlabel("FUELCONSUMPTION_COMB")
    plt.ylabel("Emission")
    plt.show()
    
    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()
    
    plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Cylinders")
    plt.ylabel("Emission")
    plt.show()
    
    
    # Tranining for Simple Linear Regression
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]
    
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()
    
    
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit (train_x, train_y)
    # The coefficients
    print ('Coefficients: ', regr.coef_)
    print ('Intercept: ',regr.intercept_)
    
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    
    from sklearn.metrics import r2_score
    
    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_hat = regr.predict(test_x)
    
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
    
    
    # Tranining for Multiple Linear Regression
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    cdf.head(9)
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]
    
    
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit (x, y)
    # The coefficients
    print ('Coefficients: ', regr.coef_)
    
    y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Residual sum of squares: %.2f"
          % np.mean((y_hat - y) ** 2))
    
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x, y))
    
    regr = linear_model.LinearRegression()
    x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
    y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit (x, y)
    print ('Coefficients: ', regr.coef_)
    y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
    x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
    print('Variance score: %.2f' % regr.score(x, y))
    
    
    ### NonLinear Regression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    
    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    
    
    poly = PolynomialFeatures(degree=2)
    train_x_poly = poly.fit_transform(train_x)
    train_x_poly
    
    clf = linear_model.LinearRegression()
    train_y_ = clf.fit(train_x_poly, train_y)
    # The coefficients
    print ('Coefficients: ', clf.coef_)
    print ('Intercept: ',clf.intercept_)
    
    
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    XX = np.arange(0.0, 10.0, 0.1)
    yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
    plt.plot(XX, yy, '-r' )
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    
    from sklearn.metrics import r2_score
    
    test_x_poly = poly.fit_transform(test_x)
    test_y_ = clf.predict(test_x_poly)
    
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
    
    poly3 = PolynomialFeatures(degree=3)
    train_x_poly3 = poly3.fit_transform(train_x)
    clf3 = linear_model.LinearRegression()
    train_y3_ = clf3.fit(train_x_poly3, train_y)
    # The coefficients
    print ('Coefficients: ', clf3.coef_)
    print ('Intercept: ',clf3.intercept_)
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    XX = np.arange(0.0, 10.0, 0.1)
    yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)
    plt.plot(XX, yy, '-r' )
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    test_x_poly3 = poly3.fit_transform(test_x)
    test_y3_ = clf3.predict(test_x_poly3)
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y3_ , test_y) )
#lin_reg_1()

###  ###
def reg_1():
    df = pd.read_csv("Data/china_gdp.csv")
    df.head(10)
    
    plt.figure(figsize=(8,5))
    x_data, y_data = (df["Year"].values, df["Value"].values)
    plt.plot(x_data, y_data, 'ro')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()
    
    X = np.arange(-5.0, 5.0, 0.1)
    Y = 1.0 / (1.0 + np.exp(-X))
    
    plt.plot(X,Y) 
    plt.ylabel('Dependent Variable')
    plt.xlabel('Indepdendent Variable')
    plt.show()
    
    def sigmoid(x, Beta_1, Beta_2):
         y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
         return y
    
    beta_1 = 0.10
    beta_2 = 1990.0
    
    #logistic function
    Y_pred = sigmoid(x_data, beta_1 , beta_2)
    
    #plot initial prediction against datapoints
    plt.plot(x_data, Y_pred*15000000000000.)
    plt.plot(x_data, y_data, 'ro')
    
    # Lets normalize our data
    xdata =x_data/max(x_data)
    ydata =y_data/max(y_data)
    
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    #print the final parameters
    print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
    
    x = np.linspace(1960, 2015, 55)
    x = x/max(x)
    plt.figure(figsize=(8,5))
    y = sigmoid(x, *popt)
    plt.plot(xdata, ydata, 'ro', label='data')
    plt.plot(x,y, linewidth=3.0, label='fit')
    plt.legend(loc='best')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()
    
    # split data into train/test
    msk = np.random.rand(len(df)) < 0.8
    train_x = xdata[msk]
    test_x = xdata[~msk]
    train_y = ydata[msk]
    test_y = ydata[~msk]
    
    # build the model using train set
    popt, pcov = curve_fit(sigmoid, train_x, train_y)
    
    # predict using test set
    y_hat = sigmoid(test_x, *popt)
    
    # evaluation
    print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
    from sklearn.metrics import r2_score
    print("R2-score: %.2f" % r2_score(y_hat , test_y) )
#reg_1()

###  ###
def ann_1():
    weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
    biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases
    
    print(weights)
    print(biases)
    
    x_1 = 0.5 # input 1
    x_2 = 0.85 # input 2
    
    print('x1 is {} and x2 is {}'.format(x_1, x_2))
    
    z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
    
    print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))
    
    z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
    print('The weighted sum of the inputs at the 2nd node in the hidden layer is {}'.format(z_12))
    
    print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))
    
    a_11 = 1.0 / (1.0 + np.exp(-z_11))
    
    print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))
    
    a_12 = 1.0 / (1.0 + np.exp(-z_12))
    
    print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))
    
    z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
    
    print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))
    
    a_2 = 1.0 / (1.0 + np.exp(-z_2))
    
    print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))
    
    ## Initialize a Network
    n = 2 # number of inputs
    num_hidden_layers = 2 # number of hidden layers
    m = [2, 2] # number of nodes in each hidden layer
    num_nodes_output = 1 # number of nodes in the output layer
    
    num_nodes_previous = n # number of nodes in the previous layer
    
    network = {} # initialize network an an empty dictionary
    
    # loop through each layer and randomly initialize the weights and biases associated with each node
    # notice how we are adding 1 to the number of hidden layers in order to include the output layer
    for layer in range(num_hidden_layers + 1): 
        
        # determine name of layer
        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = m[layer]
        
        # initialize weights and biases associated with each node in the current layer
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
        
        num_nodes_previous = num_nodes
        
    print(network) # print network
    
    def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
        
        num_nodes_previous = num_inputs # number of nodes in the previous layer
    
        network = {}
        
        # loop through each layer and randomly initialize the weights and biases associated with each layer
        for layer in range(num_hidden_layers + 1):
            
            if layer == num_hidden_layers:
                layer_name = 'output' # name last layer in the network output
                num_nodes = num_nodes_output
            else:
                layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
                num_nodes = num_nodes_hidden[layer] 
            
            # initialize weights and bias for each node
            network[layer_name] = {}
            for node in range(num_nodes):
                node_name = 'node_{}'.format(node+1)
                network[layer_name][node_name] = {
                    'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                    'bias': np.around(np.random.uniform(size=1), decimals=2),
                }
        
            num_nodes_previous = num_nodes
    
        return network # return the network
    
    small_network = initialize_network(5, 3, [3,2,3], 1)
    
    def compute_weighted_sum(inputs, weights, bias):
        return np.sum(inputs * weights) + bias
    
    from random import seed
    
    np.random.seed(12)
    inputs = np.around(np.random.uniform(size=5), decimals=2)
    
    print('The inputs to the network are {}'.format(inputs))
    
    node_weights = small_network['layer_1']['node_1']['weights']
    node_bias = small_network['layer_1']['node_1']['bias']
    
    weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
    print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))
    
    def node_activation(weighted_sum):
        return 1.0 / (1.0 + np.exp(-1 * weighted_sum))
    
    node_output  = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
    print('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0], decimals=4)))
    
    def forward_propagate(network, inputs):
        
        layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
        
        for layer in network:
            
            layer_data = network[layer]
            
            layer_outputs = [] 
            for layer_node in layer_data:
            
                node_data = layer_data[layer_node]
            
                # compute the weighted sum and the output of each node at the same time 
                node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
                layer_outputs.append(np.around(node_output[0], decimals=4))
                
            if layer != 'output':
                print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
        
            layer_inputs = layer_outputs # set the output of this layer to be the input to next layer
    
        network_predictions = layer_outputs
        return network_predictions
    
    predictions = forward_propagate(small_network, inputs)
    print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))
    
    #Summary
    my_network = initialize_network(5, 3, [2, 3, 2], 3)
    inputs = np.around(np.random.uniform(size=5), decimals=2)
    predictions = forward_propagate(my_network, inputs)
    print('The predicted values by the network for the given input are {}'.format(predictions))
#ann_1()

###  ###
def clustering_1():
    ### K Means
    
    import random 
    import numpy as np 
    import matplotlib.pyplot as plt 
    from sklearn.cluster import KMeans 
    from sklearn.datasets.samples_generator import make_blobs 
    
    np.random.seed(0)
    
    X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
    
    plt.scatter(X[:, 0], X[:, 1], marker='.')
    
    k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
    
    print(k_means.fit(X))
    
    k_means_labels = k_means.labels_
    print(k_means_labels)
    
    k_means_cluster_centers = k_means.cluster_centers_
    print(k_means_cluster_centers)
    
    # Initialize the plot with the specified dimensions.
    fig = plt.figure(figsize=(6, 4))
    
    # Colors uses a color map, which will produce an array of colors based on
    # the number of labels there are. We use set(k_means_labels) to get the
    # unique labels.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
    
    # Create a plot
    ax = fig.add_subplot(1, 1, 1)
    
    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
    
        # Create a list of all data points, where the data poitns that are 
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        my_members = (k_means_labels == k)
        
        # Define the centroid, or cluster center.
        cluster_center = k_means_cluster_centers[k]
        
        # Plots the datapoints with color col.
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
        
        # Plots the centroids with specified color, but with a darker outline
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
    
    # Title of the plot
    ax.set_title('KMeans')
    
    # Remove x-axis ticks
    ax.set_xticks(())
    
    # Remove y-axis ticks
    ax.set_yticks(())
    
    # Show the plot
    plt.show()
    
    k_means3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
    k_means3.fit(X)
    fig = plt.figure(figsize=(6, 4))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
    ax = fig.add_subplot(1, 1, 1)
    for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
        my_members = (k_means3.labels_ == k)
        cluster_center = k_means3.cluster_centers_[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
    plt.show()
    
    
    #!wget -O Cust_Segmentation.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv
    import pandas as pd
    cust_df = pd.read_csv("Data/Cust_Segmentation.csv")
    cust_df.head()
    
    df = cust_df.drop('Address', axis=1)
    
    from sklearn.preprocessing import StandardScaler
    X = df.values[:,1:]
    X = np.nan_to_num(X)
    Clus_dataSet = StandardScaler().fit_transform(X)
    
    clusterNum = 3
    k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
    k_means.fit(X)
    labels = k_means.labels_
    print(labels)
    
    df["Clus_km"] = labels
    
    df.groupby('Clus_km').mean()
    
    area = np.pi * ( X[:, 1])**2  
    plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
    plt.xlabel('Age', fontsize=18)
    plt.ylabel('Income', fontsize=16)
    
    plt.show()
    
    from mpl_toolkits.mplot3d import Axes3D 
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    plt.cla()
    # plt.ylabel('Age', fontsize=18)
    # plt.xlabel('Income', fontsize=16)
    # plt.zlabel('Education', fontsize=16)
    ax.set_xlabel('Education')
    ax.set_ylabel('Age')
    ax.set_zlabel('Income')
    
    ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
    
    
    ### Hierarchical Clustering
    import numpy as np 
    import pandas as pd
    from scipy import ndimage 
    from scipy.cluster import hierarchy 
    from scipy.spatial import distance_matrix 
    from matplotlib import pyplot as plt 
    from sklearn import manifold, datasets 
    from sklearn.cluster import AgglomerativeClustering 
    from sklearn.datasets.samples_generator import make_blobs 
    
    X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
    
    #plt.scatter(X1[:, 0], X1[:, 1], marker='o') 
    
    agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
    
    agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
    
    # Create a figure of size 6 inches by 4 inches.
    plt.figure(figsize=(6,4))
    
    # These two lines of code are used to scale the data points down,
    # Or else the data points will be scattered very far apart.
    
    # Create a minimum and maximum range of X1.
    x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)
    
    # Get the average distance for X1.
    X1 = (X1 - x_min) / (x_max - x_min)
    
    # This loop displays all of the datapoints.
    for i in range(X1.shape[0]):
        # Replace the data points with their respective cluster value 
        # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
        plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
                 #color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
        
    # Remove the x ticks, y ticks, x and y axis
    plt.xticks([])
    plt.yticks([])
    #plt.axis('off')
    
    
    
    # Display the plot of the original data before clustering
    plt.scatter(X1[:, 0], X1[:, 1], marker='.')
    # Display the plot
    plt.show()
    
    dist_matrix = distance_matrix(X1,X1) 
    print(dist_matrix)
    
    Z = hierarchy.linkage(dist_matrix, 'complete')
    
    dendro = hierarchy.dendrogram(Z)
    
    Z = hierarchy.linkage(dist_matrix, 'average')
    dendro = hierarchy.dendrogram(Z)
    
    
    #!wget -O cars_clus.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv
    
    filename = 'Data/cars_clus.csv'
    
    #Read csv
    pdf = pd.read_csv(filename)
    print ("Shape of dataset: ", pdf.shape)
    
    print ("Shape of dataset before cleaning: ", pdf.size)
    pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
           'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
           'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
           'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
           'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
    pdf = pdf.dropna()
    pdf = pdf.reset_index(drop=True)
    print ("Shape of dataset after cleaning: ", pdf.size)
    
    featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
    
    from sklearn.preprocessing import MinMaxScaler
    x = featureset.values #returns a numpy array
    min_max_scaler = MinMaxScaler()
    feature_mtx = min_max_scaler.fit_transform(x)
    
    from sklearn.preprocessing import MinMaxScaler
    x = featureset.values #returns a numpy array
    min_max_scaler = MinMaxScaler()
    feature_mtx = min_max_scaler.fit_transform(x)
    
    import scipy
    leng = feature_mtx.shape[0]
    D = scipy.zeros([leng,leng])
    for i in range(leng):
        for j in range(leng):
            D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
            
    import pylab
    import scipy.cluster.hierarchy
    Z = hierarchy.linkage(D, 'complete')
    
    from scipy.cluster.hierarchy import fcluster
    max_d = 3
    clusters = fcluster(Z, max_d, criterion='distance')
    
    from scipy.cluster.hierarchy import fcluster
    k = 5
    clusters = fcluster(Z, k, criterion='maxclust')
    
    fig = pylab.figure(figsize=(18,50))
    def llf(id):
        return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
        
    dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
    
    
    dist_matrix = distance_matrix(feature_mtx,feature_mtx) 
    print(dist_matrix)
    
    agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
    agglom.fit(feature_mtx)
    
    pdf['cluster_'] = agglom.labels_
    
    import matplotlib.cm as cm
    n_clusters = max(agglom.labels_)+1
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    cluster_labels = list(range(0, n_clusters))
    
    # Create a figure of size 6 inches by 4 inches.
    plt.figure(figsize=(16,14))
    
    for color, label in zip(colors, cluster_labels):
        subset = pdf[pdf.cluster_ == label]
        for i in subset.index:
                plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
        plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
    #    plt.scatter(subset.horsepow, subset.mpg)
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('horsepow')
    plt.ylabel('mpg')
    
    pdf.groupby(['cluster_','type'])['cluster_'].count()
    
    agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
    
    plt.figure(figsize=(16,10))
    for color, label in zip(colors, cluster_labels):
        subset = agg_cars.loc[(label,),]
        for i in subset.index:
            plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
        plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('horsepow')
    plt.ylabel('mpg')
    
    
    ### DBSCAN
    #!wget -O weather-stations20140101-20141231.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/weather-stations20140101-20141231.csv
    import csv
    import pandas as pd
    import numpy as np
    
    filename='Data/weather-stations20140101-20141231.csv'
    
    #Read csv
    pdf = pd.read_csv(filename)
    
    pdf = pdf[pd.notnull(pdf["Tm"])]
    pdf = pdf.reset_index(drop=True)
    
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = (14,10)
    
    llon=-140
    ulon=-50
    llat=40
    ulat=65
    
    pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]
    
    my_map = Basemap(projection='merc',
                resolution = 'l', area_thresh = 1000.0,
                llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
                urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)
    
    my_map.drawcoastlines()
    my_map.drawcountries()
    # my_map.drawmapboundary()
    my_map.fillcontinents(color = 'white', alpha = 0.3)
    my_map.shadedrelief()
    
    # To collect data based on stations        
    
    xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
    pdf['xm']= xs.tolist()
    pdf['ym'] =ys.tolist()
    
    #Visualization1
    for index,row in pdf.iterrows():
    #   x,y = my_map(row.Long, row.Lat)
       my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
    #plt.text(x,y,stn)
    plt.show()
    
    from sklearn.cluster import DBSCAN
    import sklearn.utils
    from sklearn.preprocessing import StandardScaler
    sklearn.utils.check_random_state(1000)
    Clus_dataSet = pdf[['xm','ym']]
    Clus_dataSet = np.nan_to_num(Clus_dataSet)
    Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)
    
    # Compute DBSCAN
    db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    pdf["Clus_Db"]=labels
    
    realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
    clusterNum = len(set(labels)) 
    
    
    # A sample of clusters
    pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)
    
    set(labels)
    
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = (14,10)
    
    my_map = Basemap(projection='merc',
                resolution = 'l', area_thresh = 1000.0,
                llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
                urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)
    
    my_map.drawcoastlines()
    my_map.drawcountries()
    #my_map.drawmapboundary()
    my_map.fillcontinents(color = 'white', alpha = 0.3)
    my_map.shadedrelief()
    
    # To create a color map
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))
    
    
    
    #Visualization1
    for clust_number in set(labels):
        c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
        clust_set = pdf[pdf.Clus_Db == clust_number]                    
        my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
        if clust_number != -1:
            cenx=np.mean(clust_set.xm) 
            ceny=np.mean(clust_set.ym) 
            plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
            print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))
#clustering_1()

###  ###
def class_1():
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    import pandas as pd
    import numpy as np
    import matplotlib.ticker as ticker
    from sklearn import preprocessing
    
    df = pd.read_csv('Data/teleCust1000t.csv')
    df.head()
    
    df['custcat'].value_counts()
    
    df.hist(column='income', bins=50)
    
    print(df.columns)
    
    X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
    print(X[0:5])
    
    y = df['custcat'].values
    print(y[0:5])
    
    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    print(X[0:5])
    
    # K Nearest Neighbors
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)
    
    from sklearn.neighbors import KNeighborsClassifier
    
    k = 4
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    print(neigh)
    
    yhat = neigh.predict(X_test)
    print(yhat[0:5])
    
    from sklearn import metrics
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
    
    k = 6
    neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    yhat6 = neigh6.predict(X_test)
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))
    
    
    # Compare multiple K values
    Ks = 10
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))
    ConfustionMx = [];
    for n in range(1,Ks):
        
        #Train Model and Predict  
        neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
        yhat=neigh.predict(X_test)
        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    
        
        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    
    mean_acc
    
    plt.plot(range(1,Ks),mean_acc,'g')
    plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Nabors (K)')
    plt.tight_layout()
    plt.show()
    
    print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
    
    
    # Decision Trees
    import numpy as np 
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    
    my_data = pd.read_csv("Data/drug200.csv", delimiter=",")
    print(my_data[0:5])
    
    X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    print(X[0:5])
    
    from sklearn import preprocessing
    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F','M'])
    X[:,1] = le_sex.transform(X[:,1]) 
    
    
    le_BP = preprocessing.LabelEncoder()
    le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
    X[:,2] = le_BP.transform(X[:,2])
    
    
    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit([ 'NORMAL', 'HIGH'])
    X[:,3] = le_Chol.transform(X[:,3]) 
    
    print(X[0:5])
    
    y = my_data["Drug"]
    
    from sklearn.model_selection import train_test_split
    
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
    
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    print(drugTree) # it shows the default parameters
    
    drugTree.fit(X_trainset,y_trainset)
    
    predTree = drugTree.predict(X_testset)
    
    print (predTree [0:5])
    print (y_testset [0:5])
    
    from sklearn import metrics
    import matplotlib.pyplot as plt
    print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
    
    # Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
    # !conda install -c conda-forge pydotplus -y
    # !conda install -c conda-forge python-graphviz -y
    
    #from sklearn.externals.six import StringIO
    #import pydotplus
    import matplotlib.image as mpimg
    from sklearn import tree
    
    # dot_data = StringIO()
    # filename = "drugtree.png"
    # featureNames = my_data.columns[0:5]
    # targetNames = my_data["Drug"].unique().tolist()
    # out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # graph.write_png(filename)
    # img = mpimg.imread(filename)
    #plt.figure(figsize=(100, 200))
    #plt.imshow(img,interpolation='nearest')
    
    
    ### Logistic Regression
    import pandas as pd
    import pylab as pl
    import numpy as np
    import scipy.optimize as opt
    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    
    churn_df = pd.read_csv("Data/ChurnData.csv")
    
    churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
    churn_df['churn'] = churn_df['churn'].astype('int')
    churn_df.head()
    
    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    
    y = np.asarray(churn_df['churn'])
    
    from sklearn import preprocessing
    X = preprocessing.StandardScaler().fit(X).transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    print(LR)
    
    yhat = LR.predict(X_test)
    
    yhat_prob = LR.predict_proba(X_test)
    
    #from sklearn.metrics import jaccard_similarity_score
    #jaccard_similarity_score(y_test, yhat)
    
    from sklearn.metrics import classification_report, confusion_matrix
    import itertools
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    print(confusion_matrix(y_test, yhat, labels=[1,0]))
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
    np.set_printoptions(precision=2)
    
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
    
    print (classification_report(y_test, yhat))
    
    from sklearn.metrics import log_loss
    print(log_loss(y_test, yhat_prob))
    
    LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
    yhat_prob2 = LR2.predict_proba(X_test)
    print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))
    
    
    ### SVM
    import pandas as pd
    import pylab as pl
    import numpy as np
    import scipy.optimize as opt
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    cell_df = pd.read_csv("Data/cell_samples.csv")
    
    ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
    cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
    plt.show()
    
    print(cell_df.dtypes)
    
    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
    print(cell_df.dtypes)
    
    feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
    X = np.asarray(feature_df)
    
    cell_df['Class'] = cell_df['Class'].astype('int')
    y = np.asarray(cell_df['Class'])
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)
    
    from sklearn import svm
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train) 
    
    yhat = clf.predict(X_test)
    yhat [0:5]
    
    from sklearn.metrics import classification_report, confusion_matrix
    import itertools
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
    np.set_printoptions(precision=2)
    
    print (classification_report(y_test, yhat))
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
    
    from sklearn.metrics import f1_score
    f1_score(y_test, yhat, average='weighted')
    
    #from sklearn.metrics import jaccard_similarity_score
    #jaccard_similarity_score(y_test, yhat)
    
    clf2 = svm.SVC(kernel='linear')
    clf2.fit(X_train, y_train) 
    yhat2 = clf2.predict(X_test)
    print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
    #print("Jaccard score: %.4f" % jaccard_similarity_score(y_test, yhat2))
#class_1()

###  ###
def recc_1():
    #!wget -O moviedataset.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
    #print('unziping ...')
    #unzip -o -j moviedataset.zip 
    
    #Dataframe manipulation library
    import pandas as pd
    #Math functions, we'll only need the sqrt function so let's import only that
    from math import sqrt
    import numpy as np
    import matplotlib.pyplot as plt
    
    #Storing the movie information into a pandas dataframe
    movies_df = pd.read_csv('Data/movies.csv')
    #Storing the user information into a pandas dataframe
    ratings_df = pd.read_csv('Data/ratings.csv')
    #Head is a function that gets the first N rows of a dataframe. N's default is 5.
    
    #Using regular expressions to find a year stored between parentheses
    #We specify the parantheses so we don't conflict with movies that have years in their titles
    movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
    #Removing the parentheses
    movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
    #Removing the years from the 'title' column
    movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
    #Applying the strip function to get rid of any ending whitespace characters that may have appeared
    movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
    
    #Every genre is separated by a | so we simply have to call the split function on |
    movies_df['genres'] = movies_df.genres.str.split('|')
    
    #Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
    moviesWithGenres_df = movies_df.copy()
    
    #For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            moviesWithGenres_df.at[index, genre] = 1
    #Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
    moviesWithGenres_df = moviesWithGenres_df.fillna(0)
    
    #Drop removes a specified row or column from a dataframe
    ratings_df = ratings_df.drop('timestamp', 1)
    
    userInput = [
                {'title':'Breakfast Club, The', 'rating':5},
                {'title':'Toy Story', 'rating':3.5},
                {'title':'Jumanji', 'rating':2},
                {'title':"Pulp Fiction", 'rating':5},
                {'title':'Akira', 'rating':4.5}
             ] 
    inputMovies = pd.DataFrame(userInput)
    
    #Filtering out the movies by title
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    #Then merging it so we can get the movieId. It's implicitly merging it by title.
    inputMovies = pd.merge(inputId, inputMovies)
    #Dropping information we won't use from the input dataframe
    inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
    #Final input dataframe
    #If a movie you added in above isn't here, then it might not be in the original 
    #dataframe or it might spelled differently, please check capitalisation.
    
    #Filtering out the movies from the input
    userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
    
    #Resetting the index to avoid future issues
    userMovies = userMovies.reset_index(drop=True)
    #Dropping unnecessary issues due to save memory and to avoid issues
    userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
    
    #Dot produt to get weights
    userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
    #The user profile
    
    #Now let's get the genres of every movie in our original dataframe
    genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
    #And drop the unnecessary information
    genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
    
    #Now let's get the genres of every movie in our original dataframe
    genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
    #And drop the unnecessary information
    genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
    
    #Multiply the genres by the weights and then take the weighted average
    recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
    
    #Sort our recommendations in descending order
    recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
    #Just a peek at the values
    
    #The final recommendation table
    print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])
    
    
    ### Collaborative Filtering
    #Dataframe manipulation library
    import pandas as pd
    #Math functions, we'll only need the sqrt function so let's import only that
    from math import sqrt
    import numpy as np
    import matplotlib.pyplot as plt
    
    #Storing the movie information into a pandas dataframe
    movies_df = pd.read_csv('Data/movies.csv')
    #Storing the user information into a pandas dataframe
    ratings_df = pd.read_csv('Data/ratings.csv')
    
    #Using regular expressions to find a year stored between parentheses
    #We specify the parantheses so we don't conflict with movies that have years in their titles
    movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
    #Removing the parentheses
    movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
    #Removing the years from the 'title' column
    movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
    #Applying the strip function to get rid of any ending whitespace characters that may have appeared
    movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
    
    #Dropping the genres column
    movies_df = movies_df.drop('genres', 1)
    
    #Drop removes a specified row or column from a dataframe
    ratings_df = ratings_df.drop('timestamp', 1)
    
    userInput = [
                {'title':'Breakfast Club, The', 'rating':5},
                {'title':'Toy Story', 'rating':3.5},
                {'title':'Jumanji', 'rating':2},
                {'title':"Pulp Fiction", 'rating':5},
                {'title':'Akira', 'rating':4.5}
             ] 
    inputMovies = pd.DataFrame(userInput)
    
    #Filtering out the movies by title
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    #Then merging it so we can get the movieId. It's implicitly merging it by title.
    inputMovies = pd.merge(inputId, inputMovies)
    #Dropping information we won't use from the input dataframe
    inputMovies = inputMovies.drop('year', 1)
    #Final input dataframe
    #If a movie you added in above isn't here, then it might not be in the original 
    #dataframe or it might spelled differently, please check capitalisation.
    
    #Filtering out users that have watched movies that the input has watched and storing it
    userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
    
    #Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
    userSubsetGroup = userSubset.groupby(['userId'])
    
    #Sorting it so users with movie most in common with the input will have priority
    userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
    
    userSubsetGroup = userSubsetGroup[0:100]
    
    #Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
    pearsonCorrelationDict = {}
    
    #For every user group in our subset
    for name, group in userSubsetGroup:
        #Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='movieId')
        inputMovies = inputMovies.sort_values(by='movieId')
        #Get the N for the formula
        nRatings = len(group)
        #Get the review scores for the movies that they both have in common
        temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
        #And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp_df['rating'].tolist()
        #Let's also put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()
        #Now let's calculate the pearson correlation between two users, so called, x and y
        Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
        Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
        Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
        
        #If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
        else:
            pearsonCorrelationDict[name] = 0
    
    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['userId'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    
    topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
    
    topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
    
    #Multiplies the similarity by the user's ratings
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
    
    #Applies a sum to the topUsers after grouping it up by userId
    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
    
    #Creates an empty dataframe
    recommendation_df = pd.DataFrame()
    #Now we take the weighted average
    recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
    recommendation_df['movieId'] = tempTopUsersRating.index
    
    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
    print('Reccomended Movies \n',recommendation_df.head(10))
    
    print(movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())])
#recc_1()

###  ###
def spark_1():
    ## Should run on IBM Watson ###
    
    from IPython.display import Markdown, display
    def printmd(string):
        display(Markdown('# <span style="color:red">'+string+'</span>'))
    
    
    if ('sc' in locals() or 'sc' in globals()):
        printmd('<<<<<!!!!! It seems that you are running in a IBM Watson Studio Apache Spark Notebook. Please run it in an IBM Watson Studio Default Runtime (without Apache Spark) !!!!!>>>>>')
        
        
    try:
        from pyspark import SparkContext, SparkConf
        from pyspark.sql import SparkSession
    except ImportError as e:
        printmd('<<<<<!!!!! Please restart your kernel after installing Apache Spark !!!!!>>>>>')    
        
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    
    spark = SparkSession \
        .builder \
        .getOrCreate()    
        
        
    
    from pyspark.sql import Row
    
    df = spark.createDataFrame([Row(id=1, value='value1'),Row(id=2, value='value2')])
    
    # let's have a look what's inside
    df.show()
    
    # let's print the schema
    df.printSchema()    
        
        
    # register dataframe as query table
    df.createOrReplaceTempView('df_view')
    
    # execute SQL query
    df_result = spark.sql('select value from df_view where id=2')
    
    # examine contents of result
    df_result.show()
    
    # get result as string
    df_result.first().value    
        
    #df.$$$   
        
        
        
        ### Averages
        
    rdd=sc.parallelize(range(100))
    
    #Mean
    sum=rdd.sum()
    n=rdd.count()
    mean=sum/n
    print (mean)
    
    #St Dev
    #ds=rdd.map(lambda x : pow(x-mean,2)).sum()/n)
    
    # Skewbess
    skew=n/((n-1)*(n-2))*rdd.map(lambda x : pow(x-mean,3)/pow(sd,3)).sum()
        
     # Kurtosis
    skew=rdd.map(lambda x : pow(x-mean,4)/pow(sd,4)).sum()/(1-n)   
        
        
    ## Covarance    
        
        
    ## Machine Leanring### PCA
    
    from pyspark.ml.feature import PCA
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.feature import VectorAssembler  
        
    assembler = VectorAssembler(inputCols=result.columns, outputCol="features")
        
    features = assembler.transform(result)
      
    features.rdd.map(lambda r : r.features).take(10)
        
    pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(features)    
        
    result_pca = model.transform(features).select("pcaFeatures")
    result_pca.show(truncate=False)    
#spark_1()

###  ###
def cement_strength_1():
    # Importing the neccessary libraries

    import pandas as pd
    import numpy as np
    import statistics as stats
    import os
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from datetime import datetime

    ### Regression with Keras ###
    concrete_data = pd.read_csv('Data/concrete_data.csv')
    print(concrete_data.shape)
    print(concrete_data.describe())
    
    # concrete_data.isnull().sum()
    # concrete_data_columns = concrete_data.columns
    
    predictors = concrete_data.iloc[:, 0:8] # all columns except Strength
    Y = concrete_data.iloc[:,8] # Strength column
    
    X = (predictors - predictors.mean()) / predictors.std()
    
    n_cols = X.shape[1] # number of predictors
    
    print("Y=")
    print(Y)
    print("X=")
    print(X)
    return Y
    
    def regression_model () :
        
        # Create the model
        model = Sequential()
        model.add(Dense(10, activation='relu', input_shape=(X.shape[1],)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
    
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def data_split() :
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        # Create a list containing X_train, X_test, Y_train, Y_test and return the list
        splits = [X_train, X_test, Y_train, Y_test] 
        return splits
    
    def predict() :
        return model.predict(X_test)
    
    def calculate_mse() :
        return mean_squared_error(Y_test,Y_predicted)

    # Split data into X_train, X_test, Y_train, Y_test
    X_train, X_test, Y_train, Y_test = data_split()
    
    # Create the model
    model = regression_model()
    
    # Fit the model on the train set
    model.fit(X_train, Y_train, validation_split=0.3, epochs=50)

    # Store the predictions in a variable Y_Predicted
    Y_predicted = predict()
    
    # Calculate the mean square error
    
    mse = calculate_mse()
    print('Mean Square Error (MSE) of the Baseline Model with Normalized Features is : ' , str(mse))

    # # Create the empty lists
    list_of_mse = []

    # Create the for loop to split the data, create, compile & fit model, evaluate & nake predictions, caluclate mse and store
    # in list_of_mse
    
    start_time = datetime.now() # Starting time of the for loop execution
    
    for i in range(50) :
        # Split the data into train and test set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        # Create and compile the regression model using the function regression_model as defined in TASK 1
        model = regression_model()
    
        # Fit the model on the train set
        print('\n\n\nTraining Model # ' , i+1 , '\n\n') # Print the Model Number that is being trained
        model.fit(X_train, Y_train, validation_split=0.3, epochs=100)
        print('\n')
        
        # Make prediction on the test set
        Y_predicted = model.predict(X_test)
        
        # Calculate the mean square error
        mse = mean_squared_error(Y_test, Y_predicted)
        
        # Add the mse to the list_of_mse list
        list_of_mse.append(mse)
    
    end_time = datetime.now() # Ending time of the for loop execution
    
    # Print time taken for fitting 50 models and calucating the Mean and Standard Deviation of MSE of 50 models
    print('\n\nTotal Execution Time : ' , format(end_time - start_time))
        
    
    # Calculate the Mean of the MSE of 50 models
    mean_of_mse = stats.mean(list_of_mse)
    
    # Calculate the Standard Deviation of the MSE of 50 models
    std_of_mse = stats.stdev(list_of_mse)
    
    # Print the Mean and Standard Deviation of MSE of 50 models
    print('\n\nMean of the MSE of 50 Models : ' , str(mean_of_mse))
    print('Standard Deviation of MSE of 50 Models : ' , str(std_of_mse))
#cement_strength_1()

###  ###
def other_keras_1():
    ### Classification with Keras ###
    import keras
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical
    import matplotlib.pyplot as plt
    
    # import the data
    from keras.datasets import mnist
    
    # read the data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    plt.imshow(X_train[0])
    
    # flatten images into one-dimensional vector
    
    num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector
    
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    
    # one hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    num_classes = y_test.shape[1]
    print(num_classes)
    
    # define classification model
    def classification_model():
        # create model
        model = Sequential()
        model.add(Dense(num_pixels, activation='sigmoid', input_shape=(num_pixels,)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        # compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # build the model
    model = classification_model()
    
    # fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, verbose=2)
    
    # evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    
    print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))        
    
    model.save('classification_model.h5')
    
    from keras.models import load_model
    
    pretrained_model = load_model('classification_model.h5')
    
    
    ## Convlutional NN ###
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical
    
    from keras.layers.convolutional import Conv2D # to add convolutional layers
    from keras.layers.convolutional import MaxPooling2D # to add pooling layers
    from keras.layers import Flatten # to flatten data for fully connected layers
    
    # import data
    from keras.datasets import mnist
    
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    
    X_train = X_train / 255 # normalize training data
    X_test = X_test / 255 # normalize test data
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    num_classes = y_test.shape[1] # number of categories
    
    def convolutional_model():
        
        # create model
        model = Sequential()
        model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        # compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
        return model
    
    # build the model
    model = convolutional_model()
    
    # fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
    
    # evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))    
#other_keras_1()

###  ###
def cement_images():
    #!wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip

    from keras.preprocessing.image import ImageDataGenerator
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.applications import VGG16
    from keras.applications.vgg16 import preprocess_input
    batch_size = 100
    num_classes = 2
    
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    
    train_generator = data_generator.flow_from_directory(
        'Data/concrete_data_week4/train',
        target_size=(224,224),
        batch_size=batch_size
    )
    
    validation_generator = data_generator.flow_from_directory(
        'Data/concrete_data_week4/valid',
        target_size=(224,224),
        batch_size=batch_size
    )
    
    model = Sequential()
    
    model.add(VGG16(include_top=False, pooling='avg', weights='imagenet'))
    
    # We will skip training the VGG16; it has already been trained!
    model.layers[0].trainable = False
    
    model.add(Dense(units=num_classes, activation='softmax'))
    
    model.summary()
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    steps_per_epoch_training = len(train_generator)
    steps_per_epoch_validation = len(validation_generator)
    num_epochs = 2
    
    fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch_training,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=steps_per_epoch_validation,
        verbose=1,
    )
    
    model.save('Data/classifier_vgg16_model.h5')

    from keras.models import load_model
    resnet_model=load_model('/content/drive/My Drive/Colab Notebooks/IBM_AI_Eng/classifier_resnet_model.h5')
    vgg16_model=load_model('/content/drive/My Drive/Colab Notebooks/IBM_AI_Eng/classifier_vgg16_model.h5')
    
    train_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    
    test_generator = train_data_generator.flow_from_directory(
        "concrete_data_week4/test",
        target_size=(224,224),
        shuffle=False
    )
    
    # ResNET
    results_resnet = resnet_model.evaluate_generator(test_generator)
    
    # VGG16
    results_vgg16 = vgg16_model.evaluate_generator(test_generator)
    
    # Make a Pandas DataFrame with the results
    import pandas as pd
    df = pd.DataFrame([results_resnet, results_vgg16], columns=['Loss', 'Accuracy'])
    df['Model'] = ['ResNet', 'VGG16']
    df = df.set_index('Model')
    
    df

    y_resnet = resnet_model.predict_generator(test_generator)
    
    y_vgg16 = vgg16_model.predict_generator(test_generator)
    
    import numpy as np
    def y_to_class(pred_y):
        return "Negative" if np.argmax(pred_y) == 0 else "Positive"
    
    # Resnet predictions
    for i in range(5):
        print(y_to_class(y_resnet[i]))
    
    # VGG16 predictions
    for i in range(5):
        print(y_to_class(y_vgg16[i]))
    
    # Classes from the test set
    for i in range(5):
        print(y_to_class(test_generator.next()[1][i]))
#cement_images()




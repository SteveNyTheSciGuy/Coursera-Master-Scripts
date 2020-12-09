# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 00:49:59 2020

@author: Steven

Copied, augmented, tweaked, created code from Coursera courses: 
    Deep Learning Specialization
        NN and Deep Learning
            log_reg_1()
            planar_class_1()
            build_DNN_1()
            apply_DNN_1()
        Improving DNN with Hyperparameters
            init_hp_1()
            reg_hp_1()
            grad_hp_1()
        Structuring ML Projs
        CNN
            step_cnn_1()
            app_cnn_1()
            res_cnn_1()
            art_gen_1()
            face_rec_1()
            tf1_tut_1()
        Sequence Models
            rnn_step_1()
            dino_isl_1()
            imp_jazz_1()
            word_vec_1()
            emojify_1()
            date_read_1()
            trig_word_1()
"""
import csv
import h5py
import json
#from lr_utils import load_dataset
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import numpy as np
import os
from PIL import Image
import random
from rnn_utils import *
import scipy
from scipy import ndimage
import sklearn
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from testCases_v2 import *
from utils import *
from rnn_utils import *
import zipfile



### Created log reg script to recognize cats ###
def log_reg_1():
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    
    
    # Example of a picture
    index = 25
    plt.imshow(train_set_x_orig[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
        
        
    ### START CODE HERE ### (≈ 3 lines of code)
    m_train = len(train_set_x_orig)
    m_test = len(test_set_x_orig)
    num_px = train_set_x_orig.shape[1]
    ### END CODE HERE ###
    
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))    
        
        
    # Reshape the training and test examples
    
    ### START CODE HERE ### (≈ 2 lines of code)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    ### END CODE HERE ###
    
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))    
        
        
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.   
        
        
    # GRADED FUNCTION: sigmoid
    
    def sigmoid(z):
        """
        Compute the sigmoid of z
    
        Arguments:
        z -- A scalar or numpy array of any size.
    
        Return:
        s -- sigmoid(z)
        """
    
        ### START CODE HERE ### (≈ 1 line of code)
        s = 1/(1+np.exp(-z))
        ### END CODE HERE ###
        
        return s    
        
        
    print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))    
        
        
    # GRADED FUNCTION: initialize_with_zeros
    
    def initialize_with_zeros(dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        
        ### START CODE HERE ### (≈ 1 line of code)
        w = np.zeros((dim,1),dtype='float')
        b = 0
        ### END CODE HERE ###
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        
        return w, b    
        
        
    dim = 2
    w, b = initialize_with_zeros(dim)
    print ("w = " + str(w))
    print ("b = " + str(b))    
        
        
    # GRADED FUNCTION: propagate
    
    def propagate(w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above
    
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    
        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        
        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
        """
        
        m = X.shape[1]
    
        # FORWARD PROPAGATION (FROM X TO COST)
        ### START CODE HERE ### (≈ 2 lines of code)
        A = sigmoid(np.dot(np.transpose(w),X)+b)                                    # compute activation
        cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))                               # compute cost
        ### END CODE HERE ###
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        ### START CODE HERE ### (≈ 2 lines of code)
        dw = (1/m)*np.dot(X,np.transpose(A-Y))
        db = (1/m)*np.sum(A-Y)
        ### END CODE HERE ###
    
        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                 "db": db}
        
        return grads, cost    
        
        
    w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
    grads, cost = propagate(w, b, X, Y)
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("cost = " + str(cost))    
        
        
    # GRADED FUNCTION: optimize
    
    def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        
        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """
        
        costs = []
        
        for i in range(num_iterations):
            
            
            # Cost and gradient calculation (≈ 1-4 lines of code)
            ### START CODE HERE ### 
            grads, cost = propagate(w, b, X, Y)
            ### END CODE HERE ###
            
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            # update rule (≈ 2 lines of code)
            ### START CODE HERE ###
            w = w-learning_rate*dw
            b = b-learning_rate*db
            ### END CODE HERE ###
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        
        params = {"w": w,
                  "b": b}
        
        grads = {"dw": dw,
                 "db": db}
        
        return params, grads, costs    
        
        
    params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
    
    print ("w = " + str(params["w"]))
    print ("b = " + str(params["b"]))
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))

    
    # GRADED FUNCTION: predict
    
    def predict(w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)
        
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        ### START CODE HERE ### (≈ 1 line of code)
        A = sigmoid(np.dot(np.transpose(w),X)+b)
        ### END CODE HERE ###
    
        for i in range(A.shape[1]):
    
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            ### START CODE HERE ### (≈ 4 lines of code)
            Y_prediction[0,i]=np.int(np.round(A[0,i]))
            ### END CODE HERE ###
        
        assert(Y_prediction.shape == (1, m))
        
        return Y_prediction


    w = np.array([[0.1124579],[0.23106775]])
    b = -0.3
    X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
    print ("predictions = " + str(predict(w, b, X)))
    
    
    # GRADED FUNCTION: model
    
    def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
        """
        Builds the logistic regression model by calling the function you've implemented previously
        
        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations
        
        Returns:
        d -- dictionary containing information about the model.
        """
        
        ### START CODE HERE ###
        
        # initialize parameters with zeros (≈ 1 line of code)
        w, b = initialize_with_zeros(len(X_train))
    
        # Gradient descent (≈ 1 line of code)
        parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = print_cost)
        
        # Retrieve parameters w and b from dictionary "parameters"
        w = parameters["w"]
        b = parameters["b"]
        
        # Predict test/train set examples (≈ 2 lines of code)
        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)
    
        ### END CODE HERE ###
    
        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
        
        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}
        
        return d


    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    
    
    # Example of a picture that was wrongly classified.
    index = 1
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
    
    
    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
    
    
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')
    
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
    
    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')
    
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    
    ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
    my_image = "my_image.jpg"   # change this to the name of your image file 
    ## END CODE HERE ##
    def testownimage(my_image):
        # We preprocess the image to fit your algorithm.
        fname = "images/" + my_image
        image = np.array(ndimage.imread(fname, flatten=False))
        image = image/255.
        my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
        my_predicted_image = predict(d["w"], d["b"], my_image)
        
        plt.imshow(image)
        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    testownimage(my_image)
#log_reg_1()

###  ###
def planar_class_1():
    # Package imports
    import sklearn.datasets
    import sklearn.linear_model
    from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
    
    
    np.random.seed(1) # set a seed so that the results are consistent
    
    X, Y = load_planar_dataset()
    
    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    
    ### START CODE HERE ### (≈ 3 lines of code)
    shape_X = X.shape
    shape_Y = Y.shape
    m = X.shape[1]  # training set size
    ### END CODE HERE ###
    
    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print ('I have m = %d training examples!' % (m))
    
    # Train the logistic regression classifier
    clf = sklearn.linear_model.LogisticRegressionCV();
    clf.fit(X.T, Y.T);
    
    # Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")
    
    # Print accuracy
    LR_predictions = clf.predict(X.T)
    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
           '% ' + "(percentage of correctly labelled datapoints)")
    
    # GRADED FUNCTION: layer_sizes
    
    def layer_sizes(X, Y):
        """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)
        
        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
        ### START CODE HERE ### (≈ 3 lines of code)
        n_x = shape_X[0] # size of input layer
        n_h = 4
        n_y = shape_Y[0] # size of output layer
        ### END CODE HERE ###
        return (n_x, n_h, n_y)
    
    X_assess, Y_assess = layer_sizes_test_case()
    (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))
    
    # GRADED FUNCTION: initialize_parameters
    
    def initialize_parameters(n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        
        np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
        
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = np.random.randn(n_h,n_x)*.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h)*.01
        b2 = np.zeros((n_y,1))
        ### END CODE HERE ###
        
        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters
    
    n_x, n_h, n_y = initialize_parameters_test_case()
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    
    # GRADED FUNCTION: forward_propagation
    
    def forward_propagation(X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)
        
        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        ### END CODE HERE ###
        
        # Implement Forward Propagation to calculate A2 (probabilities)
        ### START CODE HERE ### (≈ 4 lines of code)
        Z1 = np.dot(W1,X)+b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1)+b2
        A2 = sigmoid(Z2)
        ### END CODE HERE ###
        
        assert(A2.shape == (1, X.shape[1]))
        
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
        
        return A2, cache
    
    X_assess, parameters = forward_propagation_test_case()
    A2, cache = forward_propagation(X_assess, parameters)
    
    # Note: we use the mean here just to make sure that your output matches ours. 
    print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
    
    # GRADED FUNCTION: compute_cost
    
    def compute_cost(A2, Y, parameters):
        """
        Computes the cross-entropy cost given in equation (13)
        
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2
        [Note that the parameters argument is not used in this function, 
        but the auto-grader currently expects this parameter.
        Future version of this notebook will fix both the notebook 
        and the auto-grader so that `parameters` is not needed.
        For now, please include `parameters` in the function signature,
        and also when invoking this function.]
        
        Returns:
        cost -- cross-entropy cost given equation (13)
        
        """
        
        m = Y.shape[1] # number of example
    
        # Compute the cross-entropy cost
        ### START CODE HERE ### (≈ 2 lines of code)
        logprobs = np.multiply(np.log(A2),Y)
        cost = - np.sum(logprobs)
        ### END CODE HERE ###
        
        cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                        # E.g., turns [[17]] into 17 
        assert(isinstance(cost, float))
        
        return cost
    
    A2, Y_assess, parameters = compute_cost_test_case()
    
    print("cost = " + str(compute_cost(A2, Y_assess, parameters)))
    
    # GRADED FUNCTION: backward_propagation
    
    def backward_propagation(parameters, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.
        
        Arguments:
        parameters -- python dictionary containing our parameters 
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]
        
        # First, retrieve W1 and W2 from the dictionary "parameters".
        ### START CODE HERE ### (≈ 2 lines of code)
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        ### END CODE HERE ###
            
        # Retrieve also A1 and A2 from dictionary "cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1 = cache["A1"]
        A2 = cache["A2"]
        ### END CODE HERE ###
        
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
        dZ2 = A2-Y
        dW2 = (1/m)*np.dot(dZ2,A1.T)
        db2 = 1/m*np.sum(dZ2,axis=1,keepdims=True)
        dZ1 = np.dot(W2.T,dZ2)*(1 - np.power(A1, 2))
        dW1 = 1/m*np.dot(dZ1,X.T)
        db1 = 1/m*np.sum(dZ1,axis=1,keepdims=True)
        ### END CODE HERE ###
        
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        
        return grads
    
    parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
    
    grads = backward_propagation(parameters, cache, X_assess, Y_assess)
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dW2 = "+ str(grads["dW2"]))
    print ("db2 = "+ str(grads["db2"]))
    
    # GRADED FUNCTION: update_parameters
    
    def update_parameters(parameters, grads, learning_rate = 1.2):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients 
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        # Retrieve each parameter from the dictionary "parameters"
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        ### END CODE HERE ###
        
        # Retrieve each gradient from the dictionary "grads"
        ### START CODE HERE ### (≈ 4 lines of code)
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
        ## END CODE HERE ###
        
        # Update rule for each parameter
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = W1-learning_rate*dW1
        b1 = b1-learning_rate*db1
        W2 = W2-learning_rate*dW2
        b2 = b2-learning_rate*db2
        ### END CODE HERE ###
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters
    
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads)
    
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    # GRADED FUNCTION: nn_model
    
    def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        
        np.random.seed(3)
        n_x = layer_sizes(X, Y)[0]
        n_y = layer_sizes(X, Y)[2]
        
        # Initialize parameters
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = initialize_parameters(n_x, n_h, n_y)
        ### END CODE HERE ###
        
        # Loop (gradient descent)
    
        for i in range(0, num_iterations):
             
            ### START CODE HERE ### (≈ 4 lines of code)
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = forward_propagation(X, parameters)
            
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = compute_cost(A2, Y, parameters)
     
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = backward_propagation(parameters, cache, X, Y)
     
            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = update_parameters(parameters, grads, learning_rate = 1.2)
            
            ### END CODE HERE ###
            
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
    
        return parameters
    
    X_assess, Y_assess = nn_model_test_case()
    parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    # GRADED FUNCTION: predict
    
    def predict(parameters, X):
        """
        Using the learned parameters, predicts a class for each example in X
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        X -- input data of size (n_x, m)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        
        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        ### START CODE HERE ### (≈ 2 lines of code)
        A2, cache = forward_propagation(X, parameters)
        predictions = np.round(A2)
        ### END CODE HERE ###
        
        return predictions
    
    parameters, X_assess = predict_test_case()
    
    predictions = predict(parameters, X_assess)
    print("predictions mean = " + str(np.mean(predictions)))
    
    # Build a model with a n_h-dimensional hidden layer
    parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
    
    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    
    # Print accuracy
    predictions = predict(parameters, X)
    print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
    
    # This may take about 2 minutes to run
    
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations = 5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
        print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    
    # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    
    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}
    
    ### START CODE HERE ### (choose your dataset)
    dataset = "noisy_moons"
    ### END CODE HERE ###
    
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    
    # make blobs binary
    if dataset == "blobs":
        Y = Y%2
    
    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
#planar_class_1()

###  ###
def build_DNN_1():
    import numpy as np
    import matplotlib.pyplot as plt
    #from testCases_v4a import *
    from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
    
    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    np.random.seed(1)
    
    # GRADED FUNCTION: initialize_parameters
    
    def initialize_parameters(n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        parameters -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        
        np.random.seed(1)
        
        ### START CODE HERE ### (≈ 4 lines of code)
        W1 = np.random.randn(n_h,n_x)*.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h)*.01
        b2 = np.zeros((n_y,1))
        ### END CODE HERE ###
        
        assert(W1.shape == (n_h, n_x))
        assert(b1.shape == (n_h, 1))
        assert(W2.shape == (n_y, n_h))
        assert(b2.shape == (n_y, 1))
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters    
    
    parameters = initialize_parameters(3,2,1)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    # GRADED FUNCTION: initialize_parameters_deep
    
    def initialize_parameters_deep(layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network
    
        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            ### END CODE HERE ###
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
            
        return parameters
    
    parameters = initialize_parameters_deep([5,4,3])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    # GRADED FUNCTION: linear_forward
    
    def linear_forward(A, W, b):
        """
        Implement the linear part of a layer's forward propagation.
    
        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
    
        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        
        ### START CODE HERE ### (≈ 1 line of code)
        Z = np.dot(W,A)+b
        ### END CODE HERE ###
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache
    
    A, W, b = linear_forward_test_case()
    
    Z, linear_cache = linear_forward(A, W, b)
    print("Z = " + str(Z))
    
    # GRADED FUNCTION: linear_activation_forward
    
    def linear_activation_forward(A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
    
        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
            ### END CODE HERE ###
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
            ### END CODE HERE ###
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
    
        return A, cache
    
    A_prev, W, b = linear_activation_forward_test_case()
    
    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
    print("With sigmoid: A = " + str(A))
    
    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
    print("With ReLU: A = " + str(A))
    
    # GRADED FUNCTION: L_model_forward
    
    def L_model_forward(X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """
    
        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
           
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = linear_activation_forward(A_prev,  parameters['W' + str(l)],  parameters['b' + str(l)], activation="relu")
            caches.append(cache)
            ### END CODE HERE ###
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = linear_activation_forward(A, parameters['W' + str(L)],  parameters['b' + str(L)], activation="sigmoid")
        caches.append(cache)
        ### END CODE HERE ###
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches
    
    X, parameters = L_model_forward_test_case_2hidden()
    AL, caches = L_model_forward(X, parameters)
    print("AL = " + str(AL))
    print("Length of caches list = " + str(len(caches)))
    
    # GRADED FUNCTION: compute_cost
    
    def compute_cost(AL, Y):
        """
        Implement the cost function defined by equation (7).
    
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    
        Returns:
        cost -- cross-entropy cost
        """
        
        m = Y.shape[1]
    
        # Compute loss from aL and y.
        ### START CODE HERE ### (≈ 1 lines of code)
        cost = (-1/m)*np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),(1-Y)))
        ### END CODE HERE ###
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost
    
    Y, AL = compute_cost_test_case()
    
    print("cost = " + str(compute_cost(AL, Y)))
    
    # GRADED FUNCTION: linear_backward
    
    def linear_backward(dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)
    
        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
    
        ### START CODE HERE ### (≈ 3 lines of code)
        dW = (1/m)*np.dot(dZ,A_prev.T)
        db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        ### END CODE HERE ###
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db
    
    # Set up some test inputs
    dZ, linear_cache = linear_backward_test_case()
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    
    # GRADED FUNCTION: linear_activation_backward
    
    def linear_activation_backward(dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
            
        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
        
        return dA_prev, dW, db
    
    dAL, linear_activation_cache = linear_activation_backward_test_case()
    
    dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
    print ("sigmoid:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db) + "\n")
    
    dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
    print ("relu:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    
    # GRADED FUNCTION: L_model_backward
    
    def L_model_backward(AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
        ### END CODE HERE ###
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
        ### END CODE HERE ###
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[0]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,activation="relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###
    
        return grads
    
    AL, Y_assess, caches = L_model_backward_test_case()
    grads = L_model_backward(AL, Y_assess, caches)
    print_grads(grads)
    
    # GRADED FUNCTION: update_parameters
    
    def update_parameters(parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """
        
        L = len(parameters) // 2 # number of layers in the neural network
    
        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
        ### END CODE HERE ###
        return parameters
    
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)
    
    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))
#build_DNN_1()

###  ###
def apply_DNN_1():
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    from PIL import Image
    from scipy import ndimage
    #from dnn_app_utils_v3 import *
    
    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    np.random.seed(1)
    
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    
    # Example of a picture
    index = 10
    plt.imshow(train_x_orig[index])
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    
    # Explore your dataset 
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))
    
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    
    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))
    
    
    ### CONSTANTS DEFINING THE MODEL ####
    n_x = 12288     # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    
    # GRADED FUNCTION: two_layer_model
    
    def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        """
        Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations 
        
        Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
        """
        
        np.random.seed(1)
        grads = {}
        costs = []                              # to keep track of the cost
        m = X.shape[1]                           # number of examples
        (n_x, n_h, n_y) = layers_dims
        
        # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = initialize_parameters(n_x, n_h, n_y)
        ### END CODE HERE ###
        
        # Get W1, b1, W2 and b2 from the dictionary parameters.
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Loop (gradient descent)
    
        for i in range(0, num_iterations):
    
            # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
            ### START CODE HERE ### (≈ 2 lines of code)
            A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
            A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
            ### END CODE HERE ###
            
            # Compute cost
            ### START CODE HERE ### (≈ 1 line of code)
            cost = compute_cost(A2, Y)
            ### END CODE HERE ###
            
            # Initializing backward propagation
            dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
            
            # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
            ### START CODE HERE ### (≈ 2 lines of code)
            dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
            dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
            ### END CODE HERE ###
            
            # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2
            
            # Update parameters.
            ### START CODE HERE ### (approx. 1 line of code)
            parameters = update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
    
            # Retrieve W1, b1, W2, b2 from parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if print_cost and i % 100 == 0:
                costs.append(cost)
           
        # plot the cost
    
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters
    
    parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
    
    predictions_train = predict(train_x, train_y, parameters)
    
    predictions_test = predict(test_x, test_y, parameters)
    
    ### CONSTANTS ###
    layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
    
    # GRADED FUNCTION: L_layer_model
    
    def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
    
        np.random.seed(1)
        costs = []                         # keep track of cost
        
        # Parameters initialization. (≈ 1 line of code)
        ### START CODE HERE ###
        parameters = initialize_parameters_deep(layers_dims)
        ### END CODE HERE ###
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):
    
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = L_model_forward(X, parameters)
            ### END CODE HERE ###
            
            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            cost = compute_cost(AL, Y)
            ### END CODE HERE ###
        
            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = L_model_backward(AL, Y, caches)
            ### END CODE HERE ###
     
            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters
    
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
    
    pred_train = predict(train_x, train_y, parameters)
    
    pred_test = predict(test_x, test_y, parameters)
    
    print_mislabeled_images(classes, test_x, test_y, pred_test)
    
    def testownimage():
        ## START CODE HERE ##
        my_image = "my_image.jpg" # change this to the name of your image file 
        my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
        ## END CODE HERE ##
        
        fname = "images/" + my_image
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
        my_image = my_image/255.
        my_predicted_image = predict(my_image, my_label_y, parameters)
        
        plt.imshow(image)
        print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    testownimage()
#apply_DNN_1()

### A well chosen initialization can:Speed up the convergence of gradient descent
# Increase the odds of gradient descent converging to a lower training (and generalization) error ###
def init_hp_1():
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    import sklearn.datasets
    from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
    from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec
    
    plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    # load image dataset: blue/red dots in circles
    train_X, train_Y, test_X, test_Y = load_dataset()
    
    def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
        """
        Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
        learning_rate -- learning rate for gradient descent 
        num_iterations -- number of iterations to run gradient descent
        print_cost -- if True, print the cost every 1000 iterations
        initialization -- flag to choose which initialization to use ("zeros","random" or "he")
        
        Returns:
        parameters -- parameters learnt by the model
        """
            
        grads = {}
        costs = [] # to keep track of the loss
        m = X.shape[1] # number of examples
        layers_dims = [X.shape[0], 10, 5, 1]
        
        # Initialize parameters dictionary.
        if initialization == "zeros":
            parameters = initialize_parameters_zeros(layers_dims)
        elif initialization == "random":
            parameters = initialize_parameters_random(layers_dims)
        elif initialization == "he":
            parameters = initialize_parameters_he(layers_dims)
    
        # Loop (gradient descent)
    
        for i in range(0, num_iterations):
    
            # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
            a3, cache = forward_propagation(X, parameters)
            
            # Loss
            cost = compute_loss(a3, Y)
    
            # Backward propagation.
            grads = backward_propagation(X, Y, cache)
            
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)
            
            # Print the loss every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
                costs.append(cost)
                
        # plot the loss
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters
    
    # GRADED FUNCTION: initialize_parameters_zeros 
    
    def initialize_parameters_zeros(layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """
        
        parameters = {}
        L = len(layers_dims)            # number of layers in the network
        
        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            ### END CODE HERE ###
        return parameters
    
    parameters = initialize_parameters_zeros([3,2,1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    parameters = model(train_X, train_Y, initialization = "zeros")
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    
    print ("predictions_train = " + str(predictions_train))
    print ("predictions_test = " + str(predictions_test))
    
    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    
    # GRADED FUNCTION: initialize_parameters_random
    
    def initialize_parameters_random(layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """
        
        np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
        parameters = {}
        L = len(layers_dims)            # integer representing the number of layers
        
        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            ### END CODE HERE ###
    
        return parameters
    
    parameters = initialize_parameters_random([3, 2, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    parameters = model(train_X, train_Y, initialization = "random")
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    
    print (predictions_train)
    print (predictions_test)
    
    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    
    # GRADED FUNCTION: initialize_parameters_he
    
    def initialize_parameters_he(layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """
        
        np.random.seed(3)
        parameters = {}
        L = len(layers_dims) - 1 # integer representing the number of layers
         
        for l in range(1, L + 1):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            ### END CODE HERE ###
            
        return parameters
    
    parameters = initialize_parameters_he([2, 4, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    parameters = model(train_X, train_Y, initialization = "he")
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    
    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
#init_hp_1()

### overfitting can be a serious problem ###
def reg_hp_1():
    # import packages
    import numpy as np
    import matplotlib.pyplot as plt
    from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
    from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
    import sklearn
    import sklearn.datasets
    import scipy.io
    #from testCases import *
    
    plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    
    def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
        """
        Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
        learning_rate -- learning rate of the optimization
        num_iterations -- number of iterations of the optimization loop
        print_cost -- If True, print the cost every 10000 iterations
        lambd -- regularization hyperparameter, scalar
        keep_prob - probability of keeping a neuron active during drop-out, scalar.
        
        Returns:
        parameters -- parameters learned by the model. They can then be used to predict.
        """
            
        grads = {}
        costs = []                            # to keep track of the cost
        m = X.shape[1]                        # number of examples
        layers_dims = [X.shape[0], 20, 3, 1]
        
        # Initialize parameters dictionary.
        parameters = initialize_parameters(layers_dims)
    
        # Loop (gradient descent)
    
        for i in range(0, num_iterations):
    
            # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
            if keep_prob == 1:
                a3, cache = forward_propagation(X, parameters)
            elif keep_prob < 1:
                a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
            
            # Cost function
            if lambd == 0:
                cost = compute_cost(a3, Y)
            else:
                cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
                
            # Backward propagation.
            assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                                # but this assignment will only explore one at a time
            if lambd == 0 and keep_prob == 1:
                grads = backward_propagation(X, Y, cache)
            elif lambd != 0:
                grads = backward_propagation_with_regularization(X, Y, cache, lambd)
            elif keep_prob < 1:
                grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
            
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)
            
            # Print the loss every 10000 iterations
            if print_cost and i % 10000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
            if print_cost and i % 1000 == 0:
                costs.append(cost)
        
        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters
    
    parameters = model(train_X, train_Y)
    print ("On the training set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    
    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    
    # GRADED FUNCTION: compute_cost_with_regularization
    
    def compute_cost_with_regularization(A3, Y, parameters, lambd):
        """
        Implement the cost function with L2 regularization. See formula (2) above.
        
        Arguments:
        A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        parameters -- python dictionary containing parameters of the model
        
        Returns:
        cost - value of the regularized loss function (formula (2))
        """
        m = Y.shape[1]
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        W3 = parameters["W3"]
        
        cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
        
        ### START CODE HERE ### (approx. 1 line)
        L2_regularization_cost = (lambd/(2*m))*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))
        ### END CODER HERE ###
        
        cost = cross_entropy_cost + L2_regularization_cost
        
        return cost
    
    # GRADED FUNCTION: backward_propagation_with_regularization
    
    def backward_propagation_with_regularization(X, Y, cache, lambd):
        """
        Implements the backward propagation of our baseline model to which we added an L2 regularization.
        
        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation()
        lambd -- regularization hyperparameter, scalar
        
        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        
        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        
        ### START CODE HERE ### (approx. 1 line)
        dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m)*W3
        ### END CODE HERE ###
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
        
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        ### START CODE HERE ### (approx. 1 line)
        dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m)*W2
        ### END CODE HERE ###
        db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
        
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        ### START CODE HERE ### (approx. 1 line)
        dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m)*W1
        ### END CODE HERE ###
        db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                     "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                     "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients
    
    X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
    
    grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
    print ("dW1 = \n"+ str(grads["dW1"]))
    print ("dW2 = \n"+ str(grads["dW2"]))
    print ("dW3 = \n"+ str(grads["dW3"]))
    
    parameters = model(train_X, train_Y, lambd = 0.7)
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    
    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    
    # GRADED FUNCTION: forward_propagation_with_dropout
    
    def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
        """
        Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
        
        Arguments:
        X -- input dataset, of shape (2, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape (20, 2)
                        b1 -- bias vector of shape (20, 1)
                        W2 -- weight matrix of shape (3, 20)
                        b2 -- bias vector of shape (3, 1)
                        W3 -- weight matrix of shape (1, 3)
                        b3 -- bias vector of shape (1, 1)
        keep_prob - probability of keeping a neuron active during drop-out, scalar
        
        Returns:
        A3 -- last activation value, output of the forward propagation, of shape (1,1)
        cache -- tuple, information stored for computing the backward propagation
        """
        
        np.random.seed(1)
        
        # retrieve parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]
        
        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        Z1 = np.dot(W1, X) + b1
        A1 = relu(Z1)
        ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
        D1 = np.random.rand(A1.shape[0],A1.shape[1])      # Step 1: initialize matrix D1 = np.random.rand(..., ...)
        D1 = D1 < keep_prob                               # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
        A1 = np.multiply(A1, D1)                          # Step 3: shut down some neurons of A1
        A1 = A1 / keep_prob                               # Step 4: scale the value of neurons that haven't been shut down
        ### END CODE HERE ###
        Z2 = np.dot(W2, A1) + b2
        A2 = relu(Z2)
        ### START CODE HERE ### (approx. 4 lines)
        D2 = np.random.rand(A2.shape[0],A2.shape[1])      # Step 1: initialize matrix D2 = np.random.rand(..., ...)
        D2 = D2 < keep_prob                               # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
        A2 = np.multiply(A2, D2)                          # Step 3: shut down some neurons of A2
        A2 = A2 / keep_prob                               # Step 4: scale the value of neurons that haven't been shut down
        ### END CODE HERE ###
        Z3 = np.dot(W3, A2) + b3
        A3 = sigmoid(Z3)
        
        cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
        
        return A3, cache
    
    X_assess, parameters = forward_propagation_with_dropout_test_case()
    
    A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
    print ("A3 = " + str(A3))
    
    # GRADED FUNCTION: backward_propagation_with_dropout
    
    def backward_propagation_with_dropout(X, Y, cache, keep_prob):
        """
        Implements the backward propagation of our baseline model to which we added dropout.
        
        Arguments:
        X -- input dataset, of shape (2, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation_with_dropout()
        keep_prob - probability of keeping a neuron active during drop-out, scalar
        
        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        
        m = X.shape[1]
        (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        dW3 = 1./m * np.dot(dZ3, A2.T)
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
        dA2 = np.dot(W3.T, dZ3)
        ### START CODE HERE ### (≈ 2 lines of code)
        dA2 = np.multiply(dA2,D2)   # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
        dA2 = dA2/keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
        ### END CODE HERE ###
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1./m * np.dot(dZ2, A1.T)
        db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
        
        dA1 = np.dot(W2.T, dZ2)
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1 = np.multiply(dA1,D1)  # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
        dA1 = dA1/keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
        ### END CODE HERE ###
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1./m * np.dot(dZ1, X.T)
        db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                     "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                     "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients
    
    X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()
    
    gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob = 0.8)
    
    print ("dA1 = \n" + str(gradients["dA1"]))
    print ("dA2 = \n" + str(gradients["dA2"]))
    
    parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)
    
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    
    plt.title("Model with dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
#reg_hp_1()

###  ###
def grad_hp_1():
    # Packages
    import numpy as np
    #from testCases import *
    from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
    
    # GRADED FUNCTION: forward_propagation
    
    def forward_propagation(x, theta):
        """
        Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
        
        Arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well
        
        Returns:
        J -- the value of function J, computed using the formula J(theta) = theta * x
        """
        
        ### START CODE HERE ### (approx. 1 line)
        J = theta * x
        ### END CODE HERE ###
        
        return J
    
    x, theta = 2, 4
    J = forward_propagation(x, theta)
    print ("J = " + str(J))
    
    # GRADED FUNCTION: backward_propagation
    
    def backward_propagation(x, theta):
        """
        Computes the derivative of J with respect to theta (see Figure 1).
        
        Arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well
        
        Returns:
        dtheta -- the gradient of the cost with respect to theta
        """
        
        ### START CODE HERE ### (approx. 1 line)
        dtheta = x
        ### END CODE HERE ###
        
        return dtheta
    
    x, theta = 2, 4
    dtheta = backward_propagation(x, theta)
    print ("dtheta = " + str(dtheta))
    
    # GRADED FUNCTION: gradient_check
    
    def gradient_check(x, theta, epsilon = 1e-7):
        """
        Implement the backward propagation presented in Figure 1.
        
        Arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well
        epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
        
        Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient
        """
        
        # Compute gradapprox using left side of formula (1). epsilon is small enough, you don't need to worry about the limit.
        ### START CODE HERE ### (approx. 5 lines)
        thetaplus = theta + epsilon                           # Step 1
        thetaminus = theta - epsilon                          # Step 2
        J_plus = forward_propagation(x, thetaplus)            # Step 3
        J_minus = forward_propagation(x, thetaminus)           # Step 4
        gradapprox = (J_plus-J_minus)/(2*epsilon)             # Step 5
        ### END CODE HERE ###
    
        # Check if gradapprox is close enough to the output of backward_propagation()
        ### START CODE HERE ### (approx. 1 line)
        grad = backward_propagation(x, theta)
        ### END CODE HERE ###
        
        ### START CODE HERE ### (approx. 1 line)
        numerator = np.linalg.norm((grad-gradapprox))                     # Step 1'
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox) # Step 2'
        difference = numerator / denominator                            # Step 3'
        ### END CODE HERE ###
        
        if difference < 1e-7:
            print ("The gradient is correct!")
        else:
            print ("The gradient is wrong!")
        
        return difference
    
    x, theta = 2, 4
    difference = gradient_check(x, theta)
    print("difference = " + str(difference))
    
    def forward_propagation_n(X, Y, parameters):
        """
        Implements the forward propagation (and computes the cost) presented in Figure 3.
        
        Arguments:
        X -- training set for m examples
        Y -- labels for m examples 
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape (5, 4)
                        b1 -- bias vector of shape (5, 1)
                        W2 -- weight matrix of shape (3, 5)
                        b2 -- bias vector of shape (3, 1)
                        W3 -- weight matrix of shape (1, 3)
                        b3 -- bias vector of shape (1, 1)
        
        Returns:
        cost -- the cost function (logistic cost for one example)
        """
        
        # retrieve parameters
        m = X.shape[1]
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]
    
        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        Z1 = np.dot(W1, X) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = relu(Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = sigmoid(Z3)
    
        # Cost
        logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
        cost = 1./m * np.sum(logprobs)
        
        cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
        
        return cost, cache
    
    def backward_propagation_n(X, Y, cache):
        """
        Implement the backward propagation presented in figure 2.
        
        Arguments:
        X -- input datapoint, of shape (input size, 1)
        Y -- true "label"
        cache -- cache output from forward_propagation_n()
        
        Returns:
        gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
        """
        
        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        dW3 = 1./m * np.dot(dZ3, A2.T)
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
        
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1./m * np.dot(dZ2, A1.T) * 2
        db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
        
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1./m * np.dot(dZ1, X.T)
        db1 = 4./m * np.sum(dZ1, axis=1, keepdims = True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                     "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                     "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients
    
    def backward_propagation_n(X, Y, cache):
        """
        Implement the backward propagation presented in figure 2.
        
        Arguments:
        X -- input datapoint, of shape (input size, 1)
        Y -- true "label"
        cache -- cache output from forward_propagation_n()
        
        Returns:
        gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
        """
        
        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        dW3 = 1./m * np.dot(dZ3, A2.T)
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
        
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1./m * np.dot(dZ2, A1.T) * 2
        db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
        
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1./m * np.dot(dZ1, X.T)
        db1 = 4./m * np.sum(dZ1, axis=1, keepdims = True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                     "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                     "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients
    
    X, Y, parameters = gradient_check_n_test_case()
    
    cost, cache = forward_propagation_n(X, Y, parameters)
    gradients = backward_propagation_n(X, Y, cache)
    difference = gradient_check_n(parameters, gradients, X, Y)
#grad_hp_1()

### creates a cnn by scratch ###
def step_cnn_1():

    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    np.random.seed(1)
    
    # GRADED FUNCTION: zero_pad
    
    def zero_pad(X, pad):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
        as illustrated in Figure 1.
        
        Argument:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad -- integer, amount of padding around each image on vertical and horizontal dimensions
        
        Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
        
        ### START CODE HERE ### (≈ 1 line)
        X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))
        ### END CODE HERE ###
        
        return X_pad
    
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)
    print ("x.shape =\n", x.shape)
    print ("x_pad.shape =\n", x_pad.shape)
    print ("x[1,1] =\n", x[1,1])
    print ("x_pad[1,1] =\n", x_pad[1,1])
    
    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0,:,:,0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0,:,:,0])
    
    # GRADED FUNCTION: conv_single_step
    
    def conv_single_step(a_slice_prev, W, b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
        of the previous layer.
        
        Arguments:
        a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
        
        Returns:
        Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
        """
    
        ### START CODE HERE ### (≈ 2 lines of code)
        # Element-wise product between a_slice_prev and W. Do not add the bias yet.
        s = np.multiply(a_slice_prev,W)
        # Sum over all entries of the volume s.
        Z = np.sum(s, axis=None)
        # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
        Z = Z+float(b)
        ### END CODE HERE ###
    
        return Z
    
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)
    
    Z = conv_single_step(a_slice_prev, W, b)
    print("Z =", Z)
    
    # GRADED FUNCTION: conv_forward
    
    def conv_forward(A_prev, W, b, hparameters):
        """
        Implements the forward propagation for a convolution function
        
        Arguments:
        A_prev -- output activations of the previous layer, 
            numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
        b -- Biases, numpy array of shape (1, 1, 1, n_C)
        hparameters -- python dictionary containing "stride" and "pad"
            
        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
        
        ### START CODE HERE ###
        # Retrieve dimensions from A_prev's shape (≈1 line)  
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape (≈1 line)
        (f, f, n_C_prev, n_C) = W.shape
        
        # Retrieve information from "hparameters" (≈2 lines)
        stride = hparameters["stride"]
        pad = hparameters["pad"]
        
        # Compute the dimensions of the CONV output volume using the formula given above. 
        # Hint: use int() to apply the 'floor' operation. (≈2 lines)
        n_H = int((n_H_prev-f+2*pad)/stride+1)
        n_W = int((n_W_prev-f+2*pad)/stride+1)
        
        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros(((m, n_H, n_W, n_C)))
        
        # Create A_prev_pad by padding A_prev
        A_prev_pad = zero_pad(A_prev,pad)
        
        for i in range(m):               # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]               # Select ith training example's padded activation
            for h in range(n_H):           # loop over vertical axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                vert_start = h*stride
                vert_end = vert_start+f
                
                for w in range(n_W):       # loop over horizontal axis of the output volume
                    # Find the horizontal start and end of the current "slice" (≈2 lines)
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    for c in range(n_C):   # loop over channels (= #filters) of the output volume
                                            
                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                        
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                        weights = W[:,:,:,c]
                        biases = b[:,:,:,c]
                        Z[i, h, w, c] = conv_single_step(a_slice_prev,weights,biases)
                                            
        ### END CODE HERE ###
        
        # Making sure your output shape is correct
        assert(Z.shape == (m, n_H, n_W, n_C))
        
        # Save information in "cache" for the backprop
        cache = (A_prev, W, b, hparameters)
        
        return Z, cache
    
    np.random.seed(1)
    A_prev = np.random.randn(10,5,7,4)
    W = np.random.randn(3,3,4,8)
    b = np.random.randn(1,1,1,8)
    hparameters = {"pad" : 1,
                   "stride": 2}
    
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean =\n", np.mean(Z))
    print("Z[3,2,1] =\n", Z[3,2,1])
    print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])
    
    # GRADED FUNCTION: pool_forward
    
    def pool_forward(A_prev, hparameters, mode = "max"):
        """
        Implements the forward pass of the pooling layer
        
        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """
        
        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve hyperparameters from "hparameters"
        f = hparameters["f"]
        stride = hparameters["stride"]
        
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
        
        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))              
        
        ### START CODE HERE ###
        for i in range(m):                         # loop over the training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                vert_start = h*stride
                vert_end = vert_start+f
                
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    # Find the vertical start and end of the current "slice" (≈2 lines)
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    for c in range (n_C):            # loop over the channels of the output volume
                        
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = A_prev[i, vert_start : vert_end, horiz_start : horiz_end, c]
                        
                        # Compute the pooling operation on the slice. 
                        # Use an if statement to differentiate the modes. 
                        # Use np.max and np.mean.
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        
        ### END CODE HERE ###
        
        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, hparameters)
        
        # Making sure your output shape is correct
        assert(A.shape == (m, n_H, n_W, n_C))
        
        return A, cache
    
    # Case 1: stride of 1
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)
    hparameters = {"stride" : 1, "f": 3}
    
    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A.shape = " + str(A.shape))
    print("A =\n", A)
    print()
    A, cache = pool_forward(A_prev, hparameters, mode = "average")
    print("mode = average")
    print("A.shape = " + str(A.shape))
    print("A =\n", A)
    
    # Case 2: stride of 2
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)
    hparameters = {"stride" : 2, "f": 3}
    
    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A.shape = " + str(A.shape))
    print("A =\n", A)
    print()
    
    A, cache = pool_forward(A_prev, hparameters, mode = "average")
    print("mode = average")
    print("A.shape = " + str(A.shape))
    print("A =\n", A)
    
    def conv_backward(dZ, cache):
        """
        Implement the backward propagation for a convolution function
        
        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()
        
        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
              numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
              numpy array of shape (1, 1, 1, n_C)
        """
        
        ### START CODE HERE ###
        # Retrieve information from "cache"
        (A_prev, W, b, hparameters) = cache
        
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape
        
        # Retrieve information from "hparameters"
        stride = hparameters['stride']
        pad = hparameters['pad']
        
        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape
        
        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros(A_prev.shape)                           
        dW = np.zeros(W.shape)
        db = np.zeros(b.shape)
    
        # Pad A_prev and dA_prev
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
        
        for i in range(m):                       # loop over the training examples
            
            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            
            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = h * stride + f
                        horiz_start = w * stride
                        horiz_end = w * stride + f
                        
                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start : vert_end, horiz_start : horiz_end, : ]
    
                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[ i, h, w ,c]
                        dW[:,:,:,c] += a_slice * dZ[ i, h, w ,c]
                        db[:,:,:,c] += dZ[ i, h, w ,c]
                        
            # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        ### END CODE HERE ###
        
        # Making sure your output shape is correct
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        
        return dA_prev, dW, db
    
    # We'll run conv_forward to initialize the 'Z' and 'cache_conv",
    # which we'll use to test the conv_backward function
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hparameters = {"pad" : 2,
                   "stride": 2}
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    
    # Test conv_backward
    dA, dW, db = conv_backward(Z, cache_conv)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))
    
    def create_mask_from_window(x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.
        
        Arguments:
        x -- Array of shape (f, f)
        
        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        
        ### START CODE HERE ### (≈1 line)
        mask = (x == np.max(x))
        ### END CODE HERE ###
        
        return mask
    
    np.random.seed(1)
    x = np.random.randn(2,3)
    mask = create_mask_from_window(x)
    print('x = ', x)
    print("mask = ", mask)
    
    def distribute_value(dz, shape):
        """
        Distributes the input value in the matrix of dimension shape
        
        Arguments:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
        
        Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
        """
        
        ### START CODE HERE ###
        # Retrieve dimensions from shape (≈1 line)
        (n_H, n_W) = shape
        
        # Compute the value to distribute on the matrix (≈1 line)
        average = n_H * n_W
        
        # Create a matrix where every entry is the "average" value (≈1 line)
        a = dz / average * np.ones((n_H, n_W))
        ### END CODE HERE ###
        
        return a
    
    a = distribute_value(2, (2,2))
    print('distributed value =', a)
    
    def pool_backward(dA, cache, mode = "max"):
        """
        Implements the backward pass of the pooling layer
        
        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """
        
        ### START CODE HERE ###
        
        # Retrieve information from cache (≈1 line)
        (A_prev, hparameters) = cache
        
        # Retrieve hyperparameters from "hparameters" (≈2 lines)
        stride = hparameters['stride']
        f = hparameters['f']
        
        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        
        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(A_prev.shape)
        
        for i in range(m):                       # loop over the training examples
            
            # select training example from A_prev (≈1 line)
            a_prev = A_prev[i]
            
            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        
                        # Compute the backward propagation in both modes.
                        if mode == "max":
                            
                            # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                            a_prev_slice = a_prev[vert_start : vert_end, horiz_start : horiz_end, c]
                            # Create the mask from a_prev_slice (≈1 line)
                            mask = create_mask_from_window(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                            
                        elif mode == "average":
                            
                            # Get the value a from dA (≈1 line)
                            da = dA[i, h, w, c]
                            # Define the shape of the filter as fxf (≈1 line)
                            shape = (f, f)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
                            
        ### END CODE ###
        
        # Making sure your output shape is correct
        assert(dA_prev.shape == A_prev.shape)
        
        return dA_prev
    
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride" : 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)
    
    dA_prev = pool_backward(dA, cache, mode = "max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1])  
    print()
    dA_prev = pool_backward(dA, cache, mode = "average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1]) 
#step_cnn_1()

### uses tf1 ###
def app_cnn_1():
    import math
    from scipy import ndimage
    from tensorflow.python.framework import ops
    #from cnn_utils import *
    
    np.random.seed(1)
    
    # Loading the data (signs)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    
    # Example of a picture
    index = 6
    plt.imshow(X_train_orig[index])
    print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
    
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    conv_layers = {}
    
    # GRADED FUNCTION: create_placeholders
    
    def create_placeholders(n_H0, n_W0, n_C0, n_y):
        """
        Creates the placeholders for the tensorflow session.
        
        Arguments:
        n_H0 -- scalar, height of an input image
        n_W0 -- scalar, width of an input image
        n_C0 -- scalar, number of channels of the input
        n_y -- scalar, number of classes
            
        Returns:
        X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
        Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
        """
    
        ### START CODE HERE ### (≈2 lines)
        X = tf.placeholder(tf.float32,shape=[None,n_H0,n_W0,n_C0])
        Y = tf.placeholder(tf.float32,shape=[None,n_y])
        ### END CODE HERE ###
        
        return X, Y
    
    X, Y = create_placeholders(64, 64, 3, 6)
    print ("X = " + str(X))
    print ("Y = " + str(Y))
    
    # GRADED FUNCTION: initialize_parameters
    
    def initialize_parameters():
        """
        Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [4, 4, 3, 8]
                            W2 : [2, 2, 8, 16]
        Note that we will hard code the shape values in the function to make the grading simpler.
        Normally, functions should take values as inputs rather than hard coding.
        Returns:
        parameters -- a dictionary of tensors containing W1, W2
        """
        
        tf.set_random_seed(1)                              # so that your "random" numbers match ours
            
        ### START CODE HERE ### (approx. 2 lines of code)
        W1 = tf.get_variable("W1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W2 = tf.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
        ### END CODE HERE ###
    
        parameters = {"W1": W1,
                      "W2": W2}
        
        return parameters
    
    tf.reset_default_graph()
    with tf.Session() as sess_test:
        parameters = initialize_parameters()
        init = tf.global_variables_initializer()
        sess_test.run(init)
        print("W1[1,1,1] = \n" + str(parameters["W1"].eval()[1,1,1]))
        print("W1.shape: " + str(parameters["W1"].shape))
        print("\n")
        print("W2[1,1,1] = \n" + str(parameters["W2"].eval()[1,1,1]))
        print("W2.shape: " + str(parameters["W2"].shape))
    
    # GRADED FUNCTION: forward_propagation
    
    def forward_propagation(X, parameters):
        """
        Implements the forward propagation for the model:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
        
        Note that for simplicity and grading purposes, we'll hard-code some values
        such as the stride and kernel (filter) sizes. 
        Normally, functions should take these values as function parameters.
        
        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "W2"
                      the shapes are given in initialize_parameters
    
        Returns:
        Z3 -- the output of the last LINEAR unit
        """
        
        # Retrieve the parameters from the dictionary "parameters" 
        W1 = parameters['W1']
        W2 = parameters['W2']
        
        ### START CODE HERE ###
        # CONV2D: stride of 1, padding 'SAME'
        Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
        # RELU
        A1 = tf.nn.relu(Z1)
        # MAXPOOL: window 8x8, stride 8, padding 'SAME'
        P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')
        # CONV2D: filters W2, stride 1, padding 'SAME'
        Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
        # RELU
        A2 = tf.nn.relu(Z2)
        # MAXPOOL: window 4x4, stride 4, padding 'SAME'
        P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1], strides=[1,4,4,1],padding='SAME')
        # FLATTEN
        F = tf.contrib.layers.flatten(P2)
        # FULLY-CONNECTED without non-linear activation function (not not call softmax).
        # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
        Z3 = tf.contrib.layers.fully_connected(F,6,activation_fn=None)
        ### END CODE HERE ###
    
        return Z3
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        np.random.seed(1)
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
        print("Z3 = \n" + str(a))
    
    # GRADED FUNCTION: compute_cost 
    
    def compute_cost(Z3, Y):
        """
        Computes the cost
        
        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
        Y -- "true" labels vector placeholder, same shape as Z3
        
        Returns:
        cost - Tensor of the cost function
        """
        
        ### START CODE HERE ### (1 line of code)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
        ### END CODE HERE ###
        
        return cost
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        np.random.seed(1)
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
        print("cost = " + str(a))
    
    # GRADED FUNCTION: model
    
    def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
              num_epochs = 100, minibatch_size = 64, print_cost = True):
        """
        Implements a three-layer ConvNet in Tensorflow:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
        
        Arguments:
        X_train -- training set, of shape (None, 64, 64, 3)
        Y_train -- test set, of shape (None, n_y = 6)
        X_test -- training set, of shape (None, 64, 64, 3)
        Y_test -- test set, of shape (None, n_y = 6)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs
        
        Returns:
        train_accuracy -- real number, accuracy on the train set (X_train)
        test_accuracy -- real number, testing accuracy on the test set (X_test)
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        
        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
        seed = 3                                          # to keep results consistent (numpy seed)
        (m, n_H0, n_W0, n_C0) = X_train.shape             
        n_y = Y_train.shape[1]                            
        costs = []                                        # To keep track of the cost
        
        # Create Placeholders of the correct shape
        ### START CODE HERE ### (1 line)
        X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
        ### END CODE HERE ###
    
        # Initialize parameters
        ### START CODE HERE ### (1 line)
        parameters = initialize_parameters()
        ### END CODE HERE ###
        
        # Forward propagation: Build the forward propagation in the tensorflow graph
        ### START CODE HERE ### (1 line)
        Z3 = forward_propagation(X, parameters)
        ### END CODE HERE ###
        
        # Cost function: Add cost function to tensorflow graph
        ### START CODE HERE ### (1 line)
        cost = compute_cost(Z3, Y)
        ### END CODE HERE ###
        
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
        ### START CODE HERE ### (1 line)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        ### END CODE HERE ###
        
        # Initialize all the variables globally
        init = tf.global_variables_initializer()
         
        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:
            
            # Run the initialization
            sess.run(init)
            
            # Do the training loop
            for epoch in range(num_epochs):
    
                minibatch_cost = 0.
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
    
                for minibatch in minibatches:
    
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    """
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the optimizer and the cost.
                    # The feedict should contain a minibatch for (X,Y).
                    """
                    ### START CODE HERE ### (1 line)
                    _ , temp_cost = sess.run(fetches=[optimizer, cost], feed_dict={X: X_train, Y: Y_train})
                    ### END CODE HERE ###
                    
                    minibatch_cost += temp_cost / num_minibatches
                    
    
                # Print the cost every epoch
                if print_cost == True and epoch % 5 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                if print_cost == True and epoch % 1 == 0:
                    costs.append(minibatch_cost)
            
            
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
    
            # Calculate the correct predictions
            predict_op = tf.argmax(Z3, 1)
            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
            
            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print(accuracy)
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)
                    
            return train_accuracy, test_accuracy, parameters
    
    _, _, parameters = model(X_train, Y_train, X_test, Y_test)
    
    fname = "images/thumbs_up.jpg"
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64,64))
    plt.imshow(my_image)
#app_cnn_1()

### residual networks (tf1) ###
def res_cnn_1():
    import numpy as np
    from keras import layers
    from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
    from keras.models import Model, load_model
    from keras.preprocessing import image
    from keras.utils import layer_utils
    from keras.utils.data_utils import get_file
    from keras.applications.imagenet_utils import preprocess_input
    import pydot
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    from keras.utils import plot_model
    #from resnets_utils import *
    from keras.initializers import glorot_uniform
    import scipy.misc
    from matplotlib.pyplot import imshow
    
    import keras.backend as K
    K.set_image_data_format('channels_last')
    K.set_learning_phase(1)
    
    # GRADED FUNCTION: identity_block
    
    def identity_block(X, f, filters, stage, block):
        """
        Implementation of the identity block as defined in Figure 4
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        
        ### START CODE HERE ###
        
        # Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
        X = Activation('relu')(X)
    
        # Third component of main path (≈2 lines)
        X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid', name= conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3,name=bn_name_base+'2c')(X)
    
        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X_shortcut,X])
        X = Activation('relu')(X)
        
        ### END CODE HERE ###
        
        return X
    
    tf.reset_default_graph()
    
    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))
    
    # GRADED FUNCTION: convolutional_block
    
    def convolutional_block(X, f, filters, stage, block, s = 2):
        """
        Implementation of the convolutional block as defined in Figure 4
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used
        
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_shortcut = X
    
    
        ##### MAIN PATH #####
        # First component of main path 
        X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        
        ### START CODE HERE ###
    
        # Second component of main path (≈3 lines)
        X = Conv2D(F2, kernel_size=(f, f), strides = (1,1), padding='same',name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)
    
        # Third component of main path (≈2 lines)
        X = Conv2D(F3, kernel_size=(1, 1), strides = (1,1), padding='valid',name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv2D(F3, kernel_size=(1, 1), strides = (s,s), padding='valid',name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    
        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X,X_shortcut])
        X = Activation('relu')(X)
        
        ### END CODE HERE ###
        
        return X
    
    tf.reset_default_graph()
    
    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))
    
    # GRADED FUNCTION: ResNet50
    
    def ResNet50(input_shape = (64, 64, 3), classes = 6):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    
        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes
    
        Returns:
        model -- a Model() instance in Keras
        """
        
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)
    
        
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        
        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
        # Stage 2
        X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    
        ### START CODE HERE ###
    
        # Stage 3 (≈4 lines)
        X = convolutional_block(X,f=3,filters=[128,128,512],stage=3,block='a',s=2)
        X = identity_block(X,3,[128,128,512],stage=3,block='b')
        X = identity_block(X,3,[128,128,512],stage=3,block='c')
        X = identity_block(X,3,[128,128,512],stage=3,block='d')
    
        # Stage 4 (≈6 lines)
        X = convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block='a',s=2)
        X = identity_block(X,3,[256,256,1024],stage=4,block='b')
        X = identity_block(X,3,[256,256,1024],stage=4,block='c')
        X = identity_block(X,3,[256,256,1024],stage=4,block='d')
        X = identity_block(X,3,[256,256,1024],stage=4,block='e')
        X = identity_block(X,3,[256,256,1024],stage=4,block='f')
    
        # Stage 5 (≈3 lines)
        X = convolutional_block(X,f=3,filters=[512,512,2048],stage=5,block='a',s=2)
        X = identity_block(X,3,[512,512,2048],stage=5,block='b')
        X = identity_block(X,3,[512,512,2048],stage=5,block='c')
    
        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D(pool_size=(2,2),padding='same')(X)
        
        ### END CODE HERE ###
    
        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
        
        
        # Create model
        model = Model(inputs = X_input, outputs = X, name='ResNet50')
    
        return model
    
    model = ResNet50(input_shape = (64, 64, 3), classes = 6)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    
    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    
    model.fit(X_train, Y_train, epochs = 2, batch_size = 32)
    
    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    
    model = load_model('ResNet50.h5') 
    
    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    
    def testownimage():
        img_path = 'images/my_image.jpg'
        img = image.load_img(img_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255.0
        print('Input image shape:', x.shape)
        my_image = scipy.misc.imread(img_path)
        imshow(my_image)
        print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
        print(model.predict(x))
        
        model.summary()
        plot_model(model, to_file='model.png')
        SVG(model_to_dot(model).create(prog='dot', format='svg'))
    testownimage()
#res_cnn_1()

### Use object detection on a car detection dataset Deal with bounding boxes ###
def car_det_1():
    import argparse
    import os
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow
    import scipy.io
    import scipy.misc
    import numpy as np
    import pandas as pd
    import PIL
    import tensorflow as tf
    from keras import backend as K
    from keras.layers import Input, Lambda, Conv2D
    from keras.models import load_model, Model
    from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
    from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
    
    # GRADED FUNCTION: yolo_filter_boxes
    
    def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
        """Filters YOLO boxes by thresholding on object and class confidence.
        
        Arguments:
        box_confidence -- tensor of shape (19, 19, 5, 1)
        boxes -- tensor of shape (19, 19, 5, 4)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
        
        Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
        
        Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
        For example, the actual output size of scores would be (10,) if there are 10 boxes.
        """
        
        # Step 1: Compute box scores
        ### START CODE HERE ### (≈ 1 line)
        box_scores = np.multiply(box_confidence, box_class_probs)
        ### END CODE HERE ###
        
        # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
        ### START CODE HERE ### (≈ 2 lines)
        box_classes = K.argmax(box_scores, axis=-1)
        box_class_scores = K.max(box_scores,axis=-1)
        ### END CODE HERE ###
        
        # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
        # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
        ### START CODE HERE ### (≈ 1 line)
        filtering_mask = box_class_scores >= threshold
        ### END CODE HERE ###
        
        # Step 4: Apply the mask to box_class_scores, boxes and box_classes
        ### START CODE HERE ### (≈ 3 lines)
        scores = tf.boolean_mask(box_class_scores, filtering_mask)
        boxes = tf.boolean_mask(boxes, filtering_mask)
        classes = tf.boolean_mask(box_classes, filtering_mask)
        ### END CODE HERE ###
        
        return scores, boxes, classes
    
    with tf.Session() as test_a:
        box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
        boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
        box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
        scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.shape))
        print("boxes.shape = " + str(boxes.shape))
        print("classes.shape = " + str(classes.shape))
    
    # GRADED FUNCTION: iou
    
    def iou(box1, box2):
        """Implement the intersection over union (IoU) between box1 and box2
        
        Arguments:
        box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
        box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
        """
    
        # Assign variable names to coordinates for clarity
        (box1_x1, box1_y1, box1_x2, box1_y2) = box1
        (box2_x1, box2_y1, box2_x2, box2_y2) = box2
        
        # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
        ### START CODE HERE ### (≈ 7 lines)
        xi1 = np.maximum(box1_x1,box2_x1)
        yi1 = np.maximum(box1_y1,box2_y1)
        xi2 = np.minimum(box1_x2,box2_x2)
        yi2 = np.minimum(box1_y2,box2_y2)
        inter_width = max((xi2 - xi1), 0)
        inter_height = max((yi2 - yi1), 0)
        inter_area = inter_width*inter_height
        ### END CODE HERE ###    
    
        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        ### START CODE HERE ### (≈ 3 lines)
        box1_area =  (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
        ### END CODE HERE ###
        
        # compute the IoU
        ### START CODE HERE ### (≈ 1 line)
        iou = inter_area / union_area
        ### END CODE HERE ###
        
        return iou
    
    ## Test case 1: boxes intersect
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4) 
    print("iou for intersecting boxes = " + str(iou(box1, box2)))
    
    ## Test case 2: boxes do not intersect
    box1 = (1,2,3,4)
    box2 = (5,6,7,8)
    print("iou for non-intersecting boxes = " + str(iou(box1,box2)))
    
    ## Test case 3: boxes intersect at vertices only
    box1 = (1,1,2,2)
    box2 = (2,2,3,3)
    print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))
    
    ## Test case 4: boxes intersect at edge only
    box1 = (1,1,3,3)
    box2 = (2,3,3,4)
    print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))
    
    # GRADED FUNCTION: yolo_non_max_suppression
    
    def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
        """
        Applies Non-max suppression (NMS) to set of boxes
        
        Arguments:
        scores -- tensor of shape (None,), output of yolo_filter_boxes()
        boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
        classes -- tensor of shape (None,), output of yolo_filter_boxes()
        max_boxes -- integer, maximum number of predicted boxes you'd like
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
        
        Returns:
        scores -- tensor of shape (, None), predicted score for each box
        boxes -- tensor of shape (4, None), predicted box coordinates
        classes -- tensor of shape (, None), predicted class for each box
        
        Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
        function will transpose the shapes of scores, boxes, classes. This is made for convenience.
        """
        
        max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
        K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
        
        # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
        ### START CODE HERE ### (≈ 1 line)
        nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)
        ### END CODE HERE ###
        
        # Use K.gather() to select only nms_indices from scores, boxes and classes
        ### START CODE HERE ### (≈ 3 lines)
        scores =  K.gather(scores, nms_indices)
        boxes = K.gather(boxes, nms_indices)
        classes = K.gather(classes, nms_indices)
        ### END CODE HERE ###
        
        return scores, boxes, classes
    
    with tf.Session() as test_b:
        scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
        boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
        classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
        scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.eval().shape))
        print("boxes.shape = " + str(boxes.eval().shape))
        print("classes.shape = " + str(classes.eval().shape))
    
    # GRADED FUNCTION: yolo_eval
    
    def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
        """
        Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
        
        Arguments:
        yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                        box_confidence: tensor of shape (None, 19, 19, 5, 1)
                        box_xy: tensor of shape (None, 19, 19, 5, 2)
                        box_wh: tensor of shape (None, 19, 19, 5, 2)
                        box_class_probs: tensor of shape (None, 19, 19, 5, 80)
        image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
        max_boxes -- integer, maximum number of predicted boxes you'd like
        score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
        
        Returns:
        scores -- tensor of shape (None, ), predicted score for each box
        boxes -- tensor of shape (None, 4), predicted box coordinates
        classes -- tensor of shape (None,), predicted class for each box
        """
        
        ### START CODE HERE ### 
        
        # Retrieve outputs of the YOLO model (≈1 line)
        box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    
        # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
        boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
        # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
        scores, boxes, classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold=score_threshold)
        
        # Scale boxes back to original image shape.
        boxes = scale_boxes(boxes, image_shape)
    
        # Use one of the functions you've implemented to perform Non-max suppression with 
        # maximum number of boxes set to max_boxes and a threshold of iou_threshold (≈1 line)
        scores, boxes, classes = yolo_non_max_suppression(scores,boxes,classes,max_boxes=max_boxes,iou_threshold=iou_threshold)
        
        ### END CODE HERE ###
        
        return scores, boxes, classes
    
    with tf.Session() as test_b:
        yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                        tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                        tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                        tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
        scores, boxes, classes = yolo_eval(yolo_outputs)
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.eval().shape))
        print("boxes.shape = " + str(boxes.eval().shape))
        print("classes.shape = " + str(classes.eval().shape))
    
    sess = K.get_session()
    
    class_names = read_classes("model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    image_shape = (720., 1280.)    
    
    yolo_model = load_model("model_data/yolo.h5")
    
    yolo_model.summary()
    
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    
    def predict(sess, image_file):
        """
        Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the predictions.
        
        Arguments:
        sess -- your tensorflow/Keras session containing the YOLO graph
        image_file -- name of an image stored in the "images" folder.
        
        Returns:
        out_scores -- tensor of shape (None, ), scores of the predicted boxes
        out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
        out_classes -- tensor of shape (None, ), class index of the predicted boxes
        
        Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
        """
    
        # Preprocess your image
        image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    
        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        ### START CODE HERE ### (≈ 1 line)
        out_scores, out_boxes, out_classes = sess.run(fetches=[tensor1,tensor2,tensor3],
           feed_dict={yolo_model.input: the_input_variable,
                      K.learning_phase():0})
        ### END CODE HERE ###
    
        # Print predictions info
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))
        # Generate colors for drawing bounding boxes.
        colors = generate_colors(class_names)
        # Draw bounding boxes on the image file
        draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
        # Save the predicted bounding box on the image
        image.save(os.path.join("out", image_file), quality=90)
        # Display the results in the notebook
        output_image = scipy.misc.imread(os.path.join("out", image_file))
        imshow(output_image)
        
        return out_scores, out_boxes, out_classes
    
    out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
#car_det_1()

### Implement the neural style transfer algorithm Generate novel artistic images using your algorithm ###
def art_gen_1():
    import os
    import sys
    import scipy.io
    import scipy.misc
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow
    from PIL import Image
    #from nst_utils import *
    import numpy as np
    import tensorflow as tf
    import pprint
    
    pp = pprint.PrettyPrinter(indent=4)
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    pp.pprint(model)
    
    content_image = scipy.misc.imread("images/louvre.jpg")
    imshow(content_image);
    
    # GRADED FUNCTION: compute_content_cost
    
    def compute_content_cost(a_C, a_G):
        """
        Computes the content cost
        
        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
        
        Returns: 
        J_content -- scalar that you compute using equation 1 above.
        """
        
        ### START CODE HERE ###
        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        # Reshape a_C and a_G (≈2 lines)
        a_C_unrolled = tf.transpose(tf.reshape(a_C,[n_H*n_W, n_C]))
        a_G_unrolled = tf.transpose(tf.reshape(a_G,[n_H*n_W, n_C]))
        
        # compute the cost with tensorflow (≈1 line)
        J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))
        ### END CODE HERE ###
        
        return J_content
    
    tf.reset_default_graph()
    
    with tf.Session() as test:
        tf.set_random_seed(1)
        a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = compute_content_cost(a_C, a_G)
        print("J_content = " + str(J_content.eval()))
    
    style_image = scipy.misc.imread("images/monet_800600.jpg")
    imshow(style_image);
    
    # GRADED FUNCTION: gram_matrix
    
    def gram_matrix(A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)
        
        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """
        
        ### START CODE HERE ### (≈1 line)
        GA = tf.matmul(A,tf.transpose(A))
        ### END CODE HERE ###
        
        return GA
    
    tf.reset_default_graph()
    
    with tf.Session() as test:
        tf.set_random_seed(1)
        A = tf.random_normal([3, 2*1], mean=1, stddev=4)
        GA = gram_matrix(A)
        
        print("GA = \n" + str(GA.eval()))
    
    # GRADED FUNCTION: compute_layer_style_cost
    
    def compute_layer_style_cost(a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
        
        Returns: 
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        
        ### START CODE HERE ###
        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
        a_S = tf.transpose(tf.reshape(a_S,[n_H*n_W,n_C]))
        a_G = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))
    
        # Computing gram_matrices for both images S and G (≈2 lines)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
    
        # Computing the loss (≈1 line)
        J_style_layer = (1/(4*n_C*n_C*(n_H*n_W)*(n_H*n_W)))*(tf.reduce_sum(tf.square(tf.subtract(GS,GG))))
        
        ### END CODE HERE ###
        
        return J_style_layer
    
    tf.reset_default_graph()
    
    with tf.Session() as test:
        tf.set_random_seed(1)
        a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        
        print("J_style_layer = " + str(J_style_layer.eval()))
    
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
    
    def compute_style_cost(model, STYLE_LAYERS):
        """
        Computes the overall style cost from several chosen layers
        
        Arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them
        
        Returns: 
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        
        # initialize the overall style cost
        J_style = 0
    
        for layer_name, coeff in STYLE_LAYERS:
    
            # Select the output tensor of the currently selected layer
            out = model[layer_name]
    
            # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
            a_S = sess.run(out)
    
            # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
            # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
            # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
            a_G = out
            
            # Compute style_cost for the current layer
            J_style_layer = compute_layer_style_cost(a_S, a_G)
    
            # Add coeff * J_style_layer of this layer to overall style cost
            J_style += coeff * J_style_layer
    
        return J_style
    
    # GRADED FUNCTION: total_cost
    
    def total_cost(J_content, J_style, alpha = 10, beta = 40):
        """
        Computes the total cost function
        
        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost
        
        Returns:
        J -- total cost as defined by the formula above.
        """
        
        ### START CODE HERE ### (≈1 line)
        J = alpha*J_content+beta*J_style
        ### END CODE HERE ###
        
        return J
    
    tf.reset_default_graph()
    
    with tf.Session() as test:
        np.random.seed(3)
        J_content = np.random.randn()    
        J_style = np.random.randn()
        J = total_cost(J_content, J_style)
        print("J = " + str(J))
    
    # Reset the graph
    tf.reset_default_graph()
    
    # Start interactive session
    sess = tf.InteractiveSession()
    
    content_image = scipy.misc.imread("images/louvre_small.jpg")
    content_image = reshape_and_normalize_image(content_image)
    
    style_image = scipy.misc.imread("images/monet.jpg")
    style_image = reshape_and_normalize_image(style_image)
    
    generated_image = generate_noise_image(content_image)
    imshow(generated_image[0]);
    
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    
    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))
    
    # Select the output tensor of layer conv4_2
    out = model['conv4_2']
    
    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)
    
    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out
    
    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)
    
    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))
    
    # Compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS)
    
    ### START CODE HERE ### (1 line)
    J = total_cost(J_content,J_style,alpha=10,beta=40)
    ### END CODE HERE ###
    
    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)
    
    # define train_step (1 line)
    train_step = optimizer.minimize(J)
    
    def model_nn(sess, input_image, num_iterations = 200):
        
        # Initialize global variables (you need to run the session on the initializer)
        ### START CODE HERE ### (1 line)
        sess.run(tf.global_variables_initializer())
        ### END CODE HERE ###
        
        # Run the noisy input image (initial generated image) through the model. Use assign().
        ### START CODE HERE ### (1 line)
        sess.run(model["input"].assign(input_image))
        ### END CODE HERE ###
        
        for i in range(num_iterations):
        
            # Run the session on the train_step to minimize the total cost
            ### START CODE HERE ### (1 line)
            sess.run(train_step)
            ### END CODE HERE ###
            
            # Compute the generated image by running the session on the current model['input']
            ### START CODE HERE ### (1 line)
            generated_image = sess.run(model["input"])
            ### END CODE HERE ###
    
            # Print every 20 iteration.
            if i%20 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                
                # save current generated image in the "/output" directory
                save_image("output/" + str(i) + ".png", generated_image)
        
        # save last generated image
        save_image('output/generated_image.jpg', generated_image)
        
        return generated_image
    
    model_nn(sess, generated_image)
#art_gen_1()

""" Face Verification - "is this the claimed person?".
    Face Recognition - "who is this person?".
    Implement the triplet loss function
    Use a pretrained model to map face images into 128-dimensional encodings
    Use these encodings to perform face verification and face recognition """
def face_rec_1():
    from keras.models import Sequential
    from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization
    from keras.layers.pooling import MaxPooling2D, AveragePooling2D
    from keras.layers.merge import Concatenate
    from keras.layers.core import Lambda, Flatten, Dense
    from keras.initializers import glorot_uniform
    from keras.engine.topology import Layer
    from keras import backend as K
    K.set_image_data_format('channels_first')
    import cv2
    import os
    import numpy as np
    from numpy import genfromtxt
    import pandas as pd
    import tensorflow as tf
    #from fr_utils import *
    #from inception_blocks_v2 import *
    
    np.set_printoptions(threshold=np.nan)
    
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    
    print("Total Params:", FRmodel.count_params())
    
    # GRADED FUNCTION: triplet_loss
    
    def triplet_loss(y_true, y_pred, alpha = 0.2):
        """
        Implementation of the triplet loss as defined by formula (3)
        
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor images, of shape (None, 128)
                positive -- the encodings for the positive images, of shape (None, 128)
                negative -- the encodings for the negative images, of shape (None, 128)
        
        Returns:
        loss -- real number, value of the loss
        """
        
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        
        ### START CODE HERE ### (≈ 4 lines)
        # Step 1: Compute the (encoding) distance between the anchor and the positive
        pos_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor, positive)),axis=-1)
        # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
        neg_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor, negative)),axis=-1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
        ### END CODE HERE ###
        
        return loss
    
    with tf.Session() as test:
        tf.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
                  tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
                  tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
        loss = triplet_loss(y_true, y_pred)
        
        print("loss = " + str(loss.eval()))
    
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    
    database = {}
    database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
    database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
    database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
    database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
    database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
    database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
    database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
    database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
    database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
    database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
    database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
    database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
    
    # GRADED FUNCTION: verify
    
    def verify(image_path, identity, database, model):
        """
        Function that verifies if the person on the "image_path" image is "identity".
        
        Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras
        
        Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
        """
        
        ### START CODE HERE ###
        
        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        encoding = img_to_encoding(image_path,model)
        
        # Step 2: Compute distance with identity's image (≈ 1 line)
        dist = np.linalg.norm(encoding-database[identity])
        
        # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
        if None:
            print("It's " + str(identity) + ", welcome in!")
            door_open = True
        else:
            print("It's not " + str(identity) + ", please go away")
            door_open = False
            
        ### END CODE HERE ###
            
        return dist, door_open
    
    verify("images/camera_0.jpg", "younes", database, FRmodel)
    
    verify("images/camera_2.jpg", "kian", database, FRmodel)
    
    # GRADED FUNCTION: who_is_it
    
    def who_is_it(image_path, database, model):
        """
        Implements face recognition for the office by finding who is the person on the image_path image.
        
        Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras
        
        Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
        """
        
        ### START CODE HERE ### 
        
        ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
        encoding = img_to_encoding(image_path,model)
        
        ## Step 2: Find the closest encoding ##
        
        # Initialize "min_dist" to a large value, say 100 (≈1 line)
        min_dist = 100
        
        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():
            
            # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
            dist = np.linalg.norm(encoding-db_enc)
    
            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist<min_dist:
                min_dist = dist
                identity = name
    
        ### END CODE HERE ###
        
        if min_dist > 0.7:
            print("Not in the database.")
        else:
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))
            
        return min_dist, identity
    
    who_is_it("images/camera_0.jpg", database, FRmodel)
#face_rec_1()

"""Initialize variables,Start your own session, Train algorithms,
    Implement a Neural Network (TF1) """
def tf1_tut_1():
    import math
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.python.framework import ops
    from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
    
    np.random.seed(1)
    
    y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
    y = tf.constant(39, name='y')                    # Define y. Set to 39
    
    loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss
    
    init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                     # the loss variable will be initialized and ready to be computed
    with tf.Session() as session:                    # Create a session and print the output
        session.run(init)                            # Initializes the variables
        print(session.run(loss))                     # Prints the loss
    
    a = tf.constant(2)
    b = tf.constant(10)
    c = tf.multiply(a,b)
    print(c)
    
    sess = tf.Session()
    print(sess.run(c))
    
    # Change the value of x in the feed_dict
    
    x = tf.placeholder(tf.int64, name = 'x')
    print(sess.run(2 * x, feed_dict = {x: 3}))
    sess.close()
    
    # GRADED FUNCTION: linear_function
    
    def linear_function():
        """
        Implements a linear function: 
                Initializes X to be a random tensor of shape (3,1)
                Initializes W to be a random tensor of shape (4,3)
                Initializes b to be a random tensor of shape (4,1)
        Returns: 
        result -- runs the session for Y = WX + b 
        """
        
        np.random.seed(1)
        
        """
        Note, to ensure that the "random" numbers generated match the expected results,
        please create the variables in the order given in the starting code below.
        (Do not re-arrange the order).
        """
        ### START CODE HERE ### (4 lines of code)
        X = tf.constant(np.random.randn(3,1), name = "X")
        W = tf.constant(np.random.randn(4,3), name = "W")
        b = tf.constant(np.random.randn(4,1), name = "b")
        Y = tf.add(tf.matmul(W,X),b)
        ### END CODE HERE ### 
        
        # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
        
        ### START CODE HERE ###
        sess = tf.Session()
        result = sess.run(Y)
        ### END CODE HERE ### 
        
        # close the session 
        sess.close()
    
        return result
    
    print( "result = \n" + str(linear_function()))
    
    # GRADED FUNCTION: sigmoid
    
    def sigmoid(z):
        """
        Computes the sigmoid of z
        
        Arguments:
        z -- input value, scalar or vector
        
        Returns: 
        results -- the sigmoid of z
        """
        
        ### START CODE HERE ### ( approx. 4 lines of code)
        # Create a placeholder for x. Name it 'x'.
        x = tf.placeholder(tf.float32, name = "x")
    
        # compute sigmoid(x)
        sigmoid = tf.sigmoid(x)
    
        # Create a session, and run it. Please use the method 2 explained above. 
        # You should use a feed_dict to pass z's value to x. 
        with tf.Session() as sess: 
            # Run session and call the output "result"
            result = sess.run(sigmoid, feed_dict = {x: z})
    
        ### END CODE HERE ###
        
        return result
    
    print ("sigmoid(0) = " + str(sigmoid(0)))
    print ("sigmoid(12) = " + str(sigmoid(12)))
    
    # GRADED FUNCTION: cost
    
    def cost(logits, labels):
        """
        Computes the cost using the sigmoid cross entropy
        
        Arguments:
        logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
        labels -- vector of labels y (1 or 0) 
        
        Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
        in the TensorFlow documentation. So logits will feed into z, and labels into y. 
        
        Returns:
        cost -- runs the session of the cost (formula (2))
        """
        
        ### START CODE HERE ### 
        
        # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
        z = tf.placeholder(tf.float32, name = "z")
        y = tf.placeholder(tf.float32, name = "y")
        
        # Use the loss function (approx. 1 line)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z,  labels = y)
        
        # Create a session (approx. 1 line). See method 1 above.
        sess = tf.Session()
        
        # Run the session (approx. 1 line).
        cost = sess.run(cost,feed_dict = {z: logits, y: labels})
        
        # Close the session (approx. 1 line). See method 1 above.
        sess.close() # Close the session
        
        ### END CODE HERE ###
        
        return cost
    
    logits = np.array([0.2,0.4,0.7,0.9])
    
    cost = cost(logits, np.array([0,0,1,1]))
    print ("cost = " + str(cost))
    
    # GRADED FUNCTION: one_hot_matrix
    
    def one_hot_matrix(labels, C):
        """
        Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                         corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                         will be 1. 
                         
        Arguments:
        labels -- vector containing the labels 
        C -- number of classes, the depth of the one hot dimension
        
        Returns: 
        one_hot -- one hot matrix
        """
        
        ### START CODE HERE ###
        
        # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
        C = tf.constant(C, name = "C")
        
        # Use tf.one_hot, be careful with the axis (approx. 1 line)
        one_hot_matrix = tf.one_hot(labels, C, axis=0)
        
        # Create the session (approx. 1 line)
        sess = tf.Session()
        
        # Run the session (approx. 1 line)
        one_hot = sess.run(one_hot_matrix)
        
        # Close the session (approx. 1 line). See method 1 above.
        sess.close() # Close the session
        
        ### END CODE HERE ###
        
        return one_hot
    
    labels = np.array([1,2,3,0,2,1])
    one_hot = one_hot_matrix(labels, C = 4)
    print ("one_hot = \n" + str(one_hot))
    
    # GRADED FUNCTION: ones
    
    def ones(shape):
        """
        Creates an array of ones of dimension shape
        
        Arguments:
        shape -- shape of the array you want to create
            
        Returns: 
        ones -- array containing only ones
        """
        
        ### START CODE HERE ###
        
        # Create "ones" tensor using tf.ones(...). (approx. 1 line)
        ones = tf.ones(shape)
        
        # Create the session (approx. 1 line)
        sess = tf.Session()
        
        # Run the session to compute 'ones' (approx. 1 line)
        ones = sess.run(ones)
        
        # Close the session (approx. 1 line). See method 1 above.
        sess.close() # Close the session
        
        ### END CODE HERE ###
        return ones
    
    print ("ones = " + str(ones([3])))
    
    # Loading the dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    
    # Example of a picture
    index = 0
    plt.imshow(X_train_orig[index])
    print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
    
    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    
    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_test.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    
    # GRADED FUNCTION: create_placeholders
    
    def create_placeholders(n_x, n_y):
        """
        Creates the placeholders for the tensorflow session.
        
        Arguments:
        n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
        n_y -- scalar, number of classes (from 0 to 5, so -> 6)
        
        Returns:
        X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
        Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"
        
        Tips:
        - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
          In fact, the number of examples during test/train is different.
        """
    
        ### START CODE HERE ### (approx. 2 lines)
        X = tf.placeholder(tf.float32, shape=(n_x, None), name = "X")
        Y = tf.placeholder(tf.float32, shape=(n_y, None), name = "Y")
        ### END CODE HERE ###
        
        return X, Y
    
    X, Y = create_placeholders(12288, 6)
    print ("X = " + str(X))
    print ("Y = " + str(Y))
    
    # GRADED FUNCTION: initialize_parameters
    
    def initialize_parameters():
        """
        Initializes parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [25, 12288]
                            b1 : [25, 1]
                            W2 : [12, 25]
                            b2 : [12, 1]
                            W3 : [6, 12]
                            b3 : [6, 1]
        
        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """
        
        tf.set_random_seed(1)                   # so that your "random" numbers match ours
            
        ### START CODE HERE ### (approx. 6 lines of code)
        W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
        W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
        W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())
        ### END CODE HERE ###
    
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        
        return parameters
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        parameters = initialize_parameters()
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))
    
    # GRADED FUNCTION: forward_propagation
    
    def forward_propagation(X, parameters):
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
        
        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                      the shapes are given in initialize_parameters
    
        Returns:
        Z3 -- the output of the last LINEAR unit
        """
        
        # Retrieve the parameters from the dictionary "parameters" 
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']
        
        ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
        Z1 = tf.add(tf.matmul(W1,X),b1)                                               # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
        Z2 = tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, A1) + b2
        A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
        Z3 = tf.add(tf.matmul(W3,A2),b3)                                              # Z3 = np.dot(W3, A2) + b3
        ### END CODE HERE ###
        
        return Z3
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        X, Y = create_placeholders(12288, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        print("Z3 = " + str(Z3))
    
    # GRADED FUNCTION: compute_cost 
    
    def compute_cost(Z3, Y):
        """
        Computes the cost
        
        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3
        
        Returns:
        cost - Tensor of the cost function
        """
        
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        
        ### START CODE HERE ### (1 line of code)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
        ### END CODE HERE ###
        
        return cost
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        X, Y = create_placeholders(12288, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        print("cost = " + str(cost))
        
    def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
              num_epochs = 1500, minibatch_size = 32, print_cost = True):
        """
        Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
        
        Arguments:
        X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
        Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
        X_test -- training set, of shape (input size = 12288, number of training examples = 120)
        Y_test -- test set, of shape (output size = 6, number of test examples = 120)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        
        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)                             # to keep consistent results
        seed = 3                                          # to keep consistent results
        (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]                            # n_y : output size
        costs = []                                        # To keep track of the cost
        
        # Create Placeholders of shape (n_x, n_y)
        ### START CODE HERE ### (1 line)
        X, Y = create_placeholders(n_x,n_y)
        ### END CODE HERE ###
    
        # Initialize parameters
        ### START CODE HERE ### (1 line)
        parameters = initialize_parameters()
        ### END CODE HERE ###
        
        # Forward propagation: Build the forward propagation in the tensorflow graph
        ### START CODE HERE ### (1 line)
        Z3 = forward_propagation(X, parameters)
        ### END CODE HERE ###
        
        # Cost function: Add cost function to tensorflow graph
        ### START CODE HERE ### (1 line)
        cost = compute_cost(Z3, Y)
        ### END CODE HERE ###
        
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        ### START CODE HERE ### (1 line)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
        ### END CODE HERE ###
        
        # Initialize all the variables
        init = tf.global_variables_initializer()
    
        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:
            
            # Run the initialization
            sess.run(init)
            
            # Do the training loop
            for epoch in range(num_epochs):
    
                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
    
                for minibatch in minibatches:
    
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    ### START CODE HERE ### (1 line)
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    ### END CODE HERE ###
                    
                    epoch_cost += minibatch_cost / minibatch_size
    
                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)
                    
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per fives)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
    
            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print ("Parameters have been trained!")
    
            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    
            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
            print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            
            return parameters    
        
    parameters = model(X_train, Y_train, X_test, Y_test)    
        
    import scipy
    from PIL import Image
    from scipy import ndimage
    
    ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
    my_image = "thumbs_up.jpg"
    ## END CODE HERE ##
    
    # We preprocess your image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    image = image/255.
    my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
    my_image_prediction = predict(my_image, parameters)
    
    plt.imshow(image)
    print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))    
#tf1_tut_1()

### using np, uni-dir rnn for NLP, 1 word @ a time ###
def rnn_step_1():
    import numpy as np
    #from rnn_utils import *
    
    # GRADED FUNCTION: rnn_cell_forward
    
    def rnn_cell_forward(xt, a_prev, parameters):
        """
        Implements a single forward step of the RNN-cell as described in Figure (2)
    
        Arguments:
        xt -- your input data at timestep "t", numpy array of shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            ba --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        Returns:
        a_next -- next hidden state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
        """
        
        # Retrieve parameters from "parameters"
        Wax = parameters["Wax"]
        Waa = parameters["Waa"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]
        
        ### START CODE HERE ### (≈2 lines)
        # compute next activation state using the formula given above
        a_next = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,xt)+ba)
        # compute output of the current cell using the formula given above
        yt_pred = softmax(np.dot(Wya,a_next)+by)
        ### END CODE HERE ###
        
        # store values you need for backward propagation in cache
        cache = (a_next, a_prev, xt, parameters)
        
        return a_next, yt_pred, cache
    
    np.random.seed(1)
    xt_tmp = np.random.randn(3,10)
    a_prev_tmp = np.random.randn(5,10)
    parameters_tmp = {}
    parameters_tmp['Waa'] = np.random.randn(5,5)
    parameters_tmp['Wax'] = np.random.randn(5,3)
    parameters_tmp['Wya'] = np.random.randn(2,5)
    parameters_tmp['ba'] = np.random.randn(5,1)
    parameters_tmp['by'] = np.random.randn(2,1)
    
    a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
    print("a_next[4] = \n", a_next_tmp[4])
    print("a_next.shape = \n", a_next_tmp.shape)
    print("yt_pred[1] =\n", yt_pred_tmp[1])
    print("yt_pred.shape = \n", yt_pred_tmp.shape)
    
    # GRADED FUNCTION: rnn_forward
    
    def rnn_forward(x, a0, parameters):
        """
        Implement the forward propagation of the recurrent neural network described in Figure (3).
    
        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, T_x).
        a0 -- Initial hidden state, of shape (n_a, m)
        parameters -- python dictionary containing:
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            ba --  Bias numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    
        Returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of caches, x)
        """
        
        # Initialize "caches" which will contain the list of all caches
        caches = []
        
        # Retrieve dimensions from shapes of x and parameters["Wya"]
        n_x, m, T_x = x.shape
        n_y, n_a = parameters["Wya"].shape
        
        ### START CODE HERE ###
        
        # initialize "a" and "y_pred" with zeros (≈2 lines)
        a = np.zeros([n_a,m,T_x])
        y_pred = np.zeros([n_y,m,T_x])
        
        # Initialize a_next (≈1 line)
        a_next = a0
        
        # loop over all time-steps of the input 'x' (1 line)
        for t in range(T_x):
            # Update next hidden state, compute the prediction, get the cache (≈2 lines)
            xt = x[:,:,t]
            a_next, yt_pred, cache = rnn_cell_forward(xt, a_next, parameters)
            # Save the value of the new "next" hidden state in a (≈1 line)
            a[:,:,t] = a_next
            # Save the value of the prediction in y (≈1 line)
            y_pred[:,:,t] = yt_pred
            # Append "cache" to "caches" (≈1 line)
            caches.append(cache)
            
        ### END CODE HERE ###
        
        # store values needed for backward propagation in cache
        caches = (caches, x)
        
        return a, y_pred, caches
    
    np.random.seed(1)
    x_tmp = np.random.randn(3,10,4)
    a0_tmp = np.random.randn(5,10)
    parameters_tmp = {}
    parameters_tmp['Waa'] = np.random.randn(5,5)
    parameters_tmp['Wax'] = np.random.randn(5,3)
    parameters_tmp['Wya'] = np.random.randn(2,5)
    parameters_tmp['ba'] = np.random.randn(5,1)
    parameters_tmp['by'] = np.random.randn(2,1)
    
    a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
    print("a[4][1] = \n", a_tmp[4][1])
    print("a.shape = \n", a_tmp.shape)
    print("y_pred[1][3] =\n", y_pred_tmp[1][3])
    print("y_pred.shape = \n", y_pred_tmp.shape)
    print("caches[1][1][3] =\n", caches_tmp[1][1][3])
    print("len(caches) = \n", len(caches_tmp))
    
    # GRADED FUNCTION: lstm_cell_forward
    
    def lstm_cell_forward(xt, a_prev, c_prev, parameters):
        """
        Implement a single forward step of the LSTM-cell as described in Figure (4)
    
        Arguments:
        xt -- your input data at timestep "t", numpy array of shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
        parameters -- python dictionary containing:
                            Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                            Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                            Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                            Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                            
        Returns:
        a_next -- next hidden state, of shape (n_a, m)
        c_next -- next memory state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
        
        Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
              c stands for the cell state (memory)
        """
    
        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"] # forget gate weight
        bf = parameters["bf"]
        Wi = parameters["Wi"] # update gate weight (notice the variable name)
        bi = parameters["bi"] # (notice the variable name)
        Wc = parameters["Wc"] # candidate value weight
        bc = parameters["bc"]
        Wo = parameters["Wo"] # output gate weight
        bo = parameters["bo"]
        Wy = parameters["Wy"] # prediction weight
        by = parameters["by"]
        
        # Retrieve dimensions from shapes of xt and Wy
        n_x, m = xt.shape
        n_y, n_a = Wy.shape
    
        ### START CODE HERE ###
        # Concatenate a_prev and xt (≈1 line)
        concat = np.concatenate((a_prev,xt),axis=0)
    
        # Compute values for ft (forget gate), it (update gate),
        # cct (candidate value), c_next (cell state), 
        # ot (output gate), a_next (hidden state) (≈6 lines)
        ft = sigmoid(np.dot(Wf, concat) + bf)        # forget gate
        it = sigmoid(np.dot(Wi, concat) + bi)        # update gate
        cct = np.tanh(np.dot(Wc, concat) + bc)       # candidate value
        c_next = ft * c_prev + it * cct     # cell state
        ot = sigmoid(np.dot(Wo, concat) + bo)        # output gate
        a_next = ot * np.tanh(c_next)    # hidden state
        
        # Compute prediction of the LSTM cell (≈1 line)
        yt_pred = softmax(np.dot(Wy, a_next) + by)
        ### END CODE HERE ###
    
        # store values needed for backward propagation in cache
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    
        return a_next, c_next, yt_pred, cache
    
    np.random.seed(1)
    xt_tmp = np.random.randn(3,10)
    a_prev_tmp = np.random.randn(5,10)
    c_prev_tmp = np.random.randn(5,10)
    parameters_tmp = {}
    parameters_tmp['Wf'] = np.random.randn(5, 5+3)
    parameters_tmp['bf'] = np.random.randn(5,1)
    parameters_tmp['Wi'] = np.random.randn(5, 5+3)
    parameters_tmp['bi'] = np.random.randn(5,1)
    parameters_tmp['Wo'] = np.random.randn(5, 5+3)
    parameters_tmp['bo'] = np.random.randn(5,1)
    parameters_tmp['Wc'] = np.random.randn(5, 5+3)
    parameters_tmp['bc'] = np.random.randn(5,1)
    parameters_tmp['Wy'] = np.random.randn(2,5)
    parameters_tmp['by'] = np.random.randn(2,1)
    
    a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)
    print("a_next[4] = \n", a_next_tmp[4])
    print("a_next.shape = ", a_next_tmp.shape)
    print("c_next[2] = \n", c_next_tmp[2])
    print("c_next.shape = ", c_next_tmp.shape)
    print("yt[1] =", yt_tmp[1])
    print("yt.shape = ", yt_tmp.shape)
    print("cache[1][3] =\n", cache_tmp[1][3])
    print("len(cache) = ", len(cache_tmp))
    
    # GRADED FUNCTION: lstm_forward
    
    def lstm_forward(x, a0, parameters):
        """
        Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).
    
        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, T_x).
        a0 -- Initial hidden state, of shape (n_a, m)
        parameters -- python dictionary containing:
                            Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                            Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                            Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                            Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                            
        Returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
        """
    
        # Initialize "caches", which will track the list of all the caches
        caches = []
        
        ### START CODE HERE ###
        Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
        # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
        n_x, m, T_x = x.shape
        n_y, n_a = parameters["Wy"].shape
        
        # initialize "a", "c" and "y" with zeros (≈3 lines)
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_y, m, T_x))
        
        # Initialize a_next and c_next (≈2 lines)
        a_next = a0
        c_next = np.zeros(a_next.shape)
        
        # loop over all time-steps
        for t in range(T_x):
            # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
            xt = x[:,:,t]
            # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
            a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
            # Save the value of the new "next" hidden state in a (≈1 line)
            a[:,:,t] = a_next
            # Save the value of the next cell state (≈1 line)
            c[:,:,t]  = c_next
            # Save the value of the prediction in y (≈1 line)
            y[:,:,t] = yt
            # Append the cache into caches (≈1 line)
            caches.append(cache)
            
        ### END CODE HERE ###
        
        # store values needed for backward propagation in cache
        caches = (caches, x)
    
        return a, y, c, caches
    
    np.random.seed(1)
    x_tmp = np.random.randn(3,10,7)
    a0_tmp = np.random.randn(5,10)
    parameters_tmp = {}
    parameters_tmp['Wf'] = np.random.randn(5, 5+3)
    parameters_tmp['bf'] = np.random.randn(5,1)
    parameters_tmp['Wi'] = np.random.randn(5, 5+3)
    parameters_tmp['bi']= np.random.randn(5,1)
    parameters_tmp['Wo'] = np.random.randn(5, 5+3)
    parameters_tmp['bo'] = np.random.randn(5,1)
    parameters_tmp['Wc'] = np.random.randn(5, 5+3)
    parameters_tmp['bc'] = np.random.randn(5,1)
    parameters_tmp['Wy'] = np.random.randn(2,5)
    parameters_tmp['by'] = np.random.randn(2,1)
    
    a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
    print("a[4][3][6] = ", a_tmp[4][3][6])
    print("a.shape = ", a_tmp.shape)
    print("y[1][4][3] =", y_tmp[1][4][3])
    print("y.shape = ", y_tmp.shape)
    print("caches[1][1][1] =\n", caches_tmp[1][1][1])
    print("c[1][2][1]", c_tmp[1][2][1])
    print("len(caches) = ", len(caches_tmp))
    
    def rnn_cell_backward(da_next, cache):
        """
        Implements the backward pass for the RNN-cell (single time-step).
    
        Arguments:
        da_next -- Gradient of loss with respect to next hidden state
        cache -- python dictionary containing useful values (output of rnn_cell_forward())
    
        Returns:
        gradients -- python dictionary containing:
                            dx -- Gradients of input data, of shape (n_x, m)
                            da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                            dba -- Gradients of bias vector, of shape (n_a, 1)
        """
        
        # Retrieve values from cache
        (a_next, a_prev, xt, parameters) = cache
        
        # Retrieve values from parameters
        Wax = parameters["Wax"]
        Waa = parameters["Waa"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]
    
        ### START CODE HERE ###
        # compute the gradient of the loss with respect to z (optional) (≈1 line)
        dz = (1- a_next**2) * da_next
    
        # compute the gradient of the loss with respect to Wax (≈2 lines)
        dxt = np.dot(Wax.T, dz)
        dWax = np.dot(dz, xt.T)
    
        # compute the gradient with respect to Waa (≈2 lines)
        da_prev = np.dot(Waa.T, dz)
        dWaa = np.dot(dz, a_prev.T)
    
        # compute the gradient with respect to b (≈1 line)
        dba = np.sum(dz, 1, keepdims=True)
    
        ### END CODE HERE ###
        
        # Store the gradients in a python dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
        
        return gradients
    
    np.random.seed(1)
    xt_tmp = np.random.randn(3,10)
    a_prev_tmp = np.random.randn(5,10)
    parameters_tmp = {}
    parameters_tmp['Wax'] = np.random.randn(5,3)
    parameters_tmp['Waa'] = np.random.randn(5,5)
    parameters_tmp['Wya'] = np.random.randn(2,5)
    parameters_tmp['ba'] = np.random.randn(5,1)
    parameters_tmp['by'] = np.random.randn(2,1)
    
    a_next_tmp, yt_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
    
    da_next_tmp = np.random.randn(5,10)
    gradients_tmp = rnn_cell_backward(da_next_tmp, cache_tmp)
    print("gradients[\"dxt\"][1][2] =", gradients_tmp["dxt"][1][2])
    print("gradients[\"dxt\"].shape =", gradients_tmp["dxt"].shape)
    print("gradients[\"da_prev\"][2][3] =", gradients_tmp["da_prev"][2][3])
    print("gradients[\"da_prev\"].shape =", gradients_tmp["da_prev"].shape)
    print("gradients[\"dWax\"][3][1] =", gradients_tmp["dWax"][3][1])
    print("gradients[\"dWax\"].shape =", gradients_tmp["dWax"].shape)
    print("gradients[\"dWaa\"][1][2] =", gradients_tmp["dWaa"][1][2])
    print("gradients[\"dWaa\"].shape =", gradients_tmp["dWaa"].shape)
    print("gradients[\"dba\"][4] =", gradients_tmp["dba"][4])
    print("gradients[\"dba\"].shape =", gradients_tmp["dba"].shape)
    
    def rnn_backward(da, caches):
        """
        Implement the backward pass for a RNN over an entire sequence of input data.
    
        Arguments:
        da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
        caches -- tuple containing information from the forward pass (rnn_forward)
        
        Returns:
        gradients -- python dictionary containing:
                            dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                            da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                            dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                            dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                            dba -- Gradient w.r.t the bias, of shape (n_a, 1)
        """
            
        ### START CODE HERE ###
        
        # Retrieve values from the first cache (t=1) of caches (≈2 lines)
        (caches, x) = caches
        (a1, a0, x1, parameters) = caches[0]
        
        # Retrieve dimensions from da's and x1's shapes (≈2 lines)
        n_a, m, T_x = da.shape
        n_x, m = x1.shape
        
        # initialize the gradients with the right sizes (≈6 lines)
        dx = np.zeros((n_x, m, T_x))
        dWax = np.zeros((n_a, n_x))
        dWaa = np.zeros((n_a, n_a))
        dba = np.zeros((n_a, 1))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros((n_a, m))
        
        # Loop through all the time steps
        for t in reversed(range(T_x)):
            # Compute gradients at time step t. 
            # Remember to sum gradients from the output path (da) and the previous timesteps (da_prevt) (≈1 line)
            gradients = rnn_cell_backward(da[:,:, t] + da_prevt, caches[t])
            # Retrieve derivatives from gradients (≈ 1 line)
            dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
            # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
            dx[:, :, t] = dxt
            dWax += dWaxt
            dWaa += dWaat
            dba += dbat
            
        # Set da0 to the gradient of a which has been backpropagated through all time-steps (≈1 line) 
        da0 = da_prevt
        ### END CODE HERE ###
    
        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
        
        return gradients
    
    np.random.seed(1)
    x_tmp = np.random.randn(3,10,4)
    a0_tmp = np.random.randn(5,10)
    parameters_tmp = {}
    parameters_tmp['Wax'] = np.random.randn(5,3)
    parameters_tmp['Waa'] = np.random.randn(5,5)
    parameters_tmp['Wya'] = np.random.randn(2,5)
    parameters_tmp['ba'] = np.random.randn(5,1)
    parameters_tmp['by'] = np.random.randn(2,1)
    
    a_tmp, y_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
    da_tmp = np.random.randn(5, 10, 4)
    gradients_tmp = rnn_backward(da_tmp, caches_tmp)
    
    print("gradients[\"dx\"][1][2] =", gradients_tmp["dx"][1][2])
    print("gradients[\"dx\"].shape =", gradients_tmp["dx"].shape)
    print("gradients[\"da0\"][2][3] =", gradients_tmp["da0"][2][3])
    print("gradients[\"da0\"].shape =", gradients_tmp["da0"].shape)
    print("gradients[\"dWax\"][3][1] =", gradients_tmp["dWax"][3][1])
    print("gradients[\"dWax\"].shape =", gradients_tmp["dWax"].shape)
    print("gradients[\"dWaa\"][1][2] =", gradients_tmp["dWaa"][1][2])
    print("gradients[\"dWaa\"].shape =", gradients_tmp["dWaa"].shape)
    print("gradients[\"dba\"][4] =", gradients_tmp["dba"][4])
    print("gradients[\"dba\"].shape =", gradients_tmp["dba"].shape)
    
    def lstm_cell_backward(da_next, dc_next, cache):
        """
        Implement the backward pass for the LSTM-cell (single time-step).
    
        Arguments:
        da_next -- Gradients of next hidden state, of shape (n_a, m)
        dc_next -- Gradients of next cell state, of shape (n_a, m)
        cache -- cache storing information from the forward pass
    
        Returns:
        gradients -- python dictionary containing:
                            dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                            da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                            dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                            dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                            dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                            dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                            dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                            dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
        """
    
        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
        
        ### START CODE HERE ###
        # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
        n_x, m =xt.shape
        n_a, m = a_next.shape
        
        # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
        dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
        dft = (dc_next * c_prev + ot *(1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)
        concat = np.concatenate((a_prev, xt), axis=0)
        # Compute parameters related derivatives. Use equations (11)-(18) (≈8 lines)
        dWf = np.dot(dft, concat.T)
        dWi = np.dot(dit, concat.T)
        dWc = np.dot(dcct, concat.T)
        dWo = np.dot(dot, concat.T)
        dbf = np.sum(dft, axis=1 ,keepdims = True)
        dbi = np.sum(dit, axis=1, keepdims = True)
        dbc = np.sum(dcct, axis=1,  keepdims = True)
        dbo = np.sum(dot, axis=1, keepdims = True)
        
        # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (19)-(21). (≈3 lines)
        da_prev =  np.dot(parameters['Wf'][:, :n_a].T, dft) + np.dot(parameters['Wi'][:, :n_a].T, dit) + np.dot(parameters['Wc'][:, :n_a].T, dcct) + np.dot(parameters['Wo'][:, :n_a].T, dot)
        dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
        dxt = np.dot(parameters['Wf'][:, n_a:].T, dft) + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(parameters['Wc'][:, n_a:].T, dcct) + np.dot(parameters['Wo'][:, n_a:].T, dot)
        ### END CODE HERE ###
        
        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
        return gradients
    
    np.random.seed(1)
    xt_tmp = np.random.randn(3,10)
    a_prev_tmp = np.random.randn(5,10)
    c_prev_tmp = np.random.randn(5,10)
    parameters_tmp = {}
    parameters_tmp['Wf'] = np.random.randn(5, 5+3)
    parameters_tmp['bf'] = np.random.randn(5,1)
    parameters_tmp['Wi'] = np.random.randn(5, 5+3)
    parameters_tmp['bi'] = np.random.randn(5,1)
    parameters_tmp['Wo'] = np.random.randn(5, 5+3)
    parameters_tmp['bo'] = np.random.randn(5,1)
    parameters_tmp['Wc'] = np.random.randn(5, 5+3)
    parameters_tmp['bc'] = np.random.randn(5,1)
    parameters_tmp['Wy'] = np.random.randn(2,5)
    parameters_tmp['by'] = np.random.randn(2,1)
    
    a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)
    
    da_next_tmp = np.random.randn(5,10)
    dc_next_tmp = np.random.randn(5,10)
    gradients_tmp = lstm_cell_backward(da_next_tmp, dc_next_tmp, cache_tmp)
    print("gradients[\"dxt\"][1][2] =", gradients_tmp["dxt"][1][2])
    print("gradients[\"dxt\"].shape =", gradients_tmp["dxt"].shape)
    print("gradients[\"da_prev\"][2][3] =", gradients_tmp["da_prev"][2][3])
    print("gradients[\"da_prev\"].shape =", gradients_tmp["da_prev"].shape)
    print("gradients[\"dc_prev\"][2][3] =", gradients_tmp["dc_prev"][2][3])
    print("gradients[\"dc_prev\"].shape =", gradients_tmp["dc_prev"].shape)
    print("gradients[\"dWf\"][3][1] =", gradients_tmp["dWf"][3][1])
    print("gradients[\"dWf\"].shape =", gradients_tmp["dWf"].shape)
    print("gradients[\"dWi\"][1][2] =", gradients_tmp["dWi"][1][2])
    print("gradients[\"dWi\"].shape =", gradients_tmp["dWi"].shape)
    print("gradients[\"dWc\"][3][1] =", gradients_tmp["dWc"][3][1])
    print("gradients[\"dWc\"].shape =", gradients_tmp["dWc"].shape)
    print("gradients[\"dWo\"][1][2] =", gradients_tmp["dWo"][1][2])
    print("gradients[\"dWo\"].shape =", gradients_tmp["dWo"].shape)
    print("gradients[\"dbf\"][4] =", gradients_tmp["dbf"][4])
    print("gradients[\"dbf\"].shape =", gradients_tmp["dbf"].shape)
    print("gradients[\"dbi\"][4] =", gradients_tmp["dbi"][4])
    print("gradients[\"dbi\"].shape =", gradients_tmp["dbi"].shape)
    print("gradients[\"dbc\"][4] =", gradients_tmp["dbc"][4])
    print("gradients[\"dbc\"].shape =", gradients_tmp["dbc"].shape)
    print("gradients[\"dbo\"][4] =", gradients_tmp["dbo"][4])
    print("gradients[\"dbo\"].shape =", gradients_tmp["dbo"].shape)
    
    def lstm_backward(da, caches):
        
        """
        Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).
    
        Arguments:
        da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
        caches -- cache storing information from the forward pass (lstm_forward)
    
        Returns:
        gradients -- python dictionary containing:
                            dx -- Gradient of inputs, of shape (n_x, m, T_x)
                            da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                            dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                            dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                            dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                            dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                            dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                            dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
        """
    
        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches
        (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
        
        ### START CODE HERE ###
        # Retrieve dimensions from da's and x1's shapes (≈2 lines)
        n_a, m, T_x = da.shape
        n_x, m = x1.shape
        
        # initialize the gradients with the right sizes (≈12 lines)
        dx = np.zeros((n_x, m, T_x))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros(da0.shape)
        dc_prevt = np.zeros(da0.shape)
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros(dWf.shape)
        dWc = np.zeros(dWf.shape)
        dWo = np.zeros(dWf.shape)
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros(dbf.shape)
        dbc = np.zeros(dbf.shape)
        dbo = np.zeros(dbf.shape)
        
        # loop back over the whole sequence
        for t in reversed(range(T_x)):
            # Compute all gradients using lstm_cell_backward
            gradients = lstm_cell_backward(da[:, :, t], dc_prevt, caches[t])
            # Store or add the gradient to the parameters' previous step's gradient
            da_prevt = da[:, :, t]
            dc_prevt = dc_prevt
            dx[:,:,t] = gradients["dxt"]
            dWf += gradients["dWf"]
            dWi += gradients["dWi"]
            dWc += gradients["dWc"]
            dWo += gradients["dWo"]
            dbf += gradients["dbf"]
            dbi += gradients["dbi"]
            dbc += gradients["dbc"]
            dbo += gradients["dbo"]
        # Set the first activation's gradient to the backpropagated gradient da_prev.
        da0 = gradients["da_prev"]
        
        ### END CODE HERE ###
    
        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
        
        return gradients
    
    np.random.seed(1)
    x_tmp = np.random.randn(3,10,7)
    a0_tmp = np.random.randn(5,10)
    
    parameters_tmp = {}
    parameters_tmp['Wf'] = np.random.randn(5, 5+3)
    parameters_tmp['bf'] = np.random.randn(5,1)
    parameters_tmp['Wi'] = np.random.randn(5, 5+3)
    parameters_tmp['bi'] = np.random.randn(5,1)
    parameters_tmp['Wo'] = np.random.randn(5, 5+3)
    parameters_tmp['bo'] = np.random.randn(5,1)
    parameters_tmp['Wc'] = np.random.randn(5, 5+3)
    parameters_tmp['bc'] = np.random.randn(5,1)
    parameters_tmp['Wy'] = np.zeros((2,5))       # unused, but needed for lstm_forward
    parameters_tmp['by'] = np.zeros((2,1))       # unused, but needed for lstm_forward
    
    a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
    
    da_tmp = np.random.randn(5, 10, 4)
    gradients_tmp = lstm_backward(da_tmp, caches_tmp)
    
    print("gradients[\"dx\"][1][2] =", gradients_tmp["dx"][1][2])
    print("gradients[\"dx\"].shape =", gradients_tmp["dx"].shape)
    print("gradients[\"da0\"][2][3] =", gradients_tmp["da0"][2][3])
    print("gradients[\"da0\"].shape =", gradients_tmp["da0"].shape)
    print("gradients[\"dWf\"][3][1] =", gradients_tmp["dWf"][3][1])
    print("gradients[\"dWf\"].shape =", gradients_tmp["dWf"].shape)
    print("gradients[\"dWi\"][1][2] =", gradients_tmp["dWi"][1][2])
    print("gradients[\"dWi\"].shape =", gradients_tmp["dWi"].shape)
    print("gradients[\"dWc\"][3][1] =", gradients_tmp["dWc"][3][1])
    print("gradients[\"dWc\"].shape =", gradients_tmp["dWc"].shape)
    print("gradients[\"dWo\"][1][2] =", gradients_tmp["dWo"][1][2])
    print("gradients[\"dWo\"].shape =", gradients_tmp["dWo"].shape)
    print("gradients[\"dbf\"][4] =", gradients_tmp["dbf"][4])
    print("gradients[\"dbf\"].shape =", gradients_tmp["dbf"].shape)
    print("gradients[\"dbi\"][4] =", gradients_tmp["dbi"][4])
    print("gradients[\"dbi\"].shape =", gradients_tmp["dbi"].shape)
    print("gradients[\"dbc\"][4] =", gradients_tmp["dbc"][4])
    print("gradients[\"dbc\"].shape =", gradients_tmp["dbc"].shape)
    print("gradients[\"dbo\"][4] =", gradients_tmp["dbo"][4])
    print("gradients[\"dbo\"].shape =", gradients_tmp["dbo"].shape)
#rnn_step_1()

""" create dino names 
How to store text data for processing using an RNN
How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
How to build a character-level text generation recurrent neural network
Why clipping the gradients is important """
def dino_isl_1():
    import numpy as np
    #from utils import *
    import random
    import pprint
    data = open('dinos.txt', 'r').read()
    data= data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
    
    chars = sorted(chars)
    print(chars)
    
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(ix_to_char)
    
    ### GRADED FUNCTION: clip
    
    def clip(gradients, maxValue):
        '''
        Clips the gradients' values between minimum and maximum.
        
        Arguments:
        gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
        
        Returns: 
        gradients -- a dictionary with the clipped gradients.
        '''
        
        dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
       
        ### START CODE HERE ###
        # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
        for gradient in [dWax, dWaa, dWya, db, dby]:
            np.clip(gradient, -maxValue, maxValue, out=gradient)
        ### END CODE HERE ###
        
        gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
        
        return gradients
    
    # Test with a maxvalue of 10
    mValue = 10
    np.random.seed(3)
    dWax = np.random.randn(5,3)*10
    dWaa = np.random.randn(5,5)*10
    dWya = np.random.randn(2,5)*10
    db = np.random.randn(5,1)*10
    dby = np.random.randn(2,1)*10
    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
    gradients = clip(gradients, mValue)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"db\"][4] =", gradients["db"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])
    
    # Test with a maxValue of 5
    mValue = 5
    np.random.seed(3)
    dWax = np.random.randn(5,3)*10
    dWaa = np.random.randn(5,5)*10
    dWya = np.random.randn(2,5)*10
    db = np.random.randn(5,1)*10
    dby = np.random.randn(2,1)*10
    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
    gradients = clip(gradients, mValue)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"db\"][4] =", gradients["db"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])
    del mValue # avoid common issue
    
    matrix1 = np.array([[1,1],[2,2],[3,3]]) # (3,2)
    matrix2 = np.array([[0],[0],[0]]) # (3,1) 
    vector1D = np.array([1,1]) # (2,) 
    vector2D = np.array([[1],[1]]) # (2,1)
    print("matrix1 \n", matrix1,"\n")
    print("matrix2 \n", matrix2,"\n")
    print("vector1D \n", vector1D,"\n")
    print("vector2D \n", vector2D)
    
    print("Multiply 2D and 1D arrays: result is a 1D array\n", 
          np.dot(matrix1,vector1D))
    print("Multiply 2D and 2D arrays: result is a 2D array\n", 
          np.dot(matrix1,vector2D))
    
    print("Adding (3 x 1) vector to a (3 x 1) vector is a (3 x 1) vector\n",
          "This is what we want here!\n", 
          np.dot(matrix1,vector2D) + matrix2)
    
    print("Adding a (3,) vector to a (3 x 1) vector\n",
          "broadcasts the 1D array across the second dimension\n",
          "Not what we want here!\n",
          np.dot(matrix1,vector1D) + matrix2
         )
    
    # GRADED FUNCTION: sample
    
    def sample(parameters, char_to_ix, seed):
        """
        Sample a sequence of characters according to a sequence of probability distributions output of the RNN
    
        Arguments:
        parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
        char_to_ix -- python dictionary mapping each character to an index.
        seed -- used for grading purposes. Do not worry about it.
    
        Returns:
        indices -- a list of length n containing the indices of the sampled characters.
        """
        
        # Retrieve parameters and relevant shapes from "parameters" dictionary
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
        vocab_size = by.shape[0]
        n_a = Waa.shape[1]
        
        ### START CODE HERE ###
        # Step 1: Create the a zero vector x that can be used as the one-hot vector 
        # representing the first character (initializing the sequence generation). (≈1 line)
        x = np.zeros((vocab_size, 1))
        # Step 1': Initialize a_prev as zeros (≈1 line)
        a_prev = np.zeros((n_a, 1))
        
        # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
        indices = []
        
        # idx is the index of the one-hot vector x that is set to 1
        # All other positions in x are zero.
        # We will initialize idx to -1
        idx = -1 
        
        # Loop over time-steps t. At each time-step:
        # sample a character from a probability distribution 
        # and append its index (`idx`) to the list "indices". 
        # We'll stop if we reach 50 characters 
        # (which should be very unlikely with a well trained model).
        # Setting the maximum number of characters helps with debugging and prevents infinite loops. 
        counter = 0
        newline_character = char_to_ix['\n']
        
        while (idx != newline_character and counter != 50):
            
            # Step 2: Forward propagate x using the equations (1), (2) and (3)
            a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
            z = np.dot(Wya, a) + by
            y = softmax(z)
            
            # for grading purposes
            np.random.seed(counter+seed) 
            
            # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
            # (see additional hints above)
            idx = np.random.choice(list(range(vocab_size)), p = y.ravel())
    
            # Append the index to "indices"
            indices.append(idx)
            
            # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
            # (see additional hints above)
            x = np.zeros((vocab_size, 1))
            x[idx] = 1
            
            # Update "a_prev" to be "a"
            a_prev = a
            
            # for grading purposes
            seed += 1
            counter +=1
            
        ### END CODE HERE ###
    
        if (counter == 50):
            indices.append(char_to_ix['\n'])
        
        return indices
    
    np.random.seed(2)
    _, n_a = 20, 100
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    
    
    indices = sample(parameters, char_to_ix, 0)
    print("Sampling:")
    print("list of sampled indices:\n", indices)
    print("list of sampled characters:\n", [ix_to_char[i] for i in indices])
    
    # GRADED FUNCTION: optimize
    
    def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
        """
        Execute one step of the optimization to train the model.
        
        Arguments:
        X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
        Y -- list of integers, exactly the same as X but shifted one index to the left.
        a_prev -- previous hidden state.
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            b --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        learning_rate -- learning rate for the model.
        
        Returns:
        loss -- value of the loss function (cross-entropy)
        gradients -- python dictionary containing:
                            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                            dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                            db -- Gradients of bias vector, of shape (n_a, 1)
                            dby -- Gradients of output bias vector, of shape (n_y, 1)
        a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
        """
        
        ### START CODE HERE ###
        
        # Forward propagate through time (≈1 line)
        loss, cache = rnn_forward(X, Y, a_prev, parameters)
        
        # Backpropagate through time (≈1 line)
        gradients, a = rnn_backward(X, Y, parameters, cache)
        
        # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
        gradients = clip(gradients, maxValue=5)
        
        # Update parameters (≈1 line)
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        ### END CODE HERE ###
        
        return loss, gradients, a[len(X)-1]
    
    np.random.seed(1)
    vocab_size, n_a = 27, 100
    a_prev = np.random.randn(n_a, 1)
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    X = [12,3,5,11,22,3]
    Y = [4,14,11,22,25, 26]
    
    loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
    print("Loss =", loss)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"db\"][4] =", gradients["db"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])
    print("a_last[4] =", a_last[4])
    
    # GRADED FUNCTION: model
    
    def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
        """
        Trains the model and generates dinosaur names. 
        
        Arguments:
        data -- text corpus
        ix_to_char -- dictionary that maps the index to a character
        char_to_ix -- dictionary that maps a character to an index
        num_iterations -- number of iterations to train the model for
        n_a -- number of units of the RNN cell
        dino_names -- number of dinosaur names you want to sample at each iteration. 
        vocab_size -- number of unique characters found in the text (size of the vocabulary)
        
        Returns:
        parameters -- learned parameters
        """
        
        # Retrieve n_x and n_y from vocab_size
        n_x, n_y = vocab_size, vocab_size
        
        # Initialize parameters
        parameters = initialize_parameters(n_a, n_x, n_y)
        
        # Initialize loss (this is required because we want to smooth our loss)
        loss = get_initial_loss(vocab_size, dino_names)
        
        # Build list of all dinosaur names (training examples).
        with open("dinos.txt") as f:
            examples = f.readlines()
        examples = [x.lower().strip() for x in examples]
        
        # Shuffle list of all dinosaur names
        np.random.seed(0)
        np.random.shuffle(examples)
        
        # Initialize the hidden state of your LSTM
        a_prev = np.zeros((n_a, 1))
        
        # Optimization loop
        for j in range(num_iterations):
            
            ### START CODE HERE ###
            
            # Set the index `idx` (see instructions above)
            idx = j % len(examples)
            
            # Set the input X (see instructions above)
            single_example = examples[idx]
            single_example_chars = [c for c in single_example]
            single_example_ix = [char_to_ix[ch] for ch in examples[idx]]
            X = [None] + single_example_ix        
            #X = single_example_ix
            
            # Set the labels Y (see instructions above)
            ix_newline = [char_to_ix["\n"]]
            Y = X[1:] + ix_newline
            
            # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
            # Choose a learning rate of 0.01
            curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
            
            ### END CODE HERE ###
            
            # debug statements to aid in correctly forming X, Y
            if verbose and j in [0, len(examples) -1, len(examples)]:
                print("j = " , j, "idx = ", idx,) 
            if verbose and j in [0]:
                print("single_example =", single_example)
                print("single_example_chars", single_example_chars)
                print("single_example_ix", single_example_ix)
                print(" X = ", X, "\n", "Y =       ", Y, "\n")
            
            # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
            loss = smooth(loss, curr_loss)
    
            # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
            if j % 2000 == 0:
                
                print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
                
                # The number of dinosaur names to print
                seed = 0
                for name in range(dino_names):
                    
                    # Sample indices and print them
                    sampled_indices = sample(parameters, char_to_ix, seed)
                    print_sample(sampled_indices, ix_to_char)
                    
                    seed += 1  # To get the same result (for grading purposes), increment the seed by one. 
          
                print('\n')
            
        return parameters
    
    parameters = model(data, ix_to_char, char_to_ix, verbose = True)
    
    ### Writing Like Shakespeare ###
    def shakespeare():
        pass
        # from __future__ import print_function
        # from keras.callbacks import LambdaCallback
        # from keras.models import Model, load_model, Sequential
        # from keras.layers import Dense, Activation, Dropout, Input, Masking
        # from keras.layers import LSTM
        # from keras.utils.data_utils import get_file
        # from keras.preprocessing.sequence import pad_sequences
        # #from shakespeare_utils import *
        # import sys
        # import io
        
        # print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        
        # model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
        
        # # Run this cell to try with different inputs without having to re-train the model 
        # generate_output()
    shakespeare()   
#dino_isl_1()

### Apply an LSTM to music generation. Generate your own jazz music with deep learning. ###
def imp_jazz_1():
    # from __future__ import print_function
    # import IPython
    # import sys
    # from music21 import *
    # import numpy as np
    # from grammar import *
    # from qa import *
    # from preprocess import * 
    # from music_utils import *
    # from data_utils import *
    # from keras.models import load_model, Model
    # from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
    # from keras.initializers import glorot_uniform
    # from keras.utils import to_categorical
    # from keras.optimizers import Adam
    # from keras import backend as K
    
    IPython.display.Audio('./data/30s_seq.mp3')
    
    X, Y, n_values, indices_values = load_music_utils()
    print('number of training examples:', X.shape[0])
    print('Tx (length of sequence):', X.shape[1])
    print('total # of unique values:', n_values)
    print('shape of X:', X.shape)
    print('Shape of Y:', Y.shape)
    
    # number of dimensions for the hidden state of each LSTM cell.
    n_a = 64 
    
    n_values = 78 # number of music values
    reshapor = Reshape((1, n_values))                        # Used in Step 2.B of djmodel(), below
    LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
    densor = Dense(n_values, activation='softmax')     # Used in Step 2.D
    
    # GRADED FUNCTION: djmodel
    
    def djmodel(Tx, n_a, n_values):
        """
        Implement the model
        
        Arguments:
        Tx -- length of the sequence in a corpus
        n_a -- the number of activations used in our model
        n_values -- number of unique values in the music data 
        
        Returns:
        model -- a keras instance model with n_a activations
        """
        
        # Define the input layer and specify the shape
        X = Input(shape=(Tx, n_values))
        
        # Define the initial hidden state a0 and initial cell state c0
        # using `Input`
        a0 = Input(shape=(n_a,), name='a0')
        c0 = Input(shape=(n_a,), name='c0')
        a = a0
        c = c0
        
        ### START CODE HERE ### 
        # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
        outputs = []
        
        # Step 2: Loop
        for t in range(Tx):
            
            # Step 2.A: select the "t"th time step vector from X. 
            x = Lambda(lambda x: X[:, t, :])(X)
            # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
            x = reshapor(x)
            # Step 2.C: Perform one step of the LSTM_cell
            a, _, c =  LSTM_cell(x, initial_state=[a, c])
            # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
            out = densor(a)
            # Step 2.E: add the output to "outputs"
            outputs.append(out)
            
        # Step 3: Create model instance
        model = Model(inputs=[X, a0, c0], outputs=outputs)
        
        ### END CODE HERE ###
        
        return model
    
    model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
    
    # Check your model
    model.summary()
    
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    
    model.fit([X, a0, c0], list(Y), epochs=100)
    
    # GRADED FUNCTION: music_inference_model
    
    def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
        """
        Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
        
        Arguments:
        LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
        densor -- the trained "densor" from model(), Keras layer object
        n_values -- integer, number of unique values
        n_a -- number of units in the LSTM_cell
        Ty -- integer, number of time steps to generate
        
        Returns:
        inference_model -- Keras model instance
        """
        
        # Define the input of your model with a shape 
        x0 = Input(shape=(1, n_values))
        
        # Define s0, initial hidden state for the decoder LSTM
        a0 = Input(shape=(n_a,), name='a0')
        c0 = Input(shape=(n_a,), name='c0')
        a = a0
        c = c0
        x = x0
    
        ### START CODE HERE ###
        # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
        outputs = []
        
        # Step 2: Loop over Ty and generate a value at every time step
        for t in range(Ty):
            
            # Step 2.A: Perform one step of LSTM_cell (≈1 line)
            a, _, c = LSTM_cell(x, initial_state=[a, c])
            
            # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
            out = densor(a)
    
            # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
            outputs.append(out)
            
            # Step 2.D: 
            # Select the next value according to "out",
            # Set "x" to be the one-hot representation of the selected value
            # See instructions above.
            x = Lambda(one_hot)(out)
            
        # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
        inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
        
        ### END CODE HERE ###
        
        return inference_model
    
    inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)
    
    # Check the inference model
    inference_model.summary()
    
    x_initializer = np.zeros((1, 1, 78))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))
    
    # GRADED FUNCTION: predict_and_sample
    
    def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                           c_initializer = c_initializer):
        """
        Predicts the next value of values using the inference model.
        
        Arguments:
        inference_model -- Keras model instance for inference time
        x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
        a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
        c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
        
        Returns:
        results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
        indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
        """
        
        ### START CODE HERE ###
        # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
        pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
        # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
        indices = np.argmax(pred, axis=2)
        # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
        results = to_categorical(indices)
        ### END CODE HERE ###
        
        return results, indices
    
    results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
    print("np.argmax(results[12]) =", np.argmax(results[12]))
    print("np.argmax(results[17]) =", np.argmax(results[17]))
    print("list(indices[12:18]) =", list(indices[12:18]))
    
    out_stream = generate_music(inference_model)
    
    IPython.display.Audio('./data/30s_trained_model.mp3')
#imp_jazz_1()

"""### Load pre-trained word vectors, and measure similarity using cosine similarity
Use word embeddings to solve word analogy problems such as Man is to Woman as King is to __.
Modify word embeddings to reduce their gender bias ###"""
def word_vec_1():
    import numpy as np
    #from w2v_utils import *
    words, word_to_vec_map = read_glove_vecs('../../readonly/glove.6B.50d.txt')
    
    # GRADED FUNCTION: cosine_similarity
    
    def cosine_similarity(u, v):
        """
        Cosine similarity reflects the degree of similarity between u and v
            
        Arguments:
            u -- a word vector of shape (n,)          
            v -- a word vector of shape (n,)
    
        Returns:
            cosine_similarity -- the cosine similarity between u and v defined by the formula above.
        """
        
        distance = 0.0
        
        ### START CODE HERE ###
        # Compute the dot product between u and v (≈1 line)
        dot =  np.dot(u.T, v)
        # Compute the L2 norm of u (≈1 line)
        norm_u = np.sqrt(np.sum(np.power(u, 2)))
        
        # Compute the L2 norm of v (≈1 line)
        norm_v =  np.sqrt(np.sum(np.power(v, 2)))
        # Compute the cosine similarity defined by formula (1) (≈1 line)
        cosine_similarity = np.divide(dot, norm_u * norm_v)
        ### END CODE HERE ###
        
        return cosine_similarity
    
    father = word_to_vec_map["father"]
    mother = word_to_vec_map["mother"]
    ball = word_to_vec_map["ball"]
    crocodile = word_to_vec_map["crocodile"]
    france = word_to_vec_map["france"]
    italy = word_to_vec_map["italy"]
    paris = word_to_vec_map["paris"]
    rome = word_to_vec_map["rome"]
    
    print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
    print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
    print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))
    
    # GRADED FUNCTION: complete_analogy
    
    def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
        """
        Performs the word analogy task as explained above: a is to b as c is to ____. 
        
        Arguments:
        word_a -- a word, string
        word_b -- a word, string
        word_c -- a word, string
        word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
        
        Returns:
        best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
        """
        
        # convert words to lowercase
        word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
        
        ### START CODE HERE ###
        # Get the word embeddings e_a, e_b and e_c (≈1-3 lines)
        e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
        ### END CODE HERE ###
        
        words = word_to_vec_map.keys()
        max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
        best_word = None                   # Initialize best_word with None, it will help keep track of the word to output
    
        # to avoid best_word being one of the input words, skip the input words
        # place the input words in a set for faster searching than a list
        # We will re-use this set of input words inside the for-loop
        input_words_set = set([word_a, word_b, word_c])
        
        # loop over the whole word vector set
        for w in words:        
            # to avoid best_word being one of the input words, skip the input words
            if w in input_words_set:
                continue
            
            ### START CODE HERE ###
            # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  (≈1 line)
            cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)
            
            # If the cosine_sim is more than the max_cosine_sim seen so far,
                # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
            if cosine_sim > max_cosine_sim:
                max_cosine_sim = cosine_sim
                best_word = w
            ### END CODE HERE ###
            
        return best_word
    
    triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
    for triad in triads_to_try:
        print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
    
    g = word_to_vec_map['woman'] - word_to_vec_map['man']
    print(g)
    
    print ('List of names and their similarities with constructed vector:')
    
    # girls and boys name
    name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
    
    for w in name_list:
        print (w, cosine_similarity(word_to_vec_map[w], g))
    
    print('Other words and their similarities:')
    word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
                 'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
    for w in word_list:
        print (w, cosine_similarity(word_to_vec_map[w], g))
    
    def neutralize(word, g, word_to_vec_map):
        """
        Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
        This function ensures that gender neutral words are zero in the gender subspace.
        
        Arguments:
            word -- string indicating the word to debias
            g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
            word_to_vec_map -- dictionary mapping words to their corresponding vectors.
        
        Returns:
            e_debiased -- neutralized word vector representation of the input "word"
        """
        
        ### START CODE HERE ###
        # Select word vector representation of "word". Use word_to_vec_map. (≈ 1 line)
        e = word_to_vec_map[word]
        
        # Compute e_biascomponent using the formula given above. (≈ 1 line)
        e_biascomponent = np.divide(np.dot(e, g), np.linalg.norm(g) ** 2) * g
     
        # Neutralize e by subtracting e_biascomponent from it 
        # e_debiased should be equal to its orthogonal projection. (≈ 1 line)
        e_debiased = e - e_biascomponent
        ### END CODE HERE ###
        
        return e_debiased
    
    e = "receptionist"
    print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))
    
    e_debiased = neutralize("receptionist", g, word_to_vec_map)
    print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))
    
    def equalize(pair, bias_axis, word_to_vec_map):
        """
        Debias gender specific words by following the equalize method described in the figure above.
        
        Arguments:
        pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
        bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
        word_to_vec_map -- dictionary mapping words to their corresponding vectors
        
        Returns
        e_1 -- word vector corresponding to the first word
        e_2 -- word vector corresponding to the second word
        """
        
        ### START CODE HERE ###
        # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
        w1, w2 = pair
        e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
        
        # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
        mu = (e_w1 + e_w2) / 2.0
    
        # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
        mu_B = np.divide(np.dot(mu, bias_axis), np.linalg.norm(bias_axis) ** 2) * bias_axis
        mu_orth = mu - mu_B
    
        # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
        e_w1B = np.divide(np.dot(e_w1, bias_axis), np.linalg.norm(bias_axis) ** 2) * bias_axis
        e_w2B = np.divide(np.dot(e_w2, bias_axis), np.linalg.norm(bias_axis) ** 2) * bias_axis
            
        # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
        corrected_e_w1B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * np.divide(e_w1B - mu_B, np.abs(e_w1 - mu_orth - mu_B))
        corrected_e_w2B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * np.divide(e_w2B - mu_B, np.abs(e_w2 - mu_orth - mu_B))
    
        # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
        e1 = corrected_e_w1B + mu_orth
        e2 = corrected_e_w2B + mu_orth
                                                                    
        ### END CODE HERE ###
        
        return e1, e2
    
    print("cosine similarities before equalizing:")
    print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
    print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
    print()
    e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
    print("cosine similarities after equalizing:")
    print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
    print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
#word_vec_1()

"""### baseline model (Emojifier-V1) using word embeddings. 
    Then you will build a more sophisticated model (Emojifier-V2) 
    that further incorporates an LSTM. ### """
def emojify_1():
    import numpy as np
    #from emo_utils import *
    import emoji
    import matplotlib.pyplot as plt
    
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')
    
    maxLen = len(max(X_train, key=len).split())
    
    for idx in range(10):
        print(X_train[idx], label_to_emoji(Y_train[idx]))
    
    Y_oh_train = convert_to_one_hot(Y_train, C = 5)
    Y_oh_test = convert_to_one_hot(Y_test, C = 5)
    
    idx = 50
    print(f"Sentence '{X_train[50]}' has label index {Y_train[idx]}, which is emoji {label_to_emoji(Y_train[idx])}", )
    print(f"Label index {Y_train[idx]} in one-hot encoding format is {Y_oh_train[idx]}")
    
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../../readonly/glove.6B.50d.txt')
    
    word = "cucumber"
    idx = 289846
    print("the index of", word, "in the vocabulary is", word_to_index[word])
    print("the", str(idx) + "th word in the vocabulary is", index_to_word[idx])
    
    # GRADED FUNCTION: sentence_to_avg
    
    def sentence_to_avg(sentence, word_to_vec_map):
        """
        Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
        and averages its value into a single vector encoding the meaning of the sentence.
        
        Arguments:
        sentence -- string, one training example from X
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        
        Returns:
        avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
        """
        
        ### START CODE HERE ###
        # Step 1: Split sentence into list of lower case words (≈ 1 line)
        words = sentence.lower().split()
    
        # Initialize the average word vector, should have the same shape as your word vectors.
        avg = np.zeros((50,))
        
        # Step 2: average the word vectors. You can loop over the words in the list "words".
        total = 0
        for w in words:
            avg += word_to_vec_map[w]
        avg = avg /float(len(words))
        
        ### END CODE HERE ###
        
        return avg
    
    avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
    print("avg = \n", avg)
    
    # GRADED FUNCTION: model
    
    def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
        """
        Model to train word vector representations in numpy.
        
        Arguments:
        X -- input data, numpy array of sentences as strings, of shape (m, 1)
        Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        learning_rate -- learning_rate for the stochastic gradient descent algorithm
        num_iterations -- number of iterations
        
        Returns:
        pred -- vector of predictions, numpy-array of shape (m, 1)
        W -- weight matrix of the softmax layer, of shape (n_y, n_h)
        b -- bias of the softmax layer, of shape (n_y,)
        """
        
        np.random.seed(1)
    
        # Define number of training examples
        m = Y.shape[0]                          # number of training examples
        n_y = 5                                 # number of classes  
        n_h = 50                                # dimensions of the GloVe vectors 
        
        # Initialize parameters using Xavier initialization
        W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
        b = np.zeros((n_y,))
        
        # Convert Y to Y_onehot with n_y classes
        Y_oh = convert_to_one_hot(Y, C = n_y) 
        
        # Optimization loop
        for t in range(num_iterations): # Loop over the number of iterations
            for i in range(m):          # Loop over the training examples
                
                ### START CODE HERE ### (≈ 4 lines of code)
                # Average the word vectors of the words from the i'th training example
                avg = sentence_to_avg(X[i], word_to_vec_map)
    
                # Forward propagate the avg through the softmax layer
                z = np.dot(W, avg) + b
                a = softmax(z)
    
                # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
                cost = -np.sum(Y_oh[i] * np.log(a))
                ### END CODE HERE ###
                
                # Compute gradients 
                dz = a - Y_oh[i]
                dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
                db = dz
    
                # Update parameters with Stochastic Gradient Descent
                W = W - learning_rate * dW
                b = b - learning_rate * db
            
            if t % 100 == 0:
                print("Epoch: " + str(t) + " --- cost = " + str(cost))
                pred = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py
    
        return pred, W, b
    
    print(X_train.shape)
    print(Y_train.shape)
    print(np.eye(5)[Y_train.reshape(-1)].shape)
    print(X_train[0])
    print(type(X_train))
    Y = np.asarray([5,0,0,5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
    print(Y.shape)
    
    X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
     'Lets go party and drinks','Congrats on the new job','Congratulations',
     'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
     'You totally deserve this prize', 'Let us go play football',
     'Are you down for football this afternoon', 'Work hard play harder',
     'It is suprising how people can be dumb sometimes',
     'I am very disappointed','It is the best day in my life',
     'I think I will end up alone','My life is so boring','Good job',
     'Great so awesome'])
    
    print(X.shape)
    print(np.eye(5)[Y_train.reshape(-1)].shape)
    print(type(X_train))
    
    pred, W, b = model(X_train, Y_train, word_to_vec_map)
    print(pred)
    
    print("Training set:")
    pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
    print('Test set:')
    pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)
    
    X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
    Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])
    
    pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
    print_predictions(X_my_sentences, pred)
    
    print(Y_test.shape)
    print('           '+ label_to_emoji(0)+ '    ' + label_to_emoji(1) + '    ' +  label_to_emoji(2)+ '    ' + label_to_emoji(3)+'   ' + label_to_emoji(4))
    print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
    plot_confusion_matrix(Y_test, pred_test)
    
    import numpy as np
    np.random.seed(0)
    from keras.models import Model
    from keras.layers import Dense, Input, Dropout, LSTM, Activation
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    from keras.initializers import glorot_uniform
    np.random.seed(1)
    
    for idx, val in enumerate(["I", "like", "learning"]):
        print(idx,val)
    
    # GRADED FUNCTION: sentences_to_indices
    
    def sentences_to_indices(X, word_to_index, max_len):
        """
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
        
        Arguments:
        X -- array of sentences (strings), of shape (m, 1)
        word_to_index -- a dictionary containing the each word mapped to its index
        max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
        
        Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
        """
        
        m = X.shape[0]                                   # number of training examples
        
        ### START CODE HERE ###
        # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
        X_indices = np.zeros((m,max_len))
        
        for i in range(m):                               # loop over training examples
            
            # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
            sentence_words = X[i].lower().split()
            
            # Initialize j to 0
            j = 0
            
            # Loop over the words of sentence_words
            for w in sentence_words:
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j += 1
                
        ### END CODE HERE ###
        
        return X_indices
    
    X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
    X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
    print("X1 =", X1)
    print("X1_indices =\n", X1_indices)
    
    # GRADED FUNCTION: pretrained_embedding_layer
    
    def pretrained_embedding_layer(word_to_vec_map, word_to_index):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
        
        Arguments:
        word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    
        Returns:
        embedding_layer -- pretrained layer Keras instance
        """
        
        vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
        emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
        
        ### START CODE HERE ###
        # Step 1
        # Initialize the embedding matrix as a numpy array of zeros.
        # See instructions above to choose the correct shape.
        emb_matrix = np.zeros((vocab_len,emb_dim))
        
        # Step 2
        # Set each row "idx" of the embedding matrix to be 
        # the word vector representation of the idx'th word of the vocabulary
        for word, idx in word_to_index.items():
            emb_matrix[idx, :] = word_to_vec_map[word]
    
        # Step 3
        # Define Keras embedding layer with the correct input and output sizes
        # Make it non-trainable.
        embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)
        ### END CODE HERE ###
    
        # Step 4 (already done for you; please do not modify)
        # Build the embedding layer, it is required before setting the weights of the embedding layer. 
        embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
        
        # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
        embedding_layer.set_weights([emb_matrix])
        
        return embedding_layer
    
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
    
    # GRADED FUNCTION: Emojify_V2
    
    def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
        """
        Function creating the Emojify-v2 model's graph.
        
        Arguments:
        input_shape -- shape of the input, usually (max_len,)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    
        Returns:
        model -- a model instance in Keras
        """
        
        ### START CODE HERE ###
        # Define sentence_indices as the input of the graph.
        # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
        sentence_indices = Input(input_shape, dtype='int32')
        
        # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        
        # Propagate sentence_indices through your embedding layer
        # (See additional hints in the instructions).
        embeddings =  embedding_layer(sentence_indices)       
        
        # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
        # The returned output should be a batch of sequences.
        X =  LSTM(128, return_sequences=True)(embeddings)
        # Add dropout with a probability of 0.5
        X = Dropout(0.5)(X)
        # Propagate X trough another LSTM layer with 128-dimensional hidden state
        # The returned output should be a single hidden state, not a batch of sequences.
        X = LSTM(128,return_sequences=False)(X)
        # Add dropout with a probability of 0.5
        X = Dropout(0.5)(X)
        # Propagate X through a Dense layer with 5 units
        X = Dense(5)(X)
        # Add a softmax activation
        X = Activation('softmax')(X)
        
        # Create Model instance which converts sentence_indices into X.
        model = Model(input=sentence_indices, outputs=X)
        
        ### END CODE HERE ###
        
        return model
    
    model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C = 5)
    
    model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)
    
    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C = 5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print()
    print("Test accuracy = ", acc)
    
    # This code allows you to see the mislabelled examples
    C = 5
    y_test_oh = np.eye(C)[Y_test.reshape(-1)]
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    pred = model.predict(X_test_indices)
    for i in range(len(X_test)):
        x = X_test_indices
        num = np.argmax(pred[i])
        if(num != Y_test[i]):
            print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())
    
    # Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.  
    x_test = np.array(['not feeling happy'])
    X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
    print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
#emojify_1()

"""### You will build a Neural Machine Translation (NMT) model to translate 
    human-readable dates ("25th of June, 2009") into machine-readable 
    dates ("2009-06-25"). You will do this using an attention model, one of 
    the most sophisticated sequence-to-sequence models. ### """
def date_read_1():
    from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
    from keras.layers import RepeatVector, Dense, Activation, Lambda
    from keras.optimizers import Adam
    from keras.utils import to_categorical
    from keras.models import load_model, Model
    import keras.backend as K
    import numpy as np
    
    from faker import Faker
    import random
    from tqdm import tqdm
    from babel.dates import format_date
    #from nmt_utils import *
    import matplotlib.pyplot as plt
    
    m = 10000
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
    
    dataset[:10]
    
    Tx = 30
    Ty = 10
    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
    
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    print("Xoh.shape:", Xoh.shape)
    print("Yoh.shape:", Yoh.shape)
    
    index = 0
    print("Source date:", dataset[index][0])
    print("Target date:", dataset[index][1])
    print()
    print("Source after preprocessing (indices):", X[index])
    print("Target after preprocessing (indices):", Y[index])
    print()
    print("Source after preprocessing (one-hot):", Xoh[index])
    print("Target after preprocessing (one-hot):", Yoh[index])
    
    # Defined shared layers as global variables
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes = 1)
    
    # GRADED FUNCTION: one_step_attention
    
    def one_step_attention(a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        
        Returns:
        context -- context vector, input of the next (post-attention) LSTM cell
        """
        
        ### START CODE HERE ###
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = repeator(s_prev) 
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        # For grading purposes, please list 'a' first and 's_prev' second, in this order.
        concat = concatenator([a,s_prev]) 
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e =  densor1(concat) 
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = densor2(e) 
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = activator(energies) 
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context =  dotor([alphas,a]) 
        ### END CODE HERE ###
        
        return context
    
    n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
    n_s = 64 # number of units for the post-attention LSTM's hidden state "s"
    
    # Please note, this is the post attention LSTM cell.  
    # For the purposes of passing the automatic grader
    # please do not modify this global variable.  This will be corrected once the automatic grader is also updated.
    post_activation_LSTM_cell = LSTM(n_s, return_state = True) # post-attention LSTM 
    output_layer = Dense(len(machine_vocab), activation=softmax)
    
    # GRADED FUNCTION: model
    
    def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"
    
        Returns:
        model -- Keras model instance
        """
        
        # Define the inputs of your model with a shape (Tx,)
        # Define s0 (initial hidden state) and c0 (initial cell state)
        # for the decoder LSTM with shape (n_s,)
        X = Input(shape=(Tx, human_vocab_size))
        s0 = Input(shape=(n_s,), name='s0')
        c0 = Input(shape=(n_s,), name='c0')
        s = s0
        c = c0
        
        # Initialize empty list of outputs
        outputs = []
        
        ### START CODE HERE ###
        
        # Step 1: Define your pre-attention Bi-LSTM. (≈ 1 line)
        a = Bidirectional(LSTM(n_a, return_sequences=True),input_shape=(m,Tx,n_a*2))(X)
        
        # Step 2: Iterate for Ty steps
        for t in range(Ty):
        
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = one_step_attention(a, s)
            
            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, _, c = post_activation_LSTM_cell(context,initial_state = [s, c] ) 
            
            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = output_layer(s)
            
            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)
        
        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        model = Model(inputs=[X,s0,c0],outputs=outputs)
        
        ### END CODE HERE ###
        
        return model
    
    model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    
    model.summary()
    
    ### START CODE HERE ### (≈2 lines)
    opt =  Adam(lr=0.005, beta_1=0.9, beta_2=0.999,decay=0.01) 
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    ### END CODE HERE ###
    
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0,1))
    
    model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
    
    model.load_weights('models/model.h5')
    
    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
    for example in EXAMPLES:
        
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
        prediction = model.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis = -1)
        output = [inv_machine_vocab[int(i)] for i in prediction]
        
        print("source:", example)
        print("output:", ''.join(output),"\n")
    
    model.summary()
    
    attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64);
#date_read_1()

### speech recognition, Synthesize/process audio recordings to create 
# train/dev datasets Train a trigger word detection model and make predictions ###
def trig_word_1():
    import numpy as np
    from pydub import AudioSegment
    import random
    import sys
    import io
    import os
    import glob
    import IPython
    #from td_utils import *
    
    IPython.display.Audio("./raw_data/activates/1.wav")
    IPython.display.Audio("./raw_data/negatives/4.wav")
    IPython.display.Audio("./raw_data/backgrounds/1.wav")
    IPython.display.Audio("audio_examples/example_train.wav")
    x = graph_spectrogram("audio_examples/example_train.wav")
    
    _, data = wavfile.read("audio_examples/example_train.wav")
    print("Time steps in audio recording before spectrogram", data[:,0].shape)
    print("Time steps in input after spectrogram", x.shape)
    
    Tx = 5511 # The number of time steps input to the model from the spectrogram
    n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
    
    Ty = 1375 # The number of time steps in the output of our model
    
    # Load audio segments using pydub 
    activates, negatives, backgrounds = load_raw_audio()
    
    print("background len should be 10,000, since it is a 10 sec clip\n" + str(len(backgrounds[0])),"\n")
    print("activate[0] len may be around 1000, since an `activate` audio clip is usually around 1 second (but varies a lot) \n" + str(len(activates[0])),"\n")
    print("activate[1] len: different `activate` clips can have different lengths\n" + str(len(activates[1])),"\n")
    
    def get_random_time_segment(segment_ms):
        """
        Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
        
        Arguments:
        segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
        
        Returns:
        segment_time -- a tuple of (segment_start, segment_end) in ms
        """
        
        segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
        segment_end = segment_start + segment_ms - 1
        
        return (segment_start, segment_end)
    
    # GRADED FUNCTION: is_overlapping
    
    def is_overlapping(segment_time, previous_segments):
        """
        Checks if the time of a segment overlaps with the times of existing segments.
        
        Arguments:
        segment_time -- a tuple of (segment_start, segment_end) for the new segment
        previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
        
        Returns:
        True if the time segment overlaps with any of the existing segments, False otherwise
        """
        
        segment_start, segment_end = segment_time
        
        ### START CODE HERE ### (≈ 4 lines)
        # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
        overlap = False
        
        # Step 2: loop over the previous_segments start and end times.
        # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
        for previous_start, previous_end in previous_segments:
            if segment_start <= previous_end and segment_end >= previous_start:
                overlap = True
        ### END CODE HERE ###
    
        return overlap
    
    overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
    overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
    print("Overlap 1 = ", overlap1)
    print("Overlap 2 = ", overlap2)
    
    # GRADED FUNCTION: insert_audio_clip
    
    def insert_audio_clip(background, audio_clip, previous_segments):
        """
        Insert a new audio segment over the background noise at a random time step, ensuring that the 
        audio segment does not overlap with existing segments.
        
        Arguments:
        background -- a 10 second background audio recording.  
        audio_clip -- the audio clip to be inserted/overlaid. 
        previous_segments -- times where audio segments have already been placed
        
        Returns:
        new_background -- the updated background audio
        """
        
        # Get the duration of the audio clip in ms
        segment_ms = len(audio_clip)
        
        ### START CODE HERE ### 
        # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
        # the new audio clip. (≈ 1 line)
        segment_time = get_random_time_segment(segment_ms)
        
        # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
        # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
        while  is_overlapping(segment_time, previous_segments):
            segment_time = get_random_time_segment(segment_ms)
    
        # Step 3: Append the new segment_time to the list of previous_segments (≈ 1 line)
        previous_segments.append(segment_time)
        ### END CODE HERE ###
        
        # Step 4: Superpose audio segment and background
        new_background = background.overlay(audio_clip, position = segment_time[0])
        
        return new_background, segment_time
    
    np.random.seed(5)
    audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
    audio_clip.export("insert_test.wav", format="wav")
    print("Segment Time: ", segment_time)
    IPython.display.Audio("insert_test.wav")
    
    # Expected audio
    IPython.display.Audio("audio_examples/insert_reference.wav")
    
    # GRADED FUNCTION: insert_ones
    
    def insert_ones(y, segment_end_ms):
        """
        Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
        should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
        50 following labels should be ones.
        
        
        Arguments:
        y -- numpy array of shape (1, Ty), the labels of the training example
        segment_end_ms -- the end time of the segment in ms
        
        Returns:
        y -- updated labels
        """
        
        # duration of the background (in terms of spectrogram time-steps)
        segment_end_y = int(segment_end_ms * Ty / 10000.0)
        
        # Add 1 to the correct index in the background label (y)
        ### START CODE HERE ### (≈ 3 lines)
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < Ty:
                y[0, i] = 1
        ### END CODE HERE ###
        
        return y
    
    arr1 = insert_ones(np.zeros((1, Ty)), 9700)
    plt.plot(insert_ones(arr1, 4251)[0,:])
    print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])
    
    # GRADED FUNCTION: create_training_example
    
    def create_training_example(background, activates, negatives):
        """
        Creates a training example with a given background, activates, and negatives.
        
        Arguments:
        background -- a 10 second background audio recording
        activates -- a list of audio segments of the word "activate"
        negatives -- a list of audio segments of random words that are not "activate"
        
        Returns:
        x -- the spectrogram of the training example
        y -- the label at each time step of the spectrogram
        """
        
        # Set the random seed
        np.random.seed(18)
        
        # Make background quieter
        background = background - 20
    
        ### START CODE HERE ###
        # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
        y = np.zeros((1,Ty))
    
        # Step 2: Initialize segment times as an empty list (≈ 1 line)
        previous_segments = []
        ### END CODE HERE ###
        
        # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
        number_of_activates = np.random.randint(0, 5)
        random_indices = np.random.randint(len(activates), size=number_of_activates)
        random_activates = [activates[i] for i in random_indices]
        
        ### START CODE HERE ### (≈ 3 lines)
        # Step 3: Loop over randomly selected "activate" clips and insert in background
        for random_activate in random_activates:
            # Insert the audio clip on the background
            background, segment_time = insert_audio_clip(background,random_activate,previous_segments)
            # Retrieve segment_start and segment_end from segment_time
            segment_start, segment_end = segment_time
            # Insert labels in "y"
            y = insert_ones(y,segment_end_ms=segment_end)
        ### END CODE HERE ###
    
        # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
        number_of_negatives = np.random.randint(0, 3)
        random_indices = np.random.randint(len(negatives), size=number_of_negatives)
        random_negatives = [negatives[i] for i in random_indices]
    
        ### START CODE HERE ### (≈ 2 lines)
        # Step 4: Loop over randomly selected negative clips and insert in background
        for random_negative in random_negatives:
            # Insert the audio clip on the background 
            background, _ = insert_audio_clip(background,random_negative,previous_segments)
        ### END CODE HERE ###
        
        # Standardize the volume of the audio clip 
        background = match_target_amplitude(background, -20.0)
    
        # Export new training example 
        file_handle = background.export("train" + ".wav", format="wav")
        print("File (train.wav) was saved in your directory.")
        
        # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
        x = graph_spectrogram("train.wav")
        
        return x, y
    
    x, y = create_training_example(backgrounds[0], activates, negatives)
    
    IPython.display.Audio("train.wav")
    IPython.display.Audio("audio_examples/train_reference.wav")
    plt.plot(y[0])
    
    # Load preprocessed training examples
    X = np.load("./XY_train/X.npy")
    Y = np.load("./XY_train/Y.npy")
    
    # Load preprocessed dev set examples
    X_dev = np.load("./XY_dev/X_dev.npy")
    Y_dev = np.load("./XY_dev/Y_dev.npy")
    
    from keras.callbacks import ModelCheckpoint
    from keras.models import Model, load_model, Sequential
    from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
    from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
    from keras.optimizers import Adam
    
    # GRADED FUNCTION: model
    
    def model(input_shape):
        """
        Function creating the model's graph in Keras.
        
        Argument:
        input_shape -- shape of the model's input data (using Keras conventions)
    
        Returns:
        model -- Keras model instance
        """
        
        X_input = Input(shape = input_shape)
        
        ### START CODE HERE ###
        
        # Step 1: CONV layer (≈4 lines)
        X = Conv1D(filters=196,kernel_size=15,strides=4)(X_input)     # CONV1D
        X = BatchNormalization()(X)                           # Batch normalization
        X = Activation('relu')(X)                             # ReLu activation
        X = Dropout(.8)(X)                                    # dropout (use 0.8)
    
        # Step 2: First GRU Layer (≈4 lines)
        X = GRU(units=128,return_sequences=True)(X)           # GRU (use 128 units and return the sequences)
        X = Dropout(.8)(X)                                    # dropout (use 0.8)
        X = BatchNormalization()(X)                           # Batch normalization
        
        # Step 3: Second GRU Layer (≈4 lines)
        X = GRU(units=128,return_sequences=True)(X)            # GRU (use 128 units and return the sequences)
        X = Dropout(.8)(X)                                     # dropout (use 0.8)
        X = BatchNormalization()(X)                            # Batch normalization
        X = Dropout(.8)(X)                                     # dropout (use 0.8)
        
        # Step 4: Time-distributed dense layer (see given code in instructions) (≈1 line)
        X = TimeDistributed(Dense(1,activation="sigmoid"))(X) # time distributed  (sigmoid)
    
        ### END CODE HERE ###
    
        model = Model(inputs = X_input, outputs = X)
        
        return model  
    
    model = model(input_shape = (Tx, n_freq))
    
    model.summary()
    
    model = load_model('./models/tr_model.h5')
    
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    
    model.fit(X, Y, batch_size = 5, epochs=1)
    
    loss, acc = model.evaluate(X_dev, Y_dev)
    print("Dev set accuracy = ", acc)
    
    def detect_triggerword(filename):
        plt.subplot(2, 1, 1)
    
        x = graph_spectrogram(filename)
        # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
        x  = x.swapaxes(0,1)
        x = np.expand_dims(x, axis=0)
        predictions = model.predict(x)
        
        plt.subplot(2, 1, 2)
        plt.plot(predictions[0,:,0])
        plt.ylabel('probability')
        plt.show()
        return predictions
    
    chime_file = "audio_examples/chime.wav"
    def chime_on_activate(filename, predictions, threshold):
        audio_clip = AudioSegment.from_wav(filename)
        chime = AudioSegment.from_wav(chime_file)
        Ty = predictions.shape[1]
        # Step 1: Initialize the number of consecutive output steps to 0
        consecutive_timesteps = 0
        # Step 2: Loop over the output steps in the y
        for i in range(Ty):
            # Step 3: Increment consecutive output steps
            consecutive_timesteps += 1
            # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
            if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
                # Step 5: Superpose audio and background using pydub
                audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
                # Step 6: Reset consecutive output steps to 0
                consecutive_timesteps = 0
            
        audio_clip.export("chime_output.wav", format='wav')
    
    IPython.display.Audio("./raw_data/dev/1.wav")
    IPython.display.Audio("./raw_data/dev/2.wav")
    filename = "./raw_data/dev/1.wav"
    prediction = detect_triggerword(filename)
    chime_on_activate(filename, prediction, 0.5)
    IPython.display.Audio("./chime_output.wav")
    filename  = "./raw_data/dev/2.wav"
    prediction = detect_triggerword(filename)
    chime_on_activate(filename, prediction, 0.5)
    IPython.display.Audio("./chime_output.wav")
    
    # Preprocess the audio to the correct format
    def preprocess_audio(filename):
        # Trim or pad audio segment to 10000ms
        padding = AudioSegment.silent(duration=10000)
        segment = AudioSegment.from_wav(filename)[:10000]
        segment = padding.overlay(segment)
        # Set frame rate to 44100
        segment = segment.set_frame_rate(44100)
        # Export as wav
        segment.export(filename, format='wav')
    
    your_filename = "audio_examples/my_audio.wav"
    
    preprocess_audio(your_filename)
    IPython.display.Audio(your_filename) # listen to the audio you uploaded 
    
    chime_threshold = 0.5
    prediction = detect_triggerword(your_filename)
    chime_on_activate(your_filename, prediction, chime_threshold)
    IPython.display.Audio("./chime_output.wav")
#trig_word_1()

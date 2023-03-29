import numpy as np
import matplotlib.pyplot as plt
from math import exp
from matplotlib.colors import ListedColormap

# Data format: vector of [[x1, x2], class]

# Creating the logistic regression model

# Helper function to normalize data
def normalize(data):
    vec = data
    xs = []
    ys = []
    for element in data:
        xs.append(element[0][0])
        ys.append(element[0][1])
    meanx = sum(xs) / len(xs)
    meany = sum(ys) / len(ys)
    for element in vec:
        element[0][0] -= meanx
        element[0][1] -= meany
    return vec


# Method to make predictions
def predict(data, b0, b1, b2):
    return np.array([1 / (1 + exp(-1 * b0 + -1 * b1 * element[0][0] + -1 * b2 * element[0][1])) for element in data])

# Method to train the model
def logistic_regression(data, labels):

    vec = normalize(data)

    # Initializing variables
    b0 = 0
    b1 = 0
    b2 = 0
    L = 0.01
    epochs = 2000

    for epoch in range(epochs):
        y_pred = predict(vec, b0, b1, b2)
        x = []
        y = []
        for i in range(len(vec)):
            x.append(vec[i][0][0])
            y.append(vec[i][0][1])
        D_b0 = -2 * sum((labels - y_pred) * y_pred * (1 - y_pred))      # Derivative of loss wrt b0
        D_b1 = -2 * sum(x * (labels - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1
        D_b2 = -2 * sum(y * (labels - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b2
        b0 = b0 - L * D_b0
        b1 = b1 - L * D_b1
        b2 = b2 - L * D_b2
    
    return b0, b1, b2

def visualize_logistic_regression(train_data, test_data):

    # Training the model
    train_labels = np.array([ element[1] for element in train_data ])
    b0, b1, b2 = logistic_regression(train_data, train_labels)

    # Making predictions
    test_data_norm = normalize(test_data)
    y_pred = predict(test_data_norm, b0, b1, b2)
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

    test_data_pred = test_data
    for i in range(len(test_data_pred)):
        if y_pred[i] == test_data[i][1]:
            test_data_pred[i][1] = y_pred[i]
        else:
            test_data_pred[i][1] = 2  # recognize error

    train_class0 = []
    train_class1 = []
    for element in train_data:
        if element[1] == 0:
            train_class0.append([element[0][0], element[0][1]])
        else:
            train_class1.append([element[0][0], element[0][1]])

    test_class0 = []
    test_class1 = []
    test_class_error = []
    for element in test_data_pred:
        if element[1] == 0:
            test_class0.append([element[0][0], element[0][1]])
        elif element[1] == 1:
            test_class1.append([element[0][0], element[0][1]])
        else:
            test_class_error.append([element[0][0], element[0][1]])

    plt.clf()

    plt.scatter([train_class0[i][0] for i in range(len(train_class0))], 
                [train_class0[i][1] for i in range(len(train_class0))],
                c='#6A5ACD')

    plt.scatter([train_class1[i][0] for i in range(len(train_class1))], 
                [train_class1[i][1] for i in range(len(train_class1))],
                c='#000000')

    plt.scatter([test_class0[i][0] for i in range(len(test_class0))], 
                [test_class0[i][1] for i in range(len(test_class0))],
                c='#FF00FF')

    plt.scatter([test_class1[i][0] for i in range(len(test_class1))], 
                [test_class1[i][1] for i in range(len(test_class1))],
                c='#AAAAAA')

    plt.scatter([test_class_error[i][0] for i in range(len(test_class_error))], 
                [test_class_error[i][1] for i in range(len(test_class_error))],
                c='#DC143C')

    xs = []
    for element in train_data:
        xs.append(element[0][0])
    for element in test_data:
        xs.append(element[0][0])
    x_min, x_max = min(xs), max(xs)

    x_axis = np.linspace(x_min, x_max)
    y_axis = -(b1 / b2) * x_axis - b0 / b2
    plt.plot(x_axis, y_axis, c='r')

    plt.title('Recognizing Result')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()  

    # f1-measure
    f1_m = f1_measure(test_data, y_pred)

def f1_measure(test_data, y_pred):
    tn, tp, fn, fp = 0, 0, 0, 0

    for i in range(len(test_data)):
        if y_pred[i] == test_data[i][1]: # true prediction
            if y_pred[i] == 0: # positive
                tp += 1
            else:              # negative
                tn += 1          
        else:                            # false prediction
            if y_pred[i] == 1: # negative
                fn += 1
            else:              # positive
                fp += 1
    f1_m = tp/(tp + (fn + fp) / 2)
    print(f"\nTP (true positive) = {tp}, TN (true negative) = {tn}")
    print(f"FP (false positive) = {fp}, FN (false negative) = {fn}")
    print(f"Acurracy = (TP + TN)/(TP + TN + FP + FN) = {(tn + tp)/(tn + tp + fp + fn)}")
    print(f"Error rate = (FP + FN)/(TP + TN + FP + FN) = {(fn + fp)/(tn + tp + fp + fn)}")
    print(f"Presicion = TP/(TP + FP) = {tp/(tp + fp)}")
    print(f"Recall = TP/(TP + FN) = {tp/(tp + fn)}")
    print(f"F1 measure = TP/(TP + (FN + FP) / 2) = {f1_m}\n")

    return f1_m

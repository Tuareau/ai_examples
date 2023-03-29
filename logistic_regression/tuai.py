import numpy as np
import random
import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# TData generator (with center, 1 class)
def generate_set_of_data(number_of_class_elements, class_number, center):
    data = []
    for element in range(number_of_class_elements):
        data.append([ [random.gauss(center[0], 0.5), random.gauss(center[1], 0.5)], class_number])
    return data

# Data generator (random center)
def generate_data(number_of_class_elements, number_of_classes):
    data = []
    for class_number in range(number_of_classes):
        # Choose random center of 2-dimensional gaussian
        center_x, center_y = random.random() * 5.0, random.random() * 5.0
        # Choose random nodes with RMS = 0.5
        for element in range(number_of_class_elements):
            data.append([ [random.gauss(center_x, 0.5), random.gauss(center_y, 0.5)], class_number])
    return data

# Data generator (fixed center)
def generate_data_with_center(number_of_class_elements, number_of_classes, center):
    data = []
    for class_number in range(number_of_classes):
        for element in range(number_of_class_elements):
            data.append([ [random.gauss(center[0], 0.5), random.gauss(center[1], 0.5)], class_number])
    return data

def show_data(train_data, test_data):
    class_colormap  = ListedColormap(['#6A5ACD', '#00FF00', '#000000'])
    test_colormap   = ListedColormap(['#FF00FF', '#AAFFAA', '#AAAAAA'])
    plt.scatter([train_data[i][0][0] for i in range(len(train_data))],
                [train_data[i][0][1] for i in range(len(train_data))],
                c = [train_data[i][1] for i in range(len(train_data))],
                cmap = class_colormap)
    plt.scatter([test_data[i][0][0] for i in range(len(test_data))],
            [test_data[i][0][1] for i in range(len(test_data))],
            c = [test_data[i][1] for i in range(len(test_data))],
            cmap = test_colormap)
    plt.title('Random Dataset')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()  

# Separate N data elements in two parts:
# test_data with N * test_percent elements
# train_data with N * (1.0 - test_percent) elements
def split_train_test(data, test_percent):
    train_data = []
    test_data  = []
    for element in data:
        if random.random() < test_percent:
            test_data.append(element)
        else:
            train_data.append(element)
    return train_data, test_data	 

# Calculate classification accuracy
def calculate_accuracy(test_data, test_data_labels):
    print("Accuracy: ", 
          sum([int(test_data_labels[i] == test_data[i][1]) for i in range(len(test_data))]) / float(len(test_data)))

# f1-метрика

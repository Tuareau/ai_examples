# Numerical computing
import numpy as np
import tensorflow as tf

# Spliting data into train and test
from sklearn.model_selection import train_test_split

# Visualization
from matplotlib import pyplot as plt
# Different colors for cmap
import matplotlib.cm as cm
import seaborn as sns
sns.set()

import tuai

# First we imported logistic regression
from sklearn.linear_model import LogisticRegression

#np.random.seed(34)

#x1 = np.random.randn(500)*0.5+3
#x2 = np.random.randn(500)*0.5+2

#x3 = np.random.randn(500) *0.5 + 4
#x4 = np.random.randn(500) *0.5 + 5

data = tuai.generate_data(500, 2)
train_data, test_data = tuai.split_train_test(data, 0.2)

x1 = [train_data[i][0][0] for i in range(len(train_data))]
x2 = [train_data[i][0][1] for i in range(len(train_data))]

x3 = [train_data[i][0][0] for i in range(len(test_data))]
x4 = [train_data[i][0][1] for i in range(len(test_data))]

# Creating a matrix
X_1 = np.vstack([x1, x2])
X_2 = np.vstack([x3, x4])
X = np.hstack([X_1, X_2]).T
print(X.shape)

# Y true labels
# create classes (0, 1)
y = np.hstack([np.zeros(500), np.ones(500)])

# check the shape of input data and labels
print("Shape of X is: ", X.shape)
print("Shape of y is: ", y.shape)

plt.scatter(X[:,0], X[:,1], c=y, cmap=cm.coolwarm, edgecolors='w');
plt.title('Random Dataset')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show();

# Train -> 80%, Test -> 20%
# This returns our dataset split into training and test examples
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    return X_train, X_test, y_train, y_test
# Spliting our data
X_train, X_test, y_train, y_test = split_data(X, y)

# Reshape our label to avoid having a rank-1 array (n,)
# Don't use rank-1 arrays when implement logistic regression, instead use a rank-2 arrays (n, 1)
# We are also making sure our datatypes are converted into float32.

# Our vectorized labels
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

print('x_train:\t{}'.format(X_train.shape))
print('y_train:\t{}'.format(y_train.shape))
print('x_test:\t\t{}'.format(X_test.shape))
print('y_test:\t\t{}'.format(y_test.shape))

# Initialized our model to a variable
LR = LogisticRegression()

# We trained our model with our random dataset
LR.fit(X_train, y_train)

z_prob = LR.predict_proba(X_train)

z_prediction = LR.predict(X_test)

score = sum(z_prediction == y_test[:, 0]) / len(y_test)
print("Accuracy score : {:.2f}%".format(score*100))

w1_final = LR.coef_[0][0]
w2_final = LR.coef_[0][1]
b_final =  LR.intercept_[0]

print("Weight 1 : ",w1_final)
print("Weight 2 : ",w2_final)
print("Bias : ",b_final)

plt.figure(figsize =(10,7))

plt.scatter(X_train[:, 0], X_train[:, 1], c= z_prob[:, 0], cmap=cm.coolwarm)
plt.title('Training Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show();

# Training dataset visualization

plt.figure(figsize =(10,7))

x_axis = np.linspace(1, 6)
yy_lr = -(w1_final/w2_final)*x_axis - b_final/w2_final
plt.plot(x_axis, yy_lr, label = 'Logistic Regression Line', c='r')

plt.scatter(X_train[:, 0], X_train[:, 1], c= y_train[:, 0], cmap=cm.coolwarm)
plt.title('Training Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show();

# Test dataset visualization
plt.figure(figsize =(10,7))
x_axis = np.linspace(1, 6)

yy_lr = -(w1_final/w2_final)*x_axis - b_final/w2_final

plt.plot(x_axis, yy_lr, label = 'Logistic Regression Line', c='r')
plt.scatter(X_test[:, 0], X_test[:, 1], c= z_prediction, cmap=cm.coolwarm)
plt.title('Test Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show();
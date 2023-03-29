# Numerical computing
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Spliting data into train and test
from sklearn.model_selection import train_test_split

# Visualization
from matplotlib import pyplot as plt
# Different colors for cmap
import matplotlib.cm as cm
import seaborn as sns
sns.set()

np.random.seed(34)

x1 = np.random.randn(500)*0.5+3
x2 = np.random.randn(500)*0.5+2

x3 = np.random.randn(500) *0.5 + 3
x4 = np.random.randn(500) *0.5 + 3

# Creating a matrix

X_1 = np.vstack([x1, x2])
X_2 = np.vstack([x3, x4])
X = np.hstack([X_1, X_2]).T
print(X.shape)

# Y true labels
# create classes (0, 1)
y = np.hstack([np.zeros(500), np.ones(500)])

plt.scatter(X[:,0], X[:,1], c=y, cmap=cm.coolwarm, edgecolors='w');
plt.title('Random Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
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

# Hyper-parameters
learning_rate = 0.72 # The optimization initial learning rate
num_epochs =  5000 # Total number of training epochs
display_step = 1

# data format is as usual:
# X_train and X_train have shape (num_instances, dimension_of_features)
# Y_train and Y_test have shape (num_instances, dimension_of_classes)

dimension_of_features = X_train.shape[1]
dimension_of_classes = y_train.shape[1]

print("Dimension of our features : ",dimension_of_features)
print("Dimension of our class : ",dimension_of_classes)

# Create the graph for the linear model
# Placeholders for inputs (data) and outputs(target)
data = tf.placeholder(dtype=tf.float32, shape=[None, dimension_of_features], name='X_placeholder')
target = tf.placeholder(dtype=tf.float32, shape=[None, dimension_of_classes], name='Y_placeholder')

W = tf.Variable(tf.random_normal([dimension_of_features, dimension_of_classes])* 0.01, name='weights') # W - weights
b = tf.Variable(tf.zeros([dimension_of_classes, dimension_of_classes]), name='bias') # b - bias 

# Add summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

# Forward-propagation
# Define a simple logistic model z=wx+b
# z_pred will contain predictions the model makes.
Z = tf.add(tf.matmul(data, W) , b)

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z, labels=target))

# Create a summary to monitor the loss function
tf.summary.scalar("loss_function", loss)

# Back-propagation
optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='Gradient-optimizer').minimize(loss)

# Calculate the correct predictions
prediction = tf.nn.sigmoid(Z)

# If the prediction is greater than 0.5, it should be considered as class 1, otherwise class 0
correct_prediction = tf.equal(target , (tf.to_float(tf.greater(prediction, 0.5))))
                              
# Calculate our models performance, but first we need to convert our datatype from true and false, into 1 and 0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create the operation for initializing all variables
init = tf.global_variables_initializer()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

with tf.Session() as session:

    # run the initialization
    session.run(init)
    
    # visualize on tensorboard
    # tensorboard --logdir='logistic_regression'
    summary_writer = tf.summary.FileWriter('logistic_regression',session.graph)
    
    # keep track of the loss, weight and bias for visualization
    loss_plot = []
    weight_final = []
    bias_final = []
    
    # training loop
    for epoch in range(num_epochs):

        # feeding data to our placeholders
        feed_dict_train = {data: X_train, target: y_train}
        _ , c, prediction_values = session.run([optimizer, loss, prediction], feed_dict=feed_dict_train)
        
        # Save the loss result
        loss_plot.append(c)
        
        # Display logs per 1000 epoch step
        if epoch % 1000 == 0:
            
            print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(c),\
                "W=", session.run(W), "b=", session.run(b))
        
        # Write logs for each epoch
        summary_str = session.run(merged_summary_op, feed_dict=feed_dict_train)  
        summary_writer.add_summary(summary_str,  epoch)
    
    # Store our final weigh and bias.
    weight_final = session.run(W)
    bias_final = session.run(b)
    
    print("\nOptimization Finished!\n")
    print ("Train Accuracy:", accuracy.eval({data: X_train, target: y_train}))
    print ("Test Accuracy:", accuracy.eval({data: X_test, target: y_test}))

plt.plot(loss_plot)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show();

max_loss = np.max(loss_plot)
min_loss = np.min(loss_plot)
print("Maximum Loss : {:.4f}\n".format(max_loss))
print("Minimum Loss : {:.4f}\n".format(min_loss))
print("Loss difference : {:.4f}\n".format(max_loss-min_loss))

w1_final = weight_final[0, -1]
w2_final = weight_final[1, -1]
b_final =  bias_final[-1,-1]

print("Weight 1 : ",w1_final)
print("Weight 2 : ",w2_final)
print("Bias : ",b_final)

plt.figure(figsize =(10,7))
plt.scatter(X_train[:,0], X_train[:,1], c=prediction_values[:, 0], cmap=cm.coolwarm)
plt.title('Probability of our Training Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show();

# Training data
x_axis = np.linspace(1, 6)
yy_lr = -(w1_final/w2_final)*x_axis - b_final/w2_final

plt.figure(figsize =(10,7))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train[:, 0], cmap=cm.coolwarm)
plt.plot(x_axis, yy_lr, label = 'Logistic Regression Line', c='r')
plt.title('Train Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show();

#Test data
x_axis = np.linspace(1, 6)
yy_lr = -(w1_final/w2_final)*x_axis - b_final/w2_final

plt.figure(figsize =(10,7))
plt.scatter(X_test[:,0], X_test[:,1], c=y_test[:, 0], cmap=cm.coolwarm)
plt.plot(x_axis, yy_lr, label = 'Logistic Regression Line', c='r')
plt.title('Test Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show();
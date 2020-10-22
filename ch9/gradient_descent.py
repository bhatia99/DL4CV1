import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100,
                help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
args = vars(ap.parse_args())

# uncomment below line of code if you don't wish to use argument parser
#args = {'epochs':1000,'alpha':0.001}

# make_blobs() is used to create your own custom dataset where data points are
# scattered in shape of a blob(a drop of liquid) or as stated in the official
# documentation to "Generate isotropic Gaussian blobs for clustering"
# the no. of blobs(drops) is determined by the centres parameter
# similarly, there exist other functions to create datasets distributed in
# some other shapes like make_circle, make_moon

# generate a 2-class classification problem with 1,000 data points, 
# where each data point is a 2D feature vector
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                  cluster_std=1.5, random_state=1)
# both X,y returned have numpy's ndarray datatype
# this array datatype is perfect if X,y are multi-dimensional
# for X, the dimensions(or features) are determined by n_features parameter
# and in our case, X will have 2 dimensions so array datatype for X is fine
# But for y, a one dimensional "vector" (not array) is returned
# which may or may not cause an error due to its shape,
# so try changing its shape if there is an error
# the difference b/w 1D vector and 1D array is that their shapes are as
# vector -> (10,) ; array -> (10,1)
# the no. of columns is empty in vector's shape
y = y.reshape((y.shape[0], 1))

# plot the generated dataset 
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.figure()
plt.title("Generated Dataset")
# parameters: "marker" decides the shape of data point
# "s" decides the size of marker
# "c" colors the data points based on categories in y
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=30, cmap=plt.cm.Spectral)
plt.show()

# using the Bias trick
# insert a column of 1’s as the last entry in the feature 
# matrix -- this little trick allows us to treat the bias
# as a trainable parameter within the weight matrix 
# we do this using np.c_() which stacks 1-D arrays as columns into a 2-D array
# this function is similar to np.column_stack()
# using column_stack() we do this as: 
# X = np.column_stack((X,np.ones(X.shape[0])))
import numpy as np
X = np.c_[X, np.ones((X.shape[0]))]

# splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses 
# i am still a little confused as to how the no.of rows for the weight matrix
# was chosen but I guess it should be equal to no. of features
# here, we have dataset with 2 features + 1 for bias trick
# so 3 rows, given by X.shape[1]
# X.shape[0] -> no. of data points
# X.shape[1] -> no. of features for a data point + 1 for bias trick
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)

# the sigmoid_activation() acts as our scoring function.
# The scoring function accepts our data as an input and maps the data to class labels
# by outputting some score
def sigmoid_activation(x):
# compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))

losses = []
#loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    # take the dot product between our features ‘X‘ and the weight 
    # matrix ‘W‘, then pass this value through our sigmoid activation 
    # function, thereby giving us our predictions on the dataset 
    preds = sigmoid_activation(X_train.dot(W))
    
    # now that we have our predictions, we need to determine the 
    # ‘error‘, which is the difference between our predictions and
    # the true values
    error = preds - y_train
    # here our loss function is least squares error (different from MSE, 
    # here we are not taking mean)
    loss = np.sum(error ** 2)
    losses.append(loss)
    
    # the gradient descent update is the dot product between our
    # features and the error of the predictions 
    #To understand how gradient is being calculated as shown below, give a read to the following articles:
    # https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e
    # https://medium.com/deep-math-machine-learning-ai/chapter-1-2-gradient-descent-with-math-d4f2871af402
    # you'll realise that gradient comes of the form x*Error for a cost function of form (y_pred-y_true)^2
    gradient = X_train.T.dot(error)
    
    # in the update stage, all we need to do is "nudge" the weight
    # matrix in the negative direction of the gradient (hence the 
    # term "gradient descent" by taking a small step towards a set
    # of "more optimal" parameters)
    W += -args["alpha"] * gradient

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),loss))
        
# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

# our prediction funtion        
def predict(X,W):
    # take the dot product between our features and our final weight matrix
    # in the book, Adrian has not applied the sigmoid_function for predicting test cases
    # but according to me, it should have been applied since the thresholding that we apply
    # in the next lines take thresh as 0.5 which is chosen keeping in mind that
    # sigmoid_function has range for y:(0,1) and is y=0.5 at x=0
    preds = sigmoid_activation(X.dot(W))
    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    
    # return the predictions
    return preds

# evaluate our model 
from sklearn.metrics import classification_report
print("[INFO] evaluating...")
preds = predict(X_test, W)
print(classification_report(y_test, preds))

# plot the (testing) classification data 
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=np.squeeze(y_test), s=30, cmap=plt.cm.Spectral)
# Starter code copied from:
# https://github.com/mGalarnyk/Python_Tutorials/blob/master/
# Sklearn/Logistic_Regression/LogisticRegression_MNIST_Codementor.ipynb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # for pretty confusion matrix

from sklearn import metrics
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import cv2

###
# HYPERPARAMETERS
###
HOUGH_PARAM1 = 20
HOUGH_PARAM2 = 10

# download MNIST data
mnist = fetch_mldata('MNIST original')

# mnist.data is the (N by p) design matrix,
# where each feature is currently just the
# grayscale value of a pixel
design_matrix = mnist.data

N = design_matrix.shape[0]  # number of images
NUM_PIXELS = design_matrix.shape[1]  # number of pixels

# TODO: CONVERT FROM GRAYSCALE TO BINARY

# add additional features to mnist data

# for now use hough Transform via openCV
# to compute the number of circles in each image

num_circles = []
for image_row in mnist.data:
    img = np.reshape(image_row, (28, 28))

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 2,
                               param1=HOUGH_PARAM1,
                               param2=HOUGH_PARAM2,
                               minRadius=0,
                               maxRadius=0)
    if circles is None:
        num_circles.append(0)
    else:
        numCircles = circles.shape[1]
        num_circles.append(numCircles)

# convert to ndarray
num_circles = np.reshape(np.array(num_circles), (N, 1))

# append feature column to design matrix
design_matrix = np.hstack((design_matrix, num_circles))

# Add a column of 1s to our design matrix,
# for the intercept of the underlying linear
# model that logistic regression will use
# sklearn automatically does this according to
# the docs, but this seems to improve accuracy anyway?
design_matrix = np.hstack((np.ones(shape=(N, 1)), design_matrix))

# Split up data into training and test sets
train_img, test_img, train_lbl, test_lbl = train_test_split(
    design_matrix, mnist.target, test_size=1/7.0, random_state=0)

# Visualize a few of the training images
plt.figure(figsize=(20, 4))
for index, (row, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):

    img = np.reshape(row[1:NUM_PIXELS + 1], (28, 28))  # grab just the pixel data
    img = img.astype('uint8')  # need to convert from float
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # add BGR color channels

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 2,
                               param1=HOUGH_PARAM1,
                               param2=HOUGH_PARAM2,
                               minRadius=0,
                               maxRadius=0)  # find circles in image

    if circles is not None:
        circles = np.uint16(np.around(circles))  # round coordinates to integers
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    plt.subplot(1, 5, index + 1)
    plt.imshow(cimg, cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)

plt.show()

# create the model object
# all parameters not specified are set to their defaults
# default solver is incredibly slow thats why we change it
logisticRegr = LogisticRegression(solver='lbfgs')

# train the model
logisticRegr.fit(train_img, train_lbl)

# Make predictions on test set
predictions = logisticRegr.predict(test_img)

# Calculate test accuracy
score = logisticRegr.score(test_img, test_lbl)
print("Test Accuracy: %.2f%%" % (score * 100))

# Display Confusion Matrix (via Seaborn)
cm = metrics.confusion_matrix(test_lbl, predictions)

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)

plt.show()

# Display misclassified images with predicted labels
index = 0
misclassifiedIndexes = []
for label, predict in zip(test_lbl, predictions):
    if label != predict:
        misclassifiedIndexes.append(index)
    index += 1

plt.figure(figsize=(20, 4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(test_img[badIndex][1:NUM_PIXELS+1], (28, 28)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], test_lbl[badIndex]), fontsize=15)

plt.show()

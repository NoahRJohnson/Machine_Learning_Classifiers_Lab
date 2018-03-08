# Load in required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import KFold
#from xgboost import XGBClassifier

################################################################
#                       HYPERPARAMETERS                        #

LOWER_BOUND_DEPTH = 1
UPPER_BOUND_DEPTH = 5
DEPTH_ITER = 1

LOWER_BOUND_LAMBDA = 0
UPPER_BOUND_LAMBDA = 1.0
LAMBDA_ITER = 0.1

LOWER_BOUND_B = 1
UPPER_BOUND_B = 100
B_ITER = 10

################################################################


# Downloads the data if it's not already
mnist = fetch_mldata('MNIST original')

# Set number of folds
kf = KFold(n_splits=3, shuffle=True, random_state = 0)

CV_errors = []

# This is inefficient, but simple. TODO: Optimize
for b in np.arange(LOWER_BOUND_B, UPPER_BOUND_B + 1, B_ITER):
    for d in np.arange(LOWER_BOUND_DEPTH,
                   UPPER_BOUND_DEPTH + 1, DEPTH_ITER):
        for l in np.arange(LOWER_BOUND_LAMBDA,
                       UPPER_BOUND_LAMBDA + 1, LAMBDA_ITER):

            # Split the data into K folds
            for train_index, test_index in kf.split(mnist.data):

                train_img, test_img = mnist.data[train_index], mnist.data[test_index]

                train_lbl, test_lbl = mnist.target[train_index], mnist.target[test_index]

                # fit model
                #model = XGBClassifier(max_depth=D, learning_rate=LAMBDA, n_estimators=B)
                #model.fit(train_img, train_lbl)
                
                # Check how well the model does, and store it
                #score = model.score(test_img, test_lbl)
                
                score = np.random.randint(1,10)
                
                CV_errors.append(score)
                
                print("Num_Trees: %d, Depth: %d, Lambda: %f, Accuracy: %f" %
                      (b, d, l, score))


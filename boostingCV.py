# Load in required libraries
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import KFold
#from xgboost import XGBClassifier

##############################################################################
#                       HYPERPARAMETERS                                      #
#                                                                            #
#                    ALL BOUNDS INCLUSIVE                                    #
##############################################################################
LOWER_BOUND_DEPTH = 1
UPPER_BOUND_DEPTH = 6
DEPTH_ITER = 1

LOWER_BOUND_LAMBDA = 0.01
UPPER_BOUND_LAMBDA = 0.13
LAMBDA_ITER = 0.04

LOWER_BOUND_NUM_TREES = 100
UPPER_BOUND_NUM_TREES = 500
NUM_TREES_ITER = 50

NUM_FOLDS = 5  # k, the number of folds used in cross validation

##############################################################################


# Download the data
mnist = fetch_mldata('MNIST original')

# Set number of folds
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state = 0)

# This list will store all accuracy scores for different
# hyperparameter combinations
cv_scores = []

# This is simple, but inefficient. TODO: Optimize
for b in np.arange(LOWER_BOUND_NUM_TREES, UPPER_BOUND_NUM_TREES + 1,
                   NUM_TREES_ITER):
    for d in np.arange(LOWER_BOUND_DEPTH,
                   UPPER_BOUND_DEPTH + 1, DEPTH_ITER):
        for l in np.arange(LOWER_BOUND_LAMBDA,
                       UPPER_BOUND_LAMBDA + 1, LAMBDA_ITER):

            k_fold_scores = []
            
            # Split the data into K folds, and train the model K times
            for train_index, test_index in kf.split(mnist.data):

                train_img, test_img = mnist.data[train_index], mnist.data[test_index]

                train_lbl, test_lbl = mnist.target[train_index], mnist.target[test_index]

                # fit model
                #model = XGBClassifier(max_depth=D, learning_rate=LAMBDA, n_estimators=B)
                #model.fit(train_img, train_lbl)
                
                # Check how well the model does, and store it
                #score = model.score(test_img, test_lbl)
                
                score = np.random.randint(1,10)
                
                k_fold_scores.append(score)
                
            # Average the scores across the k folds
            cv_score = np.mean(k_fold_scores)
            
            cv_scores.append((b, d, l, cv_score))
            
            print("Num_Trees: %d, Depth: %d, Lambda: %f, Accuracy: %f" %
                  (b, d, l, cv_score))



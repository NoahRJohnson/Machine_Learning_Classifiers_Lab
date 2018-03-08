# Load in required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
from xgboost import XGBClassifier

################################################################
#                       HYPERPARAMETERS                        #

DEPTH = 2

LAMBDA = 0.1

NUM_TREES = 100

################################################################


# Downloads the data if it's not already
mnist = fetch_mldata('MNIST original')

# Make sure we have the full set (should be 70k images at 28x28 each)
print("Image Data Shape", mnist.data.shape)
print("Label Data Shape", mnist.target.shape)

# Split the data into training and testing sets
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

# Check the shapes of those sets
print("Train img shape", train_img.shape)
print("Train lbl shape", train_lbl.shape)
print("Test img shape", test_img.shape)
print("Test lbl shape", test_lbl.shape)

# View the first few images and labels
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

# fit model on training data
model = XGBClassifier(max_depth=DEPTH, learning_rate=LAMBDA, n_estimators=NUM_TREES)
model.fit(train_img, train_lbl)

predictions = model.predict(test_img)

# Check how well the model does
score = model.score(test_img, test_lbl)
print("Score", score)

# Check which images the model misclassified
index = 0
misclassifiedIndexes = []
for label, predict in zip(test_lbl, predictions):
    if label != predict:
        misclassifiedIndexes.append(index)
    index += 1

plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(test_img[badIndex], (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], test_lbl[badIndex]), fontsize = 15)


cm = metrics.confusion_matrix(test_lbl, predictions)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

plt.show()

'''
Handwriting Recognition using Random Forest Classsifier
'''

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits # digits database

digits = load_digits()

''' Analyse a sample image from the dataset '''

import pylab as pl  # PyLab is actually embedded inside Matplotlib
pl.gray() 
pl.matshow(digits.images[0]) 
pl.show()
# OR # plt.imshow(digits.images[0])

''' Analyze image pixels '''

# Each element represents the pixel of our greyscale image. The value ranges from 0 to 255 for an 8 bit pixel.
digits.images[0]

''' Visualize first 15 images '''

# List as it is easier to iterate. 
# The purpose of zip() is to map the similar index of multiple containers so that they can be used just using as single entity.
# unzip using '*'
images_and_labels = list(zip(digits.images, digits.target))
# images are the handwritten digits and target is the actual digit
plt.figure(figsize=(5,5)) # size of plot
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)
    
''' Prediction Algorithm '''

import random
from sklearn.ensemble import RandomForestClassifier as rf

# Define variables
n_samples = len(digits.images) # total number of images
x = digits.images.reshape((n_samples, -1))
y = digits.target

# Create train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Using the Random Forest Classifier
classifier = rf()

#Fit model with sample data
classifier.fit(x_train, y_train)

#Attempt to predict validation data
score=classifier.score(x_test, y_test)
print('Random Tree Classifier: ')
print('Score\t', str(score))

i=250 # Change image values accordingly
pl.gray() 
pl.matshow(digits.images[i]) 
pl.show() 
y_pred = classifier.predict([x[i]]) # give in square brackets if 'Expected 2D array, got 1D array instead' error comes up
print('The image is a ',y_pred)

''' Extra '''

# Making the Confusion Matrix
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
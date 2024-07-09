import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit 
import os
from functions import *

print("The script will now combine the data in all .pkl files into a usable format.\n")
# features, target, encoder, and array of drawing numbers 
[X, y, labels_le, draw_nums] = makeAllData()

print("...done!\n")

print("The test_size is a number between 0.0 and 1.0. \n")
while True:
    try:
        test_size = float(input("Input test size: "))
    except ValueError:
        print("Invalid input. Input test size: ")
    else:
        break

print("Splitting the dataset...\n")
[X_train, X_test, y_train, y_test] = splitDatasetByDrawing(X, y, test_size, draw_nums)

print("...done!\n")

print("Running logistic regression...\n")

lr_file = logisticRegression(X_train, X_test, y_train, y_test, labels_le)

print("The file can be found at: ", lr_file)
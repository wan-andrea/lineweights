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

pklToLst("pkl_data\\0.pkl")
print("Successful!\n")
all_files = os.listdir("pkl_data")
print(all_files)
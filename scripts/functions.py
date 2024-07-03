import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier

# function to open files and store as data
# Inputs: file location as str
# Outputs: data as a 2D list

def pklToLst(fileLocation):
    with open(fileLocation, 'rb') as filein:
        data = pickle.load(filein)
    return data

# function to convert non-int inputs to ints
# Inputs: data as 2D list
# Outputs: a list with elements: features as pandas dataframe with int-only values, target as np array, all the label encoders

def makeData(data):
    labels_str = data[0] # a list of strs
    crv_type_str = data[1] # a list of strs
    crv_closed = data[2] # ints 0 or 1 representing bools
    crv_deg = data[3] # int
    crv_def = data[4] # int
    crv_per = data[5] # int
    crv_span = data[6] # int
    crv_ctrl = data[7] # int
    crv_dist = data[8] # int
    crv_norm = data[9] # color as int
    crv_zbuff = data[10] # value as int
    crv_rid_str = data[11] # id in rhino as str
    crv_ind = data[12] # index as int

    # convert non-ints to int
    labels_le = LabelEncoder()
    labels = labels_le.fit_transform(labels_str)

    crv_type_le = LabelEncoder()
    crv_type = crv_type_le.fit_transform(crv_type_str)

    crv_rid_le = LabelEncoder()
    crv_rid = crv_rid_str.fit_transform(crv_rid_le)

    # make dataframe
    features = pd.DataFrame({
    'crv_type_int': crv_type,
    'crv_closed': crv_closed,
    'crv_deg': crv_deg,
    'crv_def': crv_def,
    'crv_per': crv_per,
    'crv_span': crv_span,
    'crv_ctrl': crv_ctrl,
    'crv_dist': crv_dist,
    'crv_norm': crv_norm,
    'crv_zbuff': crv_zbuff,
    'crv_ind': crv_ind
    })
    
    features.set_index('crv_ind', inplace=True) # sets rhino id as id

    # make target
    target = np.array(labels)

    return [features, target, labels_le] # features = X, target = y

# train_test_split wrapper
def splitDataset(X, y, test_size):
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)

# helper function for different models
# predictions to list 
def predictionsToLst(X_test, y_pred, labels_le):
    # get ids of predictions
    pred_ids = X_test.index[y_pred]
    pred_ids_lst = pred_ids.tolist() # convert to list
    y_pred_lst = labels_le.inverse_transform(y_pred).tolist()
    return [pred_ids_lst, y_pred_lst]

# helper function
# save predictions back to pickled file for use in Grasshopper
# Inputs: save location, file name, and the contents in lst
# (for use with predictionsTolst)
# Outputs: save location of the pickled file
def toGrasshopper(path, name, lst):
    save = path + name + ".pkl"
    pickle.dump(lst, open(save, 'wb'))
    return save

# logistic regression
# Inputs: features, target, test_size, and labels encoder
# Output: the file location of pickled predictions
def logisticRegression(X_train, X_test, y_train, y_test, labels_le):

    # make the model
    model = LogisticRegression().fit(X_train, y_train)
   
    # make predictions 
    y_pred = model.predict(X_test)
    lst = predictionsToLst(X_test, labels_le)
    y_pred_lst = lst[1]

    # prints the predictions and actual
    print("Predicted: ", y_pred_lst, "\n")
    print("Actual: ", y_test, "\n") # actual

    # pickle and save the rids and labels for return to Rhino
    path = input("Enter file path: \n")
    name = input("Enter file name: \n")
    return toGrasshopper(path, name, lst)

# decision trees
# Inputs: features, target, test_size, and labels encoder
# Output: the file location of pickled predictions
def decisionTrees(X_train, X_test, y_train, y_test, labels_le):
    
    # make the model
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)
    lst = predictionsToLst(X_test, labels_le)
    y_pred_lst = lst[1]

    # prints the predictions and actual
    print("Predicted: ", y_pred_lst, "\n")
    print("Actual", y_test, "\n") # actual

    # pickle and save the rids and labels for return to Rhino
    path = input("Enter file path: \n")
    name = input("Enter file name: \n")
    return toGrasshopper(path, name, lst)
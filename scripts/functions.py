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

# function to open files and store as data
# Inputs: file location as str
# Outputs: data as a 2D list

def pklToLst(fileLocation):
    with open(fileLocation, 'rb') as filein:
        data = pickle.load(filein)
    return data

# function to make array of drawing numbers - helper, combined with makeData
def drawingArray(fileLocation, numCurves):
    draw_nums = [os.path.basename(fileLocation)] * numCurves
    return draw_nums

# function to convert non-int inputs to ints
# Inputs: data as 2D list
# Outputs: a list with elements: features as pandas dataframe with int-only values, target as np array, all the label encoders

def makeData(fileLocation, data):
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

    numCurves = len(crv_deg)

    draw_nums = drawingArray(fileLocation, numCurves)

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

    return [features, target, labels_le, draw_nums] # features = X, target = y, label encoder, and array of drawing numbers

# train_test_split wrapper
def splitDataset(X, y, test_size):
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
    return [X_train, X_test, y_train, y_test]

# split which ensures all curves of a drawing are in only one dataset
# draw_nums is an array of the drawing number each curve belongs to
def splitDatasetByDrawing(X, y, test_size, draw_nums):
    gs = GroupShuffleSplit(n_splits = 2, test_size = test_size)
    train_ix, test_ix = next(gs.split(X, y, groups=draw_nums))
    X_train = X.loc[train_ix]
    X_test = X.loc[test_ix]
    y_train = y.loc[train_ix]
    y_test = y.loc[test_ix]
    return [X_train, X_test, y_train, y_test]

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

def makeAllData():
    all_files = os.listdir("pkl_data")

    data_lst = []

    # initalize values
    labels_str = []
    crv_type_str = []
    crv_closed = []
    crv_deg = []
    crv_def = []
    crv_per = []
    crv_span = []
    crv_ctrl = []
    crv_dist = []
    crv_norm = []
    crv_zbuff = []
    crv_rid_str = [] 
    crv_ind = []
    draw_nums = []

    for i in range(len(all_files)):
        fileLocation = all_files[i]
        data_item = pklToLst(fileLocation)
        data_lst.append(data_item) # [data for drawing1, data for drawing2, etc.]

        draw_nums += [os.path.basename(fileLocation)] * len(data_item[0]) # the number of curves in any given drawing

        labels_str = data_item[0] # a list of strs
        crv_type_str = data_item[1] # a list of strs
        crv_closed = data_item[2] # ints 0 or 1 representing bools
        crv_deg += data_item[3] # int
        crv_def += data_item[4] # int
        crv_per += data_item[5] # int
        crv_span += data_item[6] # int
        crv_ctrl += data_item[7] # int
        crv_dist += data_item[8] # int
        crv_norm += data_item[9] # color as int
        crv_zbuff += data_item[10] # value as int
        crv_rid_str += data_item[11] # id in rhino as str
        crv_ind += data_item[12] # index as int

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

    return [features, target, labels_le, draw_nums] # features = X, target = y, label encoder, and array of drawing numbers]
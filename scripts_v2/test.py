import pickle
import os
from PIL import Image
# function to open files and store as data
# Inputs: file location as str
# Outputs: data as a 2D list

def pklToLst(fileLocation):
    with open(fileLocation, 'rb') as filein:
        data = pickle.load(filein)
    return data

data = pklToLst("C:\\Users\\andre\\Documents\\lineweights\\scripts_v2\\demo_file.pkl")

x1_lst = data[0]
y1_lst = data[1]
x2_lst = data[2]
y2_lst = data[3]
rid_lst = data[4]
lw_lst = data[5]


normal_img = Image.open("C:\\Users\\andre\\Documents\\lineweights\\scripts_v2\\normal.jpg")
print(normal_img.width, normal_img.height)
width = normal_img.width
height = normal_img.height

print(x1_lst[0], y1_lst[0], x2_lst[0], y2_lst[0])
cropped = normal_img.crop((x1_lst[0], height - y1_lst[0], x2_lst[0], height - y2_lst[0]))
cropped.show()
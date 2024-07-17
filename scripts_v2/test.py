import pickle
import os
from PIL import Image, ImageOps
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
"""normal_img = Image.open("C:\\Users\\andre\\Documents\\lineweights\\scripts_v2\\normal.jpg")
print(normal_img.width, normal_img.height)
width = normal_img.width
height = normal_img.height

print(x1_lst[0], y1_lst[0], x2_lst[0], y2_lst[0])
cropped = normal_img.crop((x1_lst[0], height - y1_lst[0], x2_lst[0], height - y2_lst[0]))
cropped = ImageOps.pad(cropped, (300, 300), color="black", centering=(0.5, 0.5))
cropped.show()"""

# Iterate through all the bounding boxes

# Crop them

def cropImg(img_path, x1, y1, x2, y2):
    img = Image.open(img_path)
    width = img.width
    height = img.height
    return img.crop((x1, height - y1, x2, height - y2))

# Add the black padding
# Inputs: img is an image, such as Image.open("C:\\Users\\andre\\Documents\\lineweights\\scripts_v2\\normal.jpg")
# width height is pixel dimension of the resized image
def padImg(img, width, height):
    return ImageOps.pad(img, (width, height), color="black", centering=(0.5, 0.5))

# Save image
def saveImg(img, name):
    img.save(name)

# Iterate through all the bounding boxes
img_path = "C:\\Users\\andre\\Documents\\lineweights\\scripts_v2\\normal.jpg"
for i in range(len(rid_lst)):
    img = cropImg(img_path, x1_lst[i], y1_lst[i], x2_lst[i], y2_lst[i])
    img = padImg(img, img.width, img.height) # CHANGE THIS TO DESIRED
    name = os.path.basename(img_path[:-4])
    # C:\\Users\\andre\\Documents\\lineweights\\scripts_v2\\normal_CRVDINDEX_RID
    name = name + "_" + str(i) + "_" + rid_lst[i] + ".jpg"
    saveImg(img, name)
    print("Saved: ", name)
    cropped_imgs = []
    cropped_imgs.append(name)
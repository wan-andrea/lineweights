import pickle
import os
from PIL import Image, ImageOps
# function to open files and store as data
# Inputs: file location as str
# Outputs: data as a 2D list

def pklToLst(fileLocation):
    # print(fileLocation)
    with open(fileLocation, 'rb') as filein:
        data = pickle.load(filein)
    return data


"""data = pklToLst("C:\\Users\\andre\\Documents\\lineweights\\scripts_v2\\demo_file.pkl")

x1_lst = data[0]
y1_lst = data[1]
x2_lst = data[2]
y2_lst = data[3]
rid_lst = data[4]
lw_lst = data[5]
print(lw_lst)"""

# Iterate through all the bounding boxes

# Crop them

def cropImg(img_path, x1, y1, x2, y2):
    img = Image.open(img_path)
    width = img.width
    height = img.height
    new = img.crop((x1, height - y1, x2, height - y2))
    if new.width == 0 or new.height == 0:
        if new.width == 0:
            if x1 == x2 & x1 >= 0:
                x1 = x1 - 20
            elif x1 == x2 & x2 <= 2250:
                x2 = x2 + 20
        if new.height == 0:
            if y1 == y2 & y1 <= 3330:
                y1 = y1 + 20
            elif y1 == y2 & y2 >= 0:
                y2 = y2 - 20
        new = img.crop((x1, height - y1, x2, height - y2))
    # print(f"The new image size is: {new.width} x {new.height}")
    return new

# Add the black padding
# Inputs: img is an image, such as Image.open("C:\\Users\\andre\\Documents\\lineweights\\scripts_v2\\normal.jpg")
# width height is pixel dimension of the resized image
def padImg(img, width, height):
    # print(f"Padding image of size {img.width} x {img.height} to {width} x {height}")
    return ImageOps.pad(img, (width, height), color="black", centering=(0.5, 0.5))

# Save image
def saveImg(img, name):
    img.save(name)

# make folders
def makeLWFolders(parent_dir):
    profile = os.path.join(parent_dir, "Profile")
    contour = os.path.join(parent_dir, "Contour")
    detail = os.path.join(parent_dir, "Detail")
    cut = os.path.join(parent_dir, "Cut")
    if not os.path.exists(profile):
        os.makedirs(profile)
    if not os.path.exists(contour):
        os.makedirs(contour)
    if not os.path.exists(detail):
        os.makedirs(detail)
    if not os.path.exists(cut):
        os.makedirs(cut)
# Iterate through all the bounding boxes

"""# Utility to check if images are aligned properly. If so, the image should never be just white and black
# utility function
def is_color_image(img):
    colors = img.getcolors()
    if colors == None or len(colors) > 2:
        return True
    else:
        return False"""

# Inputs: image path of the normal map, new_width (usually 2250), new_height (usually 3330)
# Outputs: a dictionary with file path as key, and label lineweight as value
def makeCroppedImgs(img_path, new_width, new_height, data):
    cropped_imgs = {}
    x1_lst = data[0]
    y1_lst = data[1]
    x2_lst = data[2]
    y2_lst = data[3]
    rid_lst = data[4]
    lw_lst = data[5]
    for i in range(len(rid_lst)):
        # crop the image 
        img = cropImg(img_path, x1_lst[i], y1_lst[i], x2_lst[i], y2_lst[i])
        # UTILITY check image
        # pad the borders with black
        img = padImg(img, new_width, new_height)
        """if not is_color_image(img):
            raise Exception("The alignment on is wrong.")"""
        # this is the parent folder
        save_path = img_path.rsplit('\\', 1)[0]
        # make folders if they don't exist
        makeLWFolders(save_path)
        name = os.path.basename(img_path[:-4])
        # name = "normal", str(i) = curve #, rid_lst[i] = rhino id of the curve
        name = save_path + "\\" + lw_lst[i] + "\\" + name + "_" + "C" + str(i) + "_" + rid_lst[i] + ".jpg"
        # print(name)
        
        saveImg(img, name)
        cropped_imgs[name] = lw_lst[i]
    return cropped_imgs

test_data = pklToLst("C:\\Users\\andre\\Documents\\lineweights\\processed_data\\7.pkl")
fin = makeCroppedImgs("C:\\Users\\andre\\Documents\\lineweights\\processed_data\\normal_7.jpg", 2250, 3300, test_data)
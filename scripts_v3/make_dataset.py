from crop_imgs import *
import glob

parent_folder = "C:\\Users\\andre\\Documents\\lineweights\\uncropped_data"
# a list containing dictionaries for each drawing
labels_imgs = []

# Make sure the pkl files are read in numerical order 
filenames = sorted(glob.glob(os.path.join(parent_folder, '*.pkl')), key=lambda x: int(os.path.basename(x).split(".")[0]))

for filename in filenames:
    print(f"Processing file: {filename}")
    draw_num = os.path.basename(filename)[:-4]
    data = pklToLst(filename)
    normal_img = parent_folder + "\\" + "normal" + "_" + str(draw_num) + ".jpg"
    # if the image is just white or black, raise an error
    labels_imgs.append(makeCroppedImgs(normal_img, 2250, 3300, data))


"""
base_path = "C:\\Users\\andre\\Documents\\lineweights\\data" 

for folder in glob.glob(os.path.join(base_path, "*")):
    folder_name = os.path.basename(folder)
    num = int(folder_name)  # remove leading zeroes
    image_path = os.path.join(folder, "normal.jpg")
    new_image_path = os.path.join(folder, f"normal_{num}.jpg")
    os.rename(image_path, new_image_path)
    print(f"Renamed {image_path} to {new_image_path}")
"""
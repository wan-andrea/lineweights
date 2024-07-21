import os
import glob
import re
import shutil

src_folder = "C:\\Users\\andre\\Documents\\lineweights\\processed_data"
dst_folder = "C:\\Users\\andre\\Documents\\lineweights\\data"


# Get list of files in source folder
all_files = os.listdir(src_folder)
jpg_files = [file for file in all_files if file.endswith('.jpg')]
parent_dir = "C:\\Users\\andre\\Documents\\lineweights\\data"
# print(jpg_files)
# print(len(jpg_files))
paths = {}
for root, dirs, files in os.walk(parent_dir):
    for dir in dirs:
        number = os.path.basename(dir)
        number = int(number)
        path = os.path.join(root, dir)
        paths[number] = path

src_parent = "C:\\Users\\andre\\Documents\\lineweights\\processed_data\\"
for jpg in jpg_files:
    name = os.path.splitext(jpg)[0]
    parts = name.split("_")
    number = parts[-1]
    number = int(number)
    src_path = src_parent + jpg
    dst_path = paths[number] + "\\normal.jpg" # path we want to copy the normal image to
    shutil.copy(src_path, dst_path)

print("Done!")
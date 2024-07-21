from PIL import Image, ImageOps
import os

def is_color_image(image_path):
    img = Image.open(image_path)
    colors = img.getcolors()
    if colors == None or len(colors) > 2:
        print("Didn't find it.")
        return True
    else:
        print("Found one!")
        return False

def find_black_and_white_images(directory):
    black_and_white_images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            if not is_color_image(image_path):
                black_and_white_images.append(image_path)
    return black_and_white_images

a = find_black_and_white_images("C:\\Users\\andre\\Documents\\lineweights\\processed_data\\Profile\\")
b = find_black_and_white_images("C:\\Users\\andre\\Documents\\lineweights\\processed_data\\Contour\\")
c = find_black_and_white_images("C:\\Users\\andre\\Documents\\lineweights\\processed_data\\Detail\\")
d = find_black_and_white_images("C:\\Users\\andre\\Documents\\lineweights\\processed_data\\Cut\\")

print(a)
print(b)
print(c)
print(d)
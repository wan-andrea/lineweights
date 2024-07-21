test = "C:\\Users\\andre\\Documents\\lineweights\\processed_data\\Cut\\normal_0_C13_af2b6eaf-d307-4516-a2fd-1dd39299b365.jpg"

import os

# Utility function
# Given an image path of an image tied to a curvve
# Return the Rhino ID
def getRhinoId(path):
   rid = os.path.splitext(os.path.basename(path))[0]
   rid = rid.split("_")[-1]
   return rid
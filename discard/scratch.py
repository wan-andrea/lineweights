import pickle
def pklToLst(fileLocation):
    with open(fileLocation, 'rb') as filein:
        data = pickle.load(filein)
    return data

tester = pklToLst("crv_data\\crv_1.pkl")

"""width = 128
height = 128

class Dummy:
    def __init__(self, batch_size, max_step):
        self.batch_size = batch_size # probably 1
        self.max_step = max_step
        self.observation_space = (self.batch_size, width, height, 3)
        # in their example, it was a 13-D tuple
        # all curves must be open
        # The Grasshopper script will take Q curves,
        # (1) open all closed curves
        # (2) rebuild them with a fixed number of N control points and M degree, resulting in P knot vectors
        # Then store the following:
        # (1) Each control point is a 3D-tuple (x, y, z), resulting in a Q length list of N length list of 3D-tuples
        # (2) A Q length list of N-length list of integers representing the weight of each control point
        # (3) A P length list of N-length list of floats representing the knot vector


        # So If I fix the curve control points at 10, and degree at 3
        # control points: [[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)....(x10, y10, z10)]] # means there will be 30 elements per curve
        # weights: [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)...] # means there will be 10 elements per curve
        # knot vector [(1, 2, 3, 4, 5, 6, 7, 8, 9 10, 11, 12)] # means there will be 12 elements per curve
        
        self.action_space = ()

"""
commands_processing.py

Author:   Amber, Anbo Wu
Date:     March 2025
Project:  Topological Data Analysis in an Economic Context: Property Tax Maps
"""
import os
PATH = "C:/Projects/TDA/_0Codes"
# PATH = "/path/to/your/directory"
os.chdir(PATH) 

import numpy as np

filename = "0101"   # raw image name, string of four digits
dimension = 0       # dimension = {0, 1}, 0 = H_0, 1 = H_0^*
path = ""           # desired file output location, 
                    # empty then output to current folder
                    
cityindex = int(filename[0:2])
sampleindex = int(filename[2:4])

# %% Cell[1] Grayscale image

"""
1. Produce grayscale version
"""
from python_functions import grayscale
grayscale(filename)

# %% Cell[2] Erosion method

"""
2. Perform erosion method
"""
from python_functions import erosion
erosion(filename)

# %% Cell[3] Persistence diagram

"""
3. Plot persistence diagram
"""
from python_functions import pers
pers(filename, dimension)

# %% Cell[4] Persistence image

"""
4. Plot persistence image
"""
from python_functions import pers
from python_functions import plotPi
dgm = pers(filename, dimension, gendgm = True)
pi = plotPi(dgm, cityindex, sampleindex, path, dimension)

# %% Cell[5] Persistence image (all plots)

"""
5. Plot persistence image of all samples without outliers
"""
from python_functions import calcExtreme
name_pi = np.load("nameh" + str(dimension) + "_fin.npy", allow_pickle = True)
# replace path_pi to edit the location that images are saved to
path_pi = PATH + "/pih%s/" % str(dimension)
Pi = np.load("pih" + str(dimension) + "_fin.npy", allow_pickle = True)
_Min, _Max = calcExtreme(Pi)
for filename in name_pi:
    print(filename)
    index_pi = np.where(name_pi == filename)
    cityindex_pi = int(name_pi[index_pi][0][0:2])
    sampleindex_pi = int(name_pi[index_pi][0][2:4])
    pi = Pi[index_pi][0].reshape((100,100))
    plotPi(pi, cityindex, sampleindex, path_pi, dimension, gen = False)
    plotPi(pi, cityindex, sampleindex, path_pi, dimension, method = 'scaled', 
           _min = _Min, _max = _Max, gen = False)

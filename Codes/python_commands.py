"""
python_commands.py

Author:   Amber, Anbo Wu
Date:     July 2024
Project:  Topological Data Analysis in an Economic Context: Property Tax Maps
"""

import numpy as np

from python_functions import erosion
from python_functions import pers
from python_functions import plotPi
from python_functions import calcExtreme

filename = "0102"   # raw image name
dimension = 0       # dimension = {0, 1}, 0 = H_0, 1 = H_0^*
path = ""           # desired file output location, 
                    # empty then output to current folder
                    
name = np.load("nameh" + str(dimension) + ".npy", allow_pickle = True)
index = np.where(name == filename)
cityindex = int(name[index][0][0:2])
sampleindex = int(name[index][0][2:4])

# %%

"""
1. Perform erosion method
"""

erosion(filename)

# %%

"""
2. Plot persistence diagram
"""

pers(filename, dimension)

# %%

"""
3. Plot persistence image
"""

dgm = pers(filename, dimension, gendgm = True)
pi = plotPi(dgm, cityindex, sampleindex, path, dimension)

# %%

"""
4. Plot persistence image - scaled colormap according to global extremities
"""

Pi = np.load("repih" + str(dimension) + ".npy", allow_pickle = True)
_Min, _Max = calcExtreme(Pi)
pi = Pi[index][0].reshape((100,100))
plotPi(pi, cityindex, sampleindex, path, dimension, method = 'scaled', 
       _min = _Min, _max = _Max, gen = False)
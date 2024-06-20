"""
python_functions.py

Author:   Amber, Anbo Wu
Date:     June 2024
Project:  Topological Data Analysis in an Economic Context: Property Tax Maps
"""

import PIL
import numpy as np
import matplotlib.pyplot as plt
import copy
from ripser import lower_star_img
from persim import PersImage

"""
  pers(filename, dimension)
    produce numpy containing persistent points that
    construct the persistent diagram.
  dimension = {0, 1}, specify method of filtration.

"""
def pers(filename, dimension = 0):
  img = PIL.Image.open(filename)
  arr0 = np.array(img)
  arr1 = np.delete(arr0, 1, 2)
  arr = np.delete(arr1, 1, 2)
  x = np.zeros((arr.shape[0], arr.shape[1]))
  for t in range(arr.shape[0]):
    for s in range(arr.shape[1]):
      x[t, s] = check(arr[t, s])
  x = merge(x)
  if dimension == 0:
    x = nodata0(x)
    x = remove0(lower_star_img(x))
  elif dimension == 1:
    x = nodata1(x)
    x = remove1(lower_star_img(-x))
  return np.array(x)

"""
  plotPers(dgm, x, y, path, dimension)
    plot according persistent diagram and save to path.
  dimension = {0, 1}, specify method of filtration.
"""
def plotPers(dgm, x, y, path, dimension = 0):
  fig = plt.figure(figsize=(6, 6))
  if dimension == 0:
    plt.title("PD for $H_0$, Sample " + str('%02d' % x) + str('%02d' % y))
  elif dimension == 1:
    plt.title("PD for $H_0$, Sample " + str('%02d' % x) + str('%02d' % y))
  plt.xlabel('Birth')
  plt.ylabel('Lifetime')
  if dimension == 0:
    plt.xlim([0, 12])
  elif dimension == 1:
    plt.xlim([-12, 1])
  plt.ylim([0, 13])
  for i in range(len(dgm)):
    plt.plot(dgm[i][0], dgm[i][1]-dgm[i][0], 'o', color='black')
  check, count = read(dgm)
  for i in range(len(check)):
    plt.scatter(check[i][0], check[i][1]-check[i][0],
                s=count[i], facecolors='none', edgecolors='r')
  if dimension == 0:
    xvalue = 11
  elif dimension == 1:
    xvalue = 0
  plt.scatter(xvalue, 12, s=200, facecolors='none', edgecolors='r')
  plt.annotate('1', (xvalue+0.2, 12.2))
  plt.scatter(xvalue, 11, s=1000, facecolors='none', edgecolors='r')
  plt.annotate('5', (xvalue+0.2, 11.2))
  plt.scatter(xvalue, 10, s=2000, facecolors='none', edgecolors='r')
  plt.annotate('10', (xvalue+0.2, 10.2))
  plt.scatter(xvalue, 9, s=4000, facecolors='none', edgecolors='r')
  plt.annotate('20', (xvalue+0.2, 9.2))
  plt.show()
  fig.savefig(path +
              str('%02d' % x) + str('%02d' % y) + "pd.png")

# following are help functions for the former two.

def check(x):
  c = [235,
      215,
      195,
      175,
      155,
      135,
      115,
      95,
      75,
      55,
      35,
      25,
      0]
  t = [10.5,
      9.5,
      8.5,
      7.5,
      6.5,
      5.5,
      4.5,
      3.5,
      2.5,
      1.5,
      0.5,
      -0.5,
      -1.5]
  for i in range(len(c)):
    if c[i] == x:
      return t[i]

def calc(result, y, x):
  check = []
  val = []
  if result[0][y] > 0:
    check.append(x[result[0][y]-1, result[1][y]])
    if result[1][y] > 0:
      check.append(x[result[0][y]-1, result[1][y]-1])
    if result[1][y] < 999:
      check.append(x[result[0][y]-1, result[1][y]+1])
  if result[0][y] < 999:
    check.append(x[result[0][y]+1, result[1][y]])
    if result[1][y] > 0:
      check.append(x[result[0][y]+1, result[1][y]-1])
    if result[1][y] < 999:
      check.append(x[result[0][y]+1, result[1][y]+1])
  if result[1][y] > 0:
    check.append(x[result[0][y], result[1][y]-1])
  if result[1][y] < 999:
    check.append(x[result[0][y], result[1][y]+1])
  for z in range(len(check)):
    if check[z] > -1.5:
      val.append(check[z])
  if val:
    return min(val)
  else:
    return -1.5

def change(x0):
  new = np.copy(x0)
  result = np.where(x0 == -1.5)
  for y0 in range(len(result[0])):
    new[result[0][y0], result[1][y0]] = calc(result, y0, x0)
  return new

def merge(z):
  fin = z
  if -1.5 in fin:
    fin = change(fin)
    fin = merge(fin)
    return fin
  else:
    return fin

def nodata0(x0):
  new = np.copy(x0)
  result = np.where(x0 == -0.5)
  for y0 in range(len(result[0])):
    new[result[0][y0], result[1][y0]] = 15
  return new

def nodata1(x0):
  new = np.copy(x0)
  result = np.where(x0 == -0.5)
  for y0 in range(len(result[0])):
    new[result[0][y0], result[1][y0]] = 0.5
  return new

def remove0(dgm):
    for x in range (len(dgm)):
        if dgm[x][1] == float('inf') or dgm[x][1] == 15: dgm[x][1] = 12
    return dgm

def remove1(dgm):
    for x in range (len(dgm)):
        if dgm[x][1] == float('inf') or dgm[x][1] == 0.5: dgm[x][1] = 1.5
    return dgm

def read(dgm):
  l = dgm.tolist()
  check = []
  count = []
  for i in range(len(l)):
    if not check:
      check.append(l[i])
      count.append(l.count(l[i])*200)
    if not l[i] in check:
      check.append(l[i])
      count.append(l.count(l[i])*200)
  return check, count

"""
  plotPi(dgm, x, y, path, dimension, method, _min, _max)
    plot persistent image and return the according array (100x100)
    precondition: if want to call calcExtreme() before to know
  dimension = {0, 1}, specify method of filtration.
  method = {'unscaled','scaled'}, specify method of plotting
"""
def plotPi(dgm0, x, y, path, dimension = 0, method = 'unscaled', _min = 0, _max = 0):
  dgm = copy.deepcopy(dgm0)
  if dimension == 1:
    # For dimension 1, plot function only behave normally when shift all points to
    # positive axis.
    for n in range (len(dgm)):
      for k in range (len(dgm[n])):
          dgm[n][k] = 11 + dgm[n][k]
  pim = PersImage()
  PersImage.__init__(pim, spread=0.5, specs = {"maxBD": 13,
              "minBD": 0},pixels=[100,100], verbose=False)
  imgs = pim.transform(dgm)
  ax = plt.gca()
  fig, ax = plt.subplots(figsize=(6, 6))
  plt.xticks([])
  plt.yticks([])
  if dimension == 0:
    plt.title("PI for $H_0$\nwith 100x100 pixels, sample" + str('%02d' % x) + str('%02d' % y))
  elif dimension == 1:
    plt.title("PI for $H_1$\nwith 100x100 pixels, sample" + str('%02d' % x) + str('%02d' % y))
  if method == 'unscaled':
    ax.imshow(imgs, cmap=plt.get_cmap("plasma"))
    plt.savefig(path + str('%02d' % x) + str('%02d' % y) + "upi.png")
  elif method == 'scaled':
    ax.imshow(imgs, cmap=plt.get_cmap("CMRmap"), vmin = _min, vmax = _max) #rainbow
    plt.savefig(path + str('%02d' % x) + str('%02d' % y) + "pi.png")
  imgs_array = np.array([img.flatten() for img in imgs])
  return imgs_array

"""
  pi(dgm, x, y, dimension)
    only return the according array
  dimension = {0, 1}, specify method of filtration.
"""
def pi(dgm, x, y, dimension = 0):
  if dimension == 1:
    for n in range (len(dgm)):
      for k in range (len(dgm[n])):
          dgm[n][k] = 11 + dgm[n][k]
  pim = PersImage()
  PersImage.__init__(pim, spread=0.5, specs = {"maxBD": 13,
              "minBD": 0},pixels=[100,100], verbose=False)
  imgs = pim.transform(dgm)
  imgs_array = np.array([img.flatten() for img in imgs])
  return imgs_array

"""
  _min, _max = calcExtreme(dgm):
    generate persistent image array for desired pictures,
    concatenate into one array and pass to this function to calculate
    the extrmities for plotting scaled persistent images
"""
def calcExtreme(dgm):
  _max = 0
  _min = 0
  dgmlist = []
  for i in range (len(dgm)):
      pim = PersImage()
      PersImage.__init__(pim, spread=0.5, specs = {"maxBD": 13,
                  "minBD": 0}, pixels=[100,100], verbose=False)
      imgs = pim.transform(dgm[i])
      imgs_array = np.array([img.flatten() for img in imgs])
      dgmlist.append(imgs_array)
      
      if np.amax(imgs) > _max:
          _max = np.amax(imgs)
      if np.amin(imgs) < _min:
          _min = np.amin(imgs)
  dgmlist = np.array(dgmlist)
  return _min, _max
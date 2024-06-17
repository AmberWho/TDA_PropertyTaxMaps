"""
persistDiagram.py

Author:   Amber, Anbo Wu
Date:   March 2021
Project:  Topological Data Analaysis of Property Tax Maps

Description:
  Offers functions to read grayscale images, omit streets
  and no-data properties; plot corresponding persistent
  diagrams with circles indicating number of persistent
  points located at the same position.

Functions:
  pers(filename, dimension)
    produce numpy containing persistent points that
    construct the persistent diagram.
  dimension = {0, 1}, specify method of filtration.
  
  plotPers(dgm, x, y, path, dimension)
    plot according persistent diagram and save to path.
  dimension = {0, 1}, specify method of filtration.
"""

import numpy as np
import PIL
import matplotlib.pyplot as plt
from persim import plot_diagrams
from persim import bottleneck
from ripser import lower_star_img
from persim import PersImage
from sklearn.cluster import KMeans
import os
import shutil

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

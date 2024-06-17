"""
analysis.py

Author:   Amber, Anbo Wu
Date:   April 2021
Project:  Topological Data Analaysis of Property Tax Maps

Description:
  Offers various functions for data evaluation and refinement.
  Please check function comments for detailed information. 
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import LocallyLinearEmbedding

def k_Means(data, n = 5):
  # Produce k-means clustering result for n clusters.
  train_dataset = np.array(data)
  nsamples, nx, ny = train_dataset.shape
  d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))
  km = KMeans(n_clusters=n, random_state=0).fit(d2_train_dataset)
  return km.labels_

def pcA(data, n = 3, fitdata = False, fit = [], lle = False):
  # Construct three-component PCA, can fit given vectors into
  # the model.
  # lle: do Locally Linear Embedding after PCA.
  train_dataset = np.array(data)
  pc = PCA(n_components=n)
  nsamples, nx, ny = train_dataset.shape
  d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))
  pca = pc.fit_transform(d2_train_dataset)
  print('explained variance ratio is: '+str(pc.explained_variance_ratio_))
  if lle:
    embedding = LocallyLinearEmbedding(n_neighbors=21,n_components=2)
    pca = embedding.fit_transform(pca)
  if fitdata:
    nsamples, nx, ny = fit.shape
    d2_fit = fit.reshape((nsamples,nx*ny))
    fitted= pc.transform(d2_fit)
    if lle:
      fitted = embedding.transform(fitted)
    return pca, fitted
  return pca


def cMean(dgm, tag, cNum):
  # Calculate mean for each cluster, return an array storing
  # each mean in order.
  clustermean = []
  for k in range (cNum):
    s = np.zeros((100,100)).tolist()
    n = 0
    for x in range (len(tag)):
      if tag[x] == k:
        n += 1
        for i in range (100):
          for j in range (100):
            s[i][j] += dgm[x][i][j]
    s = np.array(s)
    clustermean.append(s/n)
  return(np.array(clustermean))

def cClose(dgm, tag, mean, cNum, name, _dim= 0):
  # Calculate three closest samples for each cluster,
  # return according results for plot uses
  nsamples, nx, ny = mean.shape
  meand2 = mean.reshape((nsamples,nx*ny))
  nsamples, nx, ny = np.array(dgm).shape
  dgmd2 = np.array(dgm).reshape((nsamples,nx*ny))
  close = []
  close2 = []
  close3 = []
  for k in range (cNum):
    dist = []
    serial = []
    for i in range (len(tag)):
      if tag[i] == k:
        dist.append(np.linalg.norm(dgmd2[i]-meand2[k]))
        serial.append(i)
    print('k = '+str(k))
    cindex = dist.index(min(dist))
    closest = serial[cindex]
    close.append(closest)
    shortest = dist[cindex]
    dist.pop(cindex)
    print('The shortest distance is '+str(shortest))
    serial.pop(cindex)
    cindex2 = dist.index(min(dist))
    closest2 = serial[cindex2]
    close2.append(closest2)
    shortest2 = dist[cindex2]
    dist.pop(cindex2)
    print('The second shortest distance is '+str(shortest2))
    serial.pop(cindex2)
    cindex3 = dist.index(min(dist))
    closest3 = serial[cindex3]
    close3.append(closest3)
    shortest3 = dist[cindex3]
    print('The third shortest distance is '+str(shortest3))
    print("1:" + str(closest) + " " + name[closest])
    print("2:" + str(closest2)+ " " + name[closest2])
    print("3:" + str(closest3)+ " " + name[closest3])
  return close, close2, close3

def cDeviation(dgm, tag, mean, cNum):
  # Calculate cluster standard deviation for given data
  stdcluster = []
  for k in range (cNum):
    s = 0
    for i in range (100):
      for j in range (100):
        s += mean[k][i][j]
    s = s/10000
    stdcluster.append(s)
  sample = np.zeros((len(tag),1)).tolist()
  c = np.zeros((cNum,1)).tolist()
  for k in range (cNum):
    n = 0
    for x in range (len(tag)):
      if tag[x] == k:
        s = 0
        n += 1
        for i in range (100):
          for j in range (100):
                  s += dgm[x][i][j]
        s = s/10000
        sample[x] = s
        c[k] = c[k]+(s-stdcluster[k])**2
    c[k] = sqrt(c[k]/n)
  for k in range (cNum):
    n = 0
    for x in range (len(tag)):
      if tag[x] == k:
        sample[x] = (sample[x]-stdcluster[k])/c[k]
  return np.array(sample)

def constructName(dimension = 0):
  # Construct name list for 210 samples,
  # city serial + sample serial
  # dimension = {0, 1}, H0 anlysis omit two outliers
  name = []
  for x in range (1,22,1):
    for y in range (1,11,1):
        name.append(str('%02d' % x)+str('%02d' % y))
  if dimension == 0:
    name.pop(26)
    name.pop(167)
  return name

def plotData(pca, tag, title, axis = 3, additional = False,
            z_Data = 0, z_Title = 'Principal Component 3',
            plotclose = False, mean = [],
            close = [], close2 = [], close3 = [],
            cluster = 5):
  # Plot graphs of analysis. Note that this function is only
  # compatible with 3 or 5 clusters.
  # axis = {2, 3}
  #   number of axis of the plot
  # additional = {True, False}
  #   if plot pca against additional data.
  #   if true, please specify z_Data, z_Title for that.
  #   can also plot with 2 axis (that will then on y axis)
  #   if false, function would generate normal pca graph
  # plotclose = {True, False}
  #   if plot closest samples
  #   if ture, specify lists generate by cMean() and cClose()
  if len(pca[0]) == 3:
    principalDf = pd.DataFrame(data = pca
              , columns = ['principal component 1', 
              'principal component 2', 'principal component 3'])
  else:
    principalDf = pd.DataFrame(data = pca
              , columns = ['principal component 1', 
              'principal component 2'])
  principalDf['target'] = tag
  if additional:
    principalDf['add'] = z_Data
  finalDf = principalDf
  fig = plt.figure(figsize = (8,8))
  if axis == 2:
    ax = fig.add_subplot(111)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    if additional:
      ax.set_ylabel(z_Title, fontsize = 15)
    else: ax.set_ylabel('Principal Component 2', fontsize = 15)
  elif axis == 3:
    ax = Axes3D(fig)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel(z_Title, fontsize = 15)
  ax.set_title(title, fontsize = 20)
  targets = [0, 1, 2, 3, 4]
  if cluster == 3:
      legend = ['cluster 0-a', 'cluster 0-b', 'cluster 0-c']
      colors = ['orange','crimson','yellow']
  elif cluster == 5 and additional:
    legend = ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3', 
              'cluster 4', z_Title]
    colors = ['r', 'g', 'b', 'y', 'm']
  else:
    legend = ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3', 
              'cluster 4']
    colors = ['r', 'g', 'b', 'y', 'm']
  for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    if axis == 3:
      ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , finalDf.loc[indicesToKeep, z_Title]
                , c = color
                , s = 50)
    elif axis == 2:
      if additional:
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, z_Title]
                , c = color
                , s = 50)
      else:
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
  if plotclose:
    if axis == 2:
      for i in range (cluster):
          ax.scatter(mean[i][0],mean[i][1],c = 'c',s = 50)
          ax.scatter(pca[close[i]][0],pca[close[i]][1],
                    c = 'k',s = 50)
          ax.scatter(pca[close2[i]][0],pca[close2[i]][1],
                    c = 'dimgray',s = 50)
          ax.scatter(pca[close3[i]][0],pca[close3[i]][1],
                    c = 'lightgray',s = 50)
  ax.legend(legend)
  ax.grid()
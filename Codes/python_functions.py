"""
python_functions.py

Author:   Amber, Anbo Wu
Date:     March 2025
Project:  Topological Data Analysis in an Economic Context: Property Tax Maps
"""

import PIL
import numpy as np
import matplotlib.pyplot as plt
import copy
from PIL import Image
from ripser import lower_star_img
from persim import PersImage
import sys

# grayscale legend, sequenced from highest to lowest
c = [(255,255,255),
     (235,235,235),
     (215,215,215),
     (195,195,195),
     (175,175,175),
     (155,155,155),
     (135,135,135),
     (115,115,115),
     (95,95,95),
     (75,75,75),
     (55,55,55),
     (35,35,35),
     (25,25,25)]

"""
  grayscale(filename, newname, is_Landuse)
    open property tax maps and generate the grayscale version.
  filename = string contains four digit sample index
  newname = optional, new file name to replace the sample index
  is_Landuse = {True, False}, toggle for land use maps
"""
def grayscale(filename, newname = None, is_Landuse = False):

    file = filename
    if newname is None:
        file2 = filename
    else:
        file2 = newname
    print('Processing: [%s]' % file, end=" ... ")
    
    city = int(file[:2])
    # select legend type based on city index.
    if not is_Landuse:
        print('Map type: Property Tax Map')
        match city:
            case 4 | 5 | 8 | 9 | 11 | 12 | 13 | 14 | 17 | 20 | 21:
                _type = 'pink'
            case 1 | 2 | 3 | 6 | 7 | 10 | 15 | 16 | 18 | 19:
                _type = 'orange'
            case _:
                sys.exit('Invalid city index. Valid range: 1-21.')
    else:
        print('Map type: Land Use Map')
        # color legend according to land use maps
        _type = 'landuse'
    
    im = Image.open(file + '.png')
    
    def legend (num):
      # returns a color corresponding to the specified rank and legend type.
      # num = integer within range [0,11]
      #   0 means the lowest tax level, 11 means the highest
      # _type = {'pink','orange'}
      #   'pink' - red~pink~darkblue legend
      #   'orange' - red~orange~grayblue legend
      #   'landuse' - particular for land use maps, range [0,17]
      match _type:
        case 'pink':
            legends = [(242,242,242),
                        (0,38,83),
                        (0,81,186),
                        (0,142,214),
                        (96,175,221),
                        (191,209,229),
                        (249,191,193),
                        (252,140,153),
                        (252,94,114),
                        (206,17,38),
                        (172,31,44),
                        (124,33,40)]
        case 'orange':
            legends = [(242,242,242),
                        (49,77,112),
                        (108,130,150),
                        (142,162,181),
                        (181,206,212),
                        (190,210,187),
                        (239,239,190),
                        (223,209,178),
                        (250,184,132),
                        (237,117,82),
                        (163,61,61),
                        (120,42,42)]
        case 'landuse':
            legends = [(237,225,144),   # Single family
                        (228,205,125),  # 2-4 family
                        (228,193,123),  # Condominium/coop units
                        (223,212,184),  # Other residential
                        (227,168,135),  # Apartments/Multi-Family
                        (228,168,168),  # Commercial condos
                        (232,157,157),  # Office
                        (203,136,136),  # Retail
                        (253,228,223),  # Hotel/Motel/Other accommodation
                        (233,199,197),  # Restaurants
                        (160,140,168),  # Manufacturing/Storage facilities
                        (212,202,212),  # Other industrial
                        (201,217,232),  # Public services/Facilities
                        (180,196,211),  # Education
                        (150,161,181),  # Cemeteries and other religious
                        (108,128,149),  # Hospitals/Care facilities
                        (99,98,128),    # Other institutional
                        (150,176,176)]  # Entertaiment/Recreational
      return legends[num]
  
    def clrmatch(target, pixel, b):
      if (target[0]-b <= pixel[0] <= target[0]+b
        and target[1]-b <= pixel[1] <= target[1]+b
        and target[2]-b <= pixel[2] <= target[2]+b):
          return True
      else:
          return False

    def clrmatch(target, pixel, b):
      if (target[0]-b <= pixel[0] <= target[0]+b
        and target[1]-b <= pixel[1] <= target[1]+b
        and target[2]-b <= pixel[2] <= target[2]+b):
          return True
      else:
          return False

    def layer (im, num, c, newim):
      # read original graph (allow error in detected colors).
      # if not call fill() before return, returns the original graph.
      im_rgb = im.convert("RGB")
      if not is_Landuse:
          b = 7 # change this parameter for different error range
      elif city == 10:
          if c == (75,75,75):
              if file == '1005':
                  b = 10
              else:
                  b = 14
          else:
              b = 10
      else:
          b = 8
      newimg = Image.new('RGB', im.size, color = 'black')
      whitecolor = (255,255,255)
      for t in range (0,num+1,1):
        if t == 0 and not is_Landuse:
          b = 2 # error range for no data legend
        target = legend(t)
        if target == (227,168,135) and city == 10:
            target == (220,168,138) # readjust for California
        for x in range (0,im.size[0],1):
          for y in range (0,im.size[1],1):
            pixel = im_rgb.getpixel((x,y))
            if clrmatch(target,pixel,b):
                if is_Landuse and 0<x<im.size[0]-1 and 0<y<im.size[1]-1:
                    # land use map has edge detection problem
                    lpixel = im_rgb.getpixel((x-1,y))
                    rpixel = im_rgb.getpixel((x+1,y))
                    upixel = im_rgb.getpixel((x,y-1))
                    dpixel = im_rgb.getpixel((x,y+1))
                    lr = (not clrmatch(target,lpixel,b)) and (not clrmatch(target,rpixel,b))
                    ud = (not clrmatch(target,upixel,b)) and (not clrmatch(target,dpixel,b))
                    if not (lr or ud):
                        newimg.putpixel((x,y), whitecolor)
                else:
                    newimg.putpixel((x,y), whitecolor)
      newim = fill(newimg, newim, c)
      return newim
    
    def fill (img, new, c):
      # get a black-and-white picture and omit noises:
      # check a 6x6 area centered with this point, if the white
      # pixels exceed number of 28 (can be changed), then this
      # should be a white point in the result graph
      # if not call widen() before return, returns a cleaned thin graph.
      if not is_Landuse:
          sidelength = 6
          rewind = 2 # step back to the top-left corner of the 6x6 area
          num_count = 28
      else:
          sidelength = 4
          rewind = 2
          num_count = 10
      img_rgb = img.convert("RGB")
      newim = Image.new('RGB', im.size, color = 'black')
      blackcolor = (0,0,0)
      whitecolor = (255,255,255)
      for x in range (0,im.size[0],1):
        for y in range (0,im.size[1],1):
          pixel = img_rgb.getpixel((x,y))
          if pixel == whitecolor:
            count = 0
            a = x-rewind
            b = y-rewind
            for t in range(sidelength):
              for s in range(sidelength):
                if (0 <= a+t < im.size[0] and 0 <= b+s < im.size[1]):
                  check = img_rgb.getpixel((a+t,b+s))
                  if check == whitecolor:
                    count = count+1
            if count < num_count:
              newim.putpixel((x,y), blackcolor)
            else:
              newim.putpixel((x,y), whitecolor)
      new = widen(newim, new, c)
      return new
    
    def widen (img, newim, c):
      # widen the white area by certain range (can be modified)
      # so that walls between properties can be taken away.
      img_rgb = img.convert("RGB")
      whitecolor = (255,255,255)
      for x in range (0,im.size[0],1):
        for y in range (0,im.size[1],1):
          pixel = img_rgb.getpixel((x,y))
          if pixel == whitecolor:
            for t in range(-5,6,1):
              for s in range(-5,6,1):
                if (0 <= x+t < im.size[0] and 0 <= y+s < im.size[1]):
                  newim.putpixel((x+t,y+s), c)
      return newim
    
    # grayscale legend, sequenced from highest to lowest
    if not _type == 'landuse':
        c = [(x,x,x) for x in range(255,15,-20)] + [(25,25,25)]
        count = 11
    else:
        c = [(x,x,x) for x in range(255,70,-10)]
        count = 17
    
    newim = Image.new('RGB', im.size, color = 'black')
    for a in range(count,-1,-1):
        newim = layer(im, a, c[count+1-a], newim)
        if a == 0:
            print("|=  Finish! (,,•ω•,,)  -=-------")
        else:
            print("==" + str('%4d' % (((count+1-a)/(count+1))*100)) +"% ╰(●’◡’●)╮  ==---=-------")
    newim.save(file2 + "gray.png")

"""
  erosion(filename)
    read grayscale images and perform the erosion
    method.
"""
def erosion(filename):
  img = PIL.Image.open(filename + "gray.png")
  arr0 = np.array(img)
  arr1 = np.delete(arr0, 1, 2)
  arr = np.delete(arr1, 1, 2)
  X = np.zeros((arr.shape[0], arr.shape[1]))
  for t in range(arr.shape[0]):
    for s in range(arr.shape[1]):
      X[t, s] = checkval(arr[t, s])
  X = merge(X)
  newim = Image.new('RGB', img.size, color = 'black')
  for x in range (0,img.size[0],1):
    for y in range (0,img.size[1],1):
        newim.putpixel((x,y), c[int(11.5-X[y,x])])
  newim.save("mod" + filename + ".png")

"""
  pers(filename, dimension, gendgm)
    produce numpy array containing persistent points and plot
    persistence diagram
  dimension = {0, 1}, specify method of filtration.
  gendgm = if True, return array containing persistent points
           if False, plot persistence diagram

"""
def pers(filename, dimension = 0, gendgm = False):
  img = PIL.Image.open("mod"+filename + ".png")
  arr0 = np.array(img)
  arr1 = np.delete(arr0, 1, 2)
  arr = np.delete(arr1, 1, 2)
  x = np.zeros((arr.shape[0], arr.shape[1]))
  for t in range(arr.shape[0]):
    for s in range(arr.shape[1]):
      x[t,s] = checkval(arr[t, s])
  if dimension == 0:
    x = nodata0(x)
    x = remove0(lower_star_img(x))
  elif dimension == 1:
    x = nodata1(x)
    x = remove1(lower_star_img(-x))
  dgm = np.array(x)
  if gendgm:
      return dgm
  fig = plt.figure(figsize=(6, 6))
  if dimension == 0:
    plt.title("PD for $H_0$, Sample " + filename)
  elif dimension == 1:
    plt.title("PD for $H_1$, Sample " + filename)
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
  fig.savefig(filename + "pd.png")

# following are help functions for the former two.

def checkval(x):
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
    precondition: 'scaled' method is only valid when having the extremities
                  produced by calcExtreme()
  dimension = {0, 1}, specify method of filtration.
  method = {'unscaled','scaled'}, specify method of plotting
  _min, _max = only available when method is 'scaled'
"""
def plotPi(dgm0, x, y, path, dimension = 0, method = 'unscaled', _min = 0, _max = 0, gen = True):
  dgm = copy.deepcopy(dgm0)
  if gen:
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
  else:
      imgs = dgm
  ax = plt.gca()
  fig, ax = plt.subplots(figsize=(6, 6))
  plt.xticks([])
  plt.yticks([])
  if dimension == 0:
    plt.title("PI for $\mathrm{PH}_0$\nwith 100x100 pixels, sample " + str('%02d' % x) + str('%02d' % y))
  elif dimension == 1:
    plt.title("PI for $\mathrm{PH}_0^{*}$\nwith 100x100 pixels, sample " + str('%02d' % x) + str('%02d' % y))
  if method == 'unscaled':
    ax.imshow(imgs, cmap=plt.get_cmap("plasma"))
    plt.savefig(path + str('%02d' % x) + str('%02d' % y) + "upi.png")
  elif method == 'scaled':
    ax.imshow(imgs, cmap=plt.get_cmap("CMRmap"), vmin = _min, vmax = _max) #rainbow
    plt.savefig(path + str('%02d' % x) + str('%02d' % y) + "pi.png")
  if gen:
      return(imgs)

"""
  _min, _max = calcExtreme(dgm):
    calculate the extremities for plotting scaled persistent images
"""
def calcExtreme(dgm):
  _max = 0
  _min = 0
  for i in range (len(dgm)):
      imgs = dgm[i]
      if np.amax(imgs) > _max:
          _max = np.amax(imgs)
      if np.amin(imgs) < _min:
          _min = np.amin(imgs)
  return _min, _max

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import copy
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
import umap
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
from scipy import stats

"""
  ss = compute_avg_ss(df, data)
    return a list of average sample silouette scores for each cluster
"""
def compute_avg_ss(df, data):
    ss_list = []
    for i in range(len(df.columns)):
        ss_list.append(silhouette_score(data, df[str(i+1)]))
    return ss_list

"""
  outliers = find_outliers_zscore(data, threshold)
    return a list of outliers entries of a data based on the z-score
"""
def find_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    outliers = [x for i, x in enumerate(data) if z_scores[i] > threshold]
    return outliers

"""
  name = genName():
    return a complete list of 210 sample indices (in strings)
"""
def genName():
    name = []
    for x in range (21):
        for y in range (10):
            name.append(str('%02d' % (x+1)) + str('%02d' % (y+1)))
    return name

"""
  dmatrix = genDmatrix(pi, name):
    return a Pandas dataframe recording a distance matrix
  pi = 2d numpy array recording persistence image vectors
  name = name list to as row indices of the dataframe
"""
def genDmatrix(pi, name=[]):
    if not name: # generate default index to prevent error
        name = list(range(len(pi)))
    pi_list = pi.tolist()
    df = pd.DataFrame(pi_list, index=name)
    dmatrix = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
    return dmatrix

"""
  genPCA(data, n):
    produce a PCA projection of the given data, will output the variance preserved to the console
  data = 2d numpy array
  n = number of principal components
  return_variance = {True, False} if return pc object to retrieve explained variance ratio
  and components
"""
def genPCA(data, n = 3, return_pc = False):
  pc = PCA(n_components=n)
  pca = pc.fit_transform(data)
  if return_pc:
      return pca, pc
  print('explained variance ratio is: ' + str(pc.explained_variance_ratio_))
  return pca

"""
  genPCA(data, n):
    produce a PCA projection of the given data, will output the variance preserved to the console
  data = 2d numpy array
  n = number of principal components
  return_variance = {True, False} if return pc object to retrieve explained variance ratio
  and components
"""
def genUMAP(data, _neighbor, n = 3):
  umap_ = umap.UMAP(n_neighbors=_neighbor, n_components = n,
                    random_state=42)
  embedding = umap_.fit_transform(data)
  return embedding

"""
  dist_c, size_c = evalCluster(dist ,tag):
    return the size of each cluster, average sample distance per cluster and
    diameter of each cluster in Pandas dataframe
  dist = distance matrix
  tag = clustering result
"""
def evalCluster(dist, tag):
    dist_c = []
    size_c = []
    diam_c = []
    df = copy.deepcopy(dist)
    df['target'] = tag
    for i in range(max(tag)+1):
        indicesToKeep = df['target'] == i
        # distance dataframe of the given cluster
        d_filt = copy.deepcopy(df)
        d_filt.drop('target', axis=1, inplace=True)
        d_filt = d_filt[d_filt.columns[indicesToKeep]]
        d_filt = d_filt[indicesToKeep]
        # size of clusters
        size_c.append(indicesToKeep.sum())
        # average sum of distance
        s = df[:].where(df['target'] == i)
        s.drop('target', axis=1, inplace=True)
        # average distance to other samples in the same cluster for each sample
        avgdist = s.loc[:, indicesToKeep].sum()/(sum(indicesToKeep)-1)
        # average distance per cluster
        dist_c.append(avgdist.sum()/sum(indicesToKeep))
        # diameter of clusters
        diam_c.append(d_filt.to_numpy().max())
    df_output = pd.DataFrame({'size': size_c})
    df_output['distance'] = dist_c
    df_output['diameter'] = diam_c
    return df_output

"""
  df_alltag, df_allmed, (df_count) = printCluster(variables):
    print a summary of clustering result to a file f from multiple iterations
  pi = persistence images
  dist = distance matrix
  name = name list
  iteration = number of iterations
  method = {'heuristic', 'random', 'k-medoids++', 'k-means'}, clustering method
  f = the file for printing
  maxiteration = number of maximum step of algorithm allowed for each iteration;
                 suppressed to get initialization results
  prtmed = {'True', 'False'}, can be set to False to suppress result details
  ctmed = {'True', 'False'}, counting the number of occurrence of solutions,
          is available only if prtmed = True
"""
def printCluster(pi, dist, name, cluster, iteration, method, f, maxiteration = 300, prtmed = True, ctmed = False):
    print("\nCLUSTER: " + str(cluster), "|| ITERATION: " + str(iteration),
          "|| METHOD: " + method, "|| MAX ITERATION: " + str(maxiteration), file = f)

    for i in range(iteration):
        # Print to console to track progress on iteration
        if (i+1) % (iteration/10) == 0:
            if (i+1) != iteration:
                print(str("%02d" % (((i+1)/iteration)*100)) + "%-", end="")
            else:
                print(str("%02d" % (((i+1)/iteration)*100)) + "% ")
        
        match method:
            case "heuristic":
                km = KMedoids(n_clusters=cluster, method='pam', init='heuristic', max_iter=maxiteration).fit(pi)
            case "random":
                km = KMedoids(n_clusters=cluster, method='pam', init='random', max_iter=maxiteration, random_state=i).fit(pi)
            case "k-medoids++":
                km = KMedoids(n_clusters=cluster, method='pam', init='k-medoids++', max_iter=maxiteration).fit(pi)
            case "k-means":
                # suppress error message
                import warnings
                warnings.filterwarnings('ignore')
                km = KMeans(n_clusters=cluster, max_iter=300, n_init="auto").fit(pi)
        
        if not prtmed: # not just print medoid counts
            sil_km = pd.DataFrame({'tag':km.labels_})
            sil_km['ss'] = silhouette_samples(pi, km.labels_)
            df_km = evalCluster(dist, km.labels_)
            sil_list = []
            for j in range(cluster):
                # average silhoute score by cluster
                sil_list.append(sil_km.loc[sil_km['tag'] == j, 'ss'].sum() / df_km.loc[j, 'size'])
            df_km['ss'] = sil_list
            df_km = df_km.sort_values(by='size')
            index_list = df_km.index.to_list()
            cluster_labels = copy.deepcopy(km.labels_) # get the new cluster labeling after relabeling by size
            for j in range(len(km.labels_)):
                cluster_labels[j] = index_list.index(km.labels_[j])
            df_km = df_km.reset_index(drop=True)
            if i == 0:
                dfkm_size = pd.DataFrame({'1':df_km['size']})
                dfkm_dist = pd.DataFrame({'1':df_km['distance']})
                dfkm_diam = pd.DataFrame({'1':df_km['diameter']})
                dfkm_ss = pd.DataFrame({'1':df_km['ss']})
            else:
                dfkm_size[str(i+1)] = df_km['size']
                dfkm_dist[str(i+1)] = df_km['distance']
                dfkm_diam[str(i+1)] = df_km['diameter']
                dfkm_ss[str(i+1)] = df_km['ss']
        # count medoid occurrence
        if method != "k-means":
            med = []
            for c in range(cluster):
                med.append(name[np.where((pi == km.cluster_centers_[c]).all(axis=1))[0][0]])
            med.sort()
            if i == 0:
                df_med = pd.DataFrame({'1':med})
                # store clustering result without repetition
                df_alltag = pd.DataFrame({'1':cluster_labels})
                df_allmed = pd.DataFrame({'1':med})
                df_count = pd.DataFrame({'1':[1]})
            else:
                df_med[str(i+1)] = med
                # if cluster label is not stored in dataframe, add it as a new column
                is_identical = any(df_alltag[col].equals(pd.Series(cluster_labels)) 
                                   for col in df_alltag.columns)
                if not is_identical:
                    df_alltag[str(len(df_alltag.columns)+1)] = cluster_labels
                    df_allmed[str(len(df_allmed.columns)+1)] = med
                    df_count[str(len(df_count.columns)+1)] = 1
                else:
                    matching_columns = [col for col in df_alltag.columns if np.array_equal(df_alltag[col].values, cluster_labels)]
                    mat_column = matching_columns[0]
                    df_count[mat_column] += 1
    if method != "k-means":
        arr_med = df_med.to_numpy()
        unique, counts = np.unique(arr_med, return_counts=True)
        print("sample index and occurrence: " + str(dict(zip(unique, counts))), file = f)
        print("")
    if prtmed:
        df_output = pd.DataFrame({'Average Size': dfkm_size.mean(axis=1)})
        df_output['Maximum Size'] = dfkm_size.max(axis='columns')
        df_output['Minimum Size'] = dfkm_size.min(axis='columns')
        df_output['Average Distance'] = dfkm_dist.mean(axis=1)
        df_output['Maximum Distance'] = dfkm_dist.max(axis='columns')
        df_output['Minimum Distance'] = dfkm_dist.min(axis='columns')
        df_output['Average Diameter'] = dfkm_diam.mean(axis=1)
        df_output['Maximum Diameter'] = dfkm_diam.max(axis='columns')
        df_output['Minimum Diameter'] = dfkm_diam.min(axis='columns')
        df_output['Average SS'] = dfkm_ss.mean(axis=1)
        df_output['Maximum SS'] = dfkm_ss.max(axis='columns')
        df_output['Minimum SS'] = dfkm_ss.min(axis='columns')
        print(df_output.to_string(), file = f)
        print("")
        if method != "k-means":
            if ctmed:
                return df_alltag, df_allmed, df_count
            else:
                return df_alltag, df_allmed
  
"""
  plotSS(range_n_clusters, data, dist, tag)
    edited algorithm written by the scikit-learn developers, mainly serves to
    generate a report on the silhouette scores of a given clustering result
  range_n_clusters = number of clusters
  data = 3d array, the projection of persistence images for visualization
  dist = distance matrix
  tag = clustering result
"""
def plotSS(range_n_clusters, data, dist, tag = None):
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause
    # Edit: Amber, Anbo Wu
    
    X = data
    
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2,projection='3d')
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.3, 0.6])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Generate k-medoid clustering if no clustering is specified
        # reorder the clusters by size
        if tag is None:
            km = KMedoids(n_clusters=n_clusters, method='pam', init='heuristic').fit(X)
            tag = km.labels_
        df_km = evalCluster(dist, tag)
        df_km = df_km.sort_values(by='size')
        index_list = df_km.index.to_list()
        cluster_labels = copy.deepcopy(tag)
        for i in range(len(tag)):
            cluster_labels[i] = index_list.index(tag[i])
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "\nFor n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg, "\n"
        )
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_negative_silhouette_values = sample_silhouette_values[(cluster_labels == i) & 
                                                                      (sample_silhouette_values < 0)]
    
            ith_cluster_silhouette_values.sort()
            ith_negative_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
            
            if ith_negative_silhouette_values.size == 0:
                avg_negative_silhouette_values = 0
            else:
                avg_negative_silhouette_values = np.mean(ith_negative_silhouette_values)
            print("Cluster %d\nNumber of samples with negative SS: %d\nAverage negative SS: %0.4f\n"
                  % (i,ith_negative_silhouette_values.shape[0],avg_negative_silhouette_values))
    
        ax1.set_title("Silhouette Score Plot")
        ax1.set_xlabel("Silhouette Score")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    
        # 2nd Plot showing the actual clusters formed
        Proj = genPCA(X, n=3)
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            Proj[:, 0], Proj[:, 1], Proj[:,2], marker=".", s=50, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )
        
        
        tag = None # reset for looping over different number of clusters
    
    plt.show()
    
"""
  s1 = calc_relative_SS(data, tag, dist, cluster1, cluster2)
    return the relative silhouette score of cluster1 with respect to cluster2
  data = dataset
  tag = clustering result
  dist = distance matrix
  cluster1 = first cluster index
  cluster2 = second cluster index
"""
def calc_relative_SS(data, tag, dist, cluster1, cluster2):
    df = copy.deepcopy(dist)
    df = df.reset_index(drop=True)  # Reset row index
    df.columns = range(df.shape[1])  # Reset column names to numeric indices
    df['target'] = tag
    indices_1 = df['target'] == cluster1
    indices_2 = df['target'] == cluster2
    size_1 = sum(indices_1)
    size_2 = sum(indices_2)
    # average intra-cluster distance
    raw1 = df[:].where(df['target'] == cluster1).sum()
    raw1.drop(index='target', inplace=True)
    a1 = raw1[indices_1]/(size_1-1)
    raw2 = df[:].where(df['target'] == cluster2).sum()
    raw2.drop(index='target', inplace=True)
    # average other-cluster distance
    b1 = raw2[indices_1]/(size_2)
    # compute SS
    s1 = (b1-a1)/(a1.combine(b1, max))
    return s1.to_numpy()

"""
  plotCluster(variables):
    plot a 2d/3d projection of the sample points color-coded by the clustering results
  Refer to sample codes in commands_evaluation.py
"""
def plotCluster(data, tag, proj, dist, title, is_UMAP = False, plot_SS = True, plot_2d = False,
                extra_plot = [], name = [], extra_color = [], z_value = [], z_title = [],
                cluster_list = [], plot_quad = False, plot_city = False):
    plt.rcParams.update({'font.size': 14})
    if plot_SS:
        if cluster_list:
            fig = plt.figure(figsize=(8,6))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1]+[0.03]*(2), wspace=0.5)
        else:
            fig = plt.figure(figsize=(12,6))
            gs = gridspec.GridSpec(1, len(np.unique(tag))+2, width_ratios=[1]+[0.03]*(len(np.unique(tag))+1), wspace=0.5)
        if plot_2d:
            ax = fig.add_subplot(gs[0,0])
        else:
            ax = fig.add_subplot(gs[0,0], projection='3d')
    else:
        fig = plt.figure(figsize=(8,6))
        if plot_2d:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection='3d')
    color_list = ['deepskyblue', 'orange', 'blueviolet', 'crimson', 'forestgreen', 'magenta',
                  'powderblue', 'peachpuff', 'plum', 'pink', 'greenyellow', 'lavenderblush',
                  'darkblue', 'saddlebrown', 'indigo', 'maroon', 'darkgreen', 'darkmagenta']
    if not extra_color:
        extra_color = ['r','b','g','c','m','y']
    if plot_city:
        if not name:
            print('Need to input a sample name list.')
        else:
            colors = plt.get_cmap("gist_ncar")  # Many distinct colors
            cmap = [colors(i / 21) for i in range(21)]  # Sample 21 evenly spaced colors
            citytag = []
            for i in range(len(name)):
                citytag.append(int(name[i][0:2]))
            if isinstance(plot_2d,list): # axis is specified
                axis1 = plot_2d[0] - 1
                axis2 = plot_2d[1] - 1
                for j in range(1,22):
                    jth_proj = proj[[entry == j for entry in citytag]]
                    ax.scatter(jth_proj[:,axis1],jth_proj[:,axis2], s=20, alpha=0.7, c=cmap[j-1],
                               label='City %02d' % (j))
            else:
                ax.scatter(jth_proj[:,0],jth_proj[:,1], s=20, alpha=0.7, c=cmap[j-1],
                           label='City %02d' % (j))
            legend = ax.legend()
            handles, labels = ax.get_legend_handles_labels()
    else:
        if cluster_list:
            c_list = cluster_list
        else:
            c_list = range(max(tag)+1)
        for i in c_list:
            ith_proj = proj[[entry == i for entry in tag]] # filter samples for each cluster
            if z_value:
                ith_zvalue = [value for value, entry in zip(z_value,tag) if entry == i]
            if plot_SS:
                # scatter plot each cluster using a heat map, visualizing the samle SS
                custom_cmap = LinearSegmentedColormap.from_list("cmap_"+str(i), 
                                                                [color_list[i+6],color_list[i],color_list[i+12]])
                if cluster_list:
                    ith_ss_values = calc_relative_SS(data, tag, dist, i, c_list[not c_list.index(i)])
                else:
                    ss_values = silhouette_samples(data, tag)
                    ith_ss_values = ss_values[[entry == i for entry in tag]]
                _min = np.min(ith_ss_values)
                _max = np.max(ith_ss_values)
                if plot_2d:
                    if isinstance(plot_2d,list): # axis is specified
                        axis1 = plot_2d[0] - 1
                        axis2 = plot_2d[1] - 1
                        im_i = ax.scatter(ith_proj[:,axis1],ith_proj[:,axis2], s=20, alpha=0.7,
                                          c=ith_ss_values, cmap=custom_cmap, vmin=_min, vmax=_max)
                    else:
                        im_i = ax.scatter(ith_proj[:,0],ith_proj[:,1], s=20, alpha=0.7,
                                      c=ith_ss_values, cmap=custom_cmap, vmin=_min, vmax=_max)
                elif z_value:
                    im_i = ax.scatter(ith_proj[:,0],ith_proj[:,1], ith_zvalue[:], s=20, alpha=0.7,
                                      c=ith_ss_values, cmap=custom_cmap, vmin=_min, vmax=_max)
                else:
                    im_i = ax.scatter(ith_proj[:,0],ith_proj[:,1], ith_proj[:,2], s=20, alpha=0.7,
                                      c=ith_ss_values, cmap=custom_cmap, vmin=_min, vmax=_max)
                if c_list:
                    cbar_ax_i = fig.add_subplot(gs[0,c_list.index(i)+1])
                else:
                    cbar_ax_i = fig.add_subplot(gs[0,i+1])
                cbar_i = fig.colorbar(im_i, cax=cbar_ax_i)
                # custom colorbar
                cbar_i.set_ticks([])
                cbar_i.ax.text(0.5, -0.03, f'{_min:.2f}', ha='center', va='center', transform=cbar_i.ax.transAxes)  # Bottom
                cbar_i.ax.text(0.5, 1.03, f'{_max:.2f}', ha='center', va='center', transform=cbar_i.ax.transAxes)  # Top
                cbar_i.set_label('Cluster %d (%d)' % (i, len(ith_ss_values)))
            else:
                # color each cluster using a single color
                if plot_2d:
                    if isinstance(plot_2d,list): # axis is specified
                        axis1 = plot_2d[0] - 1
                        axis2 = plot_2d[1] - 1
                        ax.scatter(ith_proj[:,axis1],ith_proj[:,axis2], s=20, alpha=0.7, c=color_list[i],
                                   label='Cluster %d (%d)' % (i, len(ith_proj)))
                    else:
                        ax.scatter(ith_proj[:,0],ith_proj[:,1], s=20, alpha=0.7, c=color_list[i],
                                   label='Cluster %d (%d)' % (i, len(ith_proj)))
                elif z_value:
                    ax.scatter(ith_proj[:,0],ith_proj[:,1], ith_zvalue[:], s=20, alpha=0.7, c=color_list[i],
                               label='Cluster %d (%d)' % (i, len(ith_proj)))
                else:
                    ax.scatter(ith_proj[:,0],ith_proj[:,1], ith_proj[:,2], s=20, alpha=0.7, c=color_list[i],
                               label='Cluster %d (%d)' % (i, len(ith_proj)))
                legend = ax.legend()
                handles, labels = ax.get_legend_handles_labels()
    if is_UMAP:
        if not plot_2d and not z_value:
            # plot edges between samples for visual, only available for 3d umap plots
            index = []
            D = genDmatrix(proj)
            min_non_zero = D.replace(0, np.nan).min() # find distance to the closest sample
            t = min_non_zero.max() # find the largest, use this as threshold to build edges
            for i in range(len(D)):
              for j in range(i+1,len(D)):
                if D[i][j] <= t:
                  index.append((i,j))
            for i in range(len(index)):
                ce = 'gray'
                for j in range(max(tag)+1):
                    if tag[index[i][0]] == tag[index[i][1]] == j:
                      ce = color_list[j+12]
                ax.plot([proj[index[i][0]][0],proj[index[i][1]][0]], 
                        [proj[index[i][0]][1],proj[index[i][1]][1]],
                        [proj[index[i][0]][2],proj[index[i][1]][2]], color=ce, alpha=0.3)
        # # remove tick labels for all axes because doesn't have meaning for UMAP projection
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # if not plot_2d and not z_value:
        #     ax.set_zticklabels([])
        ax.set_xlabel('UMAP dimension 1')
        ax.set_ylabel('UMAP dimension 2')
        if not plot_2d:
            ax.set_zlabel('UMAP dimension 3')
    else:
        # axis title for pca proj
        if plot_2d and isinstance(plot_2d,list): # axis redefined
            axis_title = ['1st','2nd','3rd']
            titlex = axis_title[plot_2d[0]-1]
            titley = axis_title[plot_2d[1]-1]
            ax.set_xlabel(titlex+' component')
            ax.set_ylabel(titley+' component')
        else:
            ax.set_xlabel('1st component')
            ax.set_ylabel('2nd component')
        if not plot_2d and not z_value:
            ax.set_zlabel('3rd component') # add zlable if there's no measure to plot
    if z_value:
        ax.set_zlabel(z_title)
    if extra_plot:
        # emphasize some extra sample by name
        if plot_SS:
            handles, labels = [], [] # initialize for legend use
        if not name:
            print('Need to input a sample name list.')
        else:
            for i in extra_plot:
                ind = name.index(i)
                print('plotted: '+i+" in cluster %d" % tag[ind])
                extra_label = i+' (Cluster %d)' % tag[ind]
                if plot_2d:
                    if isinstance(plot_2d,list): # axis is specified
                        axis1 = plot_2d[0] - 1
                        axis2 = plot_2d[1] - 1
                        ax.scatter(proj[ind,axis1],proj[ind,axis2], marker="o", alpha=0.5, s=150,
                                   c=extra_color[extra_plot.index(i)], label=extra_label)
                    else:
                        ax.scatter(proj[ind,0],proj[ind,1], marker="o", alpha=0.5, s=150,
                                   c=extra_color[extra_plot.index(i)], label=extra_label)
                else:
                    ax.scatter(proj[ind,0],proj[ind,1],proj[ind,2], marker="o", alpha=0.5, s=150,
                               c=extra_color[extra_plot.index(i)], label=extra_label)
                # Collect handles and labels
                new_handles, new_labels = ax.get_legend_handles_labels()
                handles.extend(new_handles[len(handles):])  # Only add new handles
                labels.extend(new_labels[len(labels):])
    if plot_SS and extra_plot:
        ax.legend(handles, labels)
    elif not plot_SS:
        plt.subplots_adjust(right=0.8)
        if plot_city:
            ax.legend(handles, labels, bbox_to_anchor=(1.3, 1.15), fontsize=13)
        else:
            ax.legend(handles, labels, bbox_to_anchor=(1.3, 1)) # without colormap adjust the legend position
    if plot_2d and plot_quad: # plot Q1 and Q3 for the two axis
        if isinstance(plot_2d,list): # axis is specified
            axis1 = plot_2d[0] - 1
            axis2 = plot_2d[1] - 1
        else:
            axis1 = 0
            axis2 = 1
        data1 = proj[:,axis1]
        # x coord
        q1_x = np.percentile(data1,25) # first quartile
        q2_x = np.percentile(data1,50) # median
        q3_x = np.percentile(data1,80) # third quartile
        x_existing_ticks = [round(value,2) for value in ax.get_xticks()]
        x_existing_labels = [str(label) for label in x_existing_ticks]
        x_extra_ticks = [q1_x, q3_x]
        x_extra_labels = ["P25", "P80"]
        x_all_ticks = list(x_existing_ticks) + x_extra_ticks
        x_all_labels = x_existing_labels + x_extra_labels
        ax.set_xticks(x_all_ticks)
        ax.set_xticklabels(x_all_labels)
        for label in ax.get_xticklabels():
            if label.get_text() in x_extra_labels:  # Change these to bold
                label.set_fontweight("bold")
        # y coord
        data2 = proj[:,axis2]
        q1_y = np.percentile(data2,25) # first quartile
        q2_y = np.percentile(data2,50) # median
        q3_y = np.percentile(data2,80) # third quartile
        y_existing_ticks = [round(value,2) for value in ax.get_yticks()]
        y_existing_labels = [str(label) for label in y_existing_ticks]
        y_extra_ticks = [q1_y, q3_y]
        y_extra_labels = ["P25","P80"]
        y_all_ticks = list(y_existing_ticks) + y_extra_ticks
        y_all_labels = y_existing_labels + y_extra_labels
        ax.set_yticks(y_all_ticks)
        ax.set_yticklabels(y_all_labels)
        for label in ax.get_yticklabels():
            if label.get_text() in y_extra_labels:  # Change these to bold
                label.set_fontweight("bold")
        # plot line
        for value in [q1_x,q3_x]:
            ax.axvline(x=value, color='red', linestyle='--', linewidth=1)
        ax.axvline(x=q2_x, color='red', linestyle=':',linewidth=1)
        for value in [q1_y,q3_y]:
            ax.axhline(y=value, color='cyan', linestyle='--', linewidth=1)
        ax.axhline(y=q2_y, color='cyan', linestyle=':',linewidth=1)
    ax.set_title(title)
    plt.show()
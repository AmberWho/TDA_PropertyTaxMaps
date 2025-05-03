"""
commands_processing.py

Author:   Amber, Anbo Wu
Date:     May 2025
Project:  Topological Data Analysis in an Economic Context: Property Tax Maps
"""

import os
#PATH = "/path/to/your/directory" # folder path containing the data file
PATH = "C:/Projects/TDA/_0Codes/"
os.chdir(PATH)

import numpy as np
import pandas as pd
import copy

dimension = 0       # dimension = {0, 1}, 0 = PH_0, 1 = PH_0^*
match dimension:
    case 0:
        dimension_title = "$\mathrm{PH}_0$"
        final_cluster = 6
    case 1:
        dimension_title = "$\mathrm{PH}_0^*$"
        final_cluster = 5

# %% Cell[1] Import data

"""
1. Load persistence images, average and standard deviation of property taxes, and HHI for each sample
"""
pi = np.load("pih" + str(dimension) + ".npy", allow_pickle = True)
mean = np.load(PATH + "mean.npy")
std = np.load(PATH + "std.npy")
df = pd.read_csv(PATH + "HHI.csv")
hhi = df["HHI"].to_numpy()

# %% Cell[2] Outlier

"""
2. Identify and remove outliers
"""
from python_functions import genName
from python_functions import genDmatrix
from python_functions import find_outliers_zscore
name = genName()
dist = genDmatrix(pi,name)
df = copy.deepcopy(dist)
df['sum of distance'] = dist.sum(axis=1)
sumdist = df['sum of distance'].to_list()
print('The average sum of distance per sample is ' + str("%0.2f" % df['sum of distance'].mean()))
""" Console Outputs
=== PH0 ===
The average sum of distance per sample is 146.97
=== PH0* ==
The average sum of distance per sample is 235.66
"""
outliers = find_outliers_zscore(sumdist)
print(df['sum of distance'].nlargest(10))
""" Console Outputs
=== PH0 ===
1709    1080.755066
0307     586.556682
1007     547.408487
1001     344.528018
1507     334.509549
1003     308.233322
1006     285.718727
1002     272.586214
0103     258.820557
0706     249.407834
=== PH0* ==
1007    1044.079563
1004     761.442898
1002     754.797460
1005     737.560329
1001     672.040578
1003     608.321876
1006     586.172180
1506     544.956355
1509     531.300699
1508     487.172700
"""
# = PH0 ===
# index: "1709", "0307", "1007"
Outliers = [168,26,95] 
# = PH0* ==
# index: "1007", "1004", "1002", "1005", "1001", "1003"
# Outliers = [96,93,91,92,90,90] 
name_re = copy.deepcopy(name)
pi_re = pi
for outlier in Outliers:
    pi_re = np.delete(pi_re, outlier, axis = 0)
    print('Removed from name list:',end=" ")
    print(name_re.pop(outlier)) # suppress output of pop()
dist_re = genDmatrix(pi_re,name_re)

# %% Cell[3] Iteration-based methods

"""
3. Various evaluation on iteration-based clustering

See paper Section 4.1.

File output(s):
    clustering_result.txt: general summary of iteration-based clustering using 3 different initializations
"""
from python_functions import printCluster
from python_functions import compute_avg_ss
with open("clustering_result.txt", "w") as f:
    methods = ['heuristic', 'random', 'k-medoids++']
    clusters = [5,6]
    iteration = 100
    print("Row index: cluster indices, range from 0 to (number of clusters-1)\n\n" +
          "Size: number of samples in each cluster\n\n" + 
          "Distance: for sample A in the given cluster, compute the average distance from A\n" +
          "  to all the other samples within the same cluster, and take the average over\n" +
          "  all sample A's in the cluster\n\n" +
          "Diameter: maximum pairwise distance among the given cluster\n\n" +
          "SS: Silhouette Score, [wrong assignment -1 <--- overlapping clusters 0 --> good assignment 1]\n" +
          "  recorded is the average SS per sample over the given cluster\n\n"
          "Average, Maximum, Minimum: statistics for multiple iterations\n\n\n", file = f)
    for cluster in clusters:
        print("Cluster:", cluster)
        for method in methods:
            print("Method:", method)
            tag_count, med_count = printCluster(pi_re, dist_re, name_re, cluster, iteration, method, f)
            print(method + ' method, k = %d, number of local solutions: %d, maximum average SS: %0.4f' 
                  % (cluster, len(tag_count.columns), max(compute_avg_ss(tag_count, pi_re))))
            print("", file = f)
            print("")
        print("==============", file = f)
""" Console Outputs (the heuristic method output suppressed)
=== PH0 ===
The average sum of distance per sample is 146.97
random method, k = 5, number of local solutions: 1, maximum average SS: 0.1869
kmedoids++ method, k = 5, number of local solutions: 1, maximum average SS: 0.1869
random method, k = 6, number of local solutions: 2, maximum average SS: 0.2129
kmedoids++ method, k = 6, number of local solutions: 2, maximum average SS: 0.2129
=== PH0* ==
random method, k = 5, number of local solutions: 4, maximum average SS: 0.2470
kmedoids++ method, k = 5, number of local solutions: 4, maximum average SS: 0.2470
random method, k = 6, number of local solutions: 7, maximum average SS: 0.2310
kmedoids++ method, k = 6, number of local solutions: 7, maximum average SS: 0.2310
"""

# %% Cell[4] PCA-related plots

"""
4. Visual representations related to the Principal Component Analysis (PCA)

See paper Section 4.2, 4.3.
"""
import matplotlib.pyplot as plt
from python_functions import genPCA
pca, pc = genPCA(pi_re, n=100, return_pc = True)
variance = pc.explained_variance_ratio_
total_variance = np.zeros((100))
for i in range(len(variance)):
    total_variance[i] = np.sum(variance[0:i])
fig100 = plt.figure(100)
ax100 = fig100.add_subplot(111)
ax100.plot(total_variance)
thrd_list = [1,2,3,10] # set a component number threshold
for threshold in thrd_list:
    ax100.plot(threshold,total_variance[threshold],'ro')
    ax100.annotate('(%s, %0.5f)' % (threshold,total_variance[threshold]),
                   (threshold+2.5,total_variance[threshold]-0.03),fontsize=13)
plt.xlabel("number of principal components",fontsize=13)
plt.ylabel("sum of variance",fontsize=13)
plt.title("PCA, sum of explained variance ratio, " + dimension_title,fontsize=14)
plt.show()
pi_pca,pc = genPCA(pi_re, n=10, return_pc=True)
comp = pc.components_
_min = comp.min()
_max = comp.max()
exp = pc.explained_variance_ratio_
fig, axes = plt.subplots(1, 3, figsize=(16,5))
titles = []
for i in range (3):
    titles.append("Component %d, explained variance ratio %0.2f" % (i+1,exp[i]))
for i, ax in enumerate(axes.flat):
    data = comp[i].reshape((100,100))
    im = ax.imshow((data), cmap=plt.get_cmap("CMRmap"), vmin = _min, vmax = _max)
    # plot PI ticks
    plt.rcParams.update({'font.size': 15})
    xticks = np.arange(0, 13, 2) * 99 / 13
    yticks = np.arange(1, 14, 2) * 99 / 13
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    xlabels = [str(label) for label in list(range(0,14,2))]
    ylabels = [str(label) for label in list(range(12,-2,-2))]
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.set_title(titles[i],fontsize=14)
    fig.colorbar(im, orientation='vertical', shrink=0.7)
plt.suptitle("PCA, 3 components visualization, " + dimension_title, fontsize=15)
plt.tight_layout()
plt.show()

# %% Cell[5] Clustering results plots

"""
5. Visual representations related to the clustering results

See paper Section 4.2, 4.5, 5.

File output(s):
    Null.txt: byproduct, no use
"""
from python_functions import plotCluster
from python_functions import genUMAP
from python_functions import calc_relative_SS
with open("Null.txt","w") as f: # pass a txt file object for the function
    iteration = 100
    tag_random_, med_random_, count_random_ = printCluster(pi_re, dist_re, name_re, final_cluster, iteration, 'random', f, ctmed=True)
ss_random_ = compute_avg_ss(tag_random_, pi_re)
print(max(ss_random_))
max_index = ss_random_.index(max(ss_random_))
tag_fin = tag_random_[str(max_index+1)]
medoids_list = med_random_[str(max_index+1)].to_list()
medoids_sorted_list = []
for i in range(final_cluster):
    medoid_i = [item for item in medoids_list if tag_fin[name_re.index(item)] == i]
    medoids_sorted_list.append(medoid_i[0])
p_proj = genPCA(pi_re)
mean_re = copy.deepcopy(mean).tolist()
std_re = copy.deepcopy(std).tolist()
hhi_re = copy.deepcopy(hhi).tolist()
name_test = copy.deepcopy(name)
# = PH0 ===
# index: "1709", "0307", "1007"
Outliers = [168,26,95] 
# = PH0* ==
# index: "1007", "1004", "1002", "1005", "1001", "1003"
# Outliers = [96,93,91,92,90,90] 
print('Removed the data corresponding to the following sample:')
for outlier in Outliers:
    print(name_test.pop(outlier))
    mean_re.pop(outlier)
    std_re.pop(outlier)
    hhi_re.pop(outlier)

# pca (3d)
plotCluster(pi_re, tag_fin, p_proj, dist_re, "PCA, " + dimension_title)

# umap for dff neighborhood size (3d)
for b in [100,200]:
    u_proj = genUMAP(pi_re,b)
    plotCluster(pi_re, tag_fin, u_proj, dist_re,
                "UMAP, " + dimension_title + ", \u03B2 = %d" % b, is_UMAP=True)

# pca with medoids (2d)
plotCluster(pi_re, tag_fin, p_proj, dist_re, "PCA, " + dimension_title,
            extra_plot = medoids_sorted_list, name=name_re, plot_2d = [1,2], plot_quad=True)
plotCluster(pi_re, tag_fin, p_proj, dist_re, "PCA, " + dimension_title,
            extra_plot = medoids_sorted_list, name=name_re, plot_2d = [1,3], plot_quad=True)
plotCluster(pi_re, tag_fin, p_proj, dist_re, "PCA, " + dimension_title,
            extra_plot = medoids_sorted_list, name=name_re, plot_2d = [3,2], plot_quad=True)

# city
plotCluster(pi_re, tag_fin, p_proj, dist_re, "PCA, " + dimension_title, plot_SS=False, plot_2d=[1,2],
            plot_city=True, name=name_re)

# plot against other measures
titles = ["tax value mean","tax value standard deviation","HHI"]
data_list = [mean_re,std_re,hhi_re]
for title, measure in zip(titles,data_list):
    plotCluster(pi_re, tag_fin, p_proj, dist_re, "PCA, " + dimension_title + ", %s" % title,
                z_value=measure, z_title=title)

# boundary sample - exclusively for PH_0
name_arr = np.array(name_re)
ss_list_c2 = calc_relative_SS(pi_re, tag_fin, dist_re, 2, 5)
name_c2 = name_arr[[entry == 2 for entry in tag_fin]]
bdr_c2 = name_c2[np.argmin(ss_list_c2)]
ss_list_c5 = calc_relative_SS(pi_re, tag_fin, dist_re, 5, 2)
name_c5 = name_arr[[entry == 5 for entry in tag_fin]]
bdr_c5 = name_c5[np.argmin(ss_list_c5)]
boundary_list = ['1405',bdr_c2,'0510',bdr_c5]
c_list = ['g','c','y','b']
plotCluster(pi_re, tag_fin, p_proj, dist_re, "PCA, $\mathrm{PH}_{0}$, Cluster 2 and 5",
            plot_2d = [1,2], cluster_list=[2,5], extra_plot=boundary_list, name=name_re, extra_color=c_list)
plotCluster(pi_re, tag_fin, p_proj, dist_re, "PCA, $\mathrm{PH}_{0}$, Cluster 2 and 5",
            plot_2d = [1,3], cluster_list=[2,5], extra_plot=boundary_list, name=name_re, extra_color=c_list)

# sample 0204, 1509 - exclusively for PH_0
plotCluster(pi_re, tag_fin, p_proj, dist_re, "PCA, $\mathrm{PH}_{0}$",
            extra_plot = ['0204','1509'], name=name_re, plot_2d = [1,2], plot_quad=True, extra_color=['b','r'])
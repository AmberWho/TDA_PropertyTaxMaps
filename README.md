__This Github site is currently under construction. It will provide data and code supporting [David Damiano, Anbo Wu, "Topological Data Analysis in an Economic Context: Property Tax Maps"](https://ieeexplore.ieee.org/document/9671276).__

# Supplementary materials to Topological Data Analysis in an Economic Context: Property Tax Maps
This is a joint work by David Damiano (Department of Mathematics and Computer Science, College of the Holy Cross, ddamiano@holycross.edu) and Anbo Wu (Department of Mathematics, The Ohio State University, wu.3488@osu.edu).
## Available Files
### Raw property tax maps and grayscale images
Property tax maps provided by PropertyShark are available in [Maps](/Maps), which contains 21 subfolders, each corresponding to one city, ordered by the total population. 

In each city folder, the file _00.txt_ records the Census tracts contained by each map parcel (a tract will be included if roughly over 3/4 of the area is contained in the parcel). Property tax map parcels are named in the format of _city index + sample index + .png_, and the corresponding grayscale images are named in the format of _city index + sample index + gray.png_.

To reproduce the grayscale version of a map parcel, for example, _0101.png_, place the map parcel and _grayscale.py_ (available in [Codes](/Codes)) in the same folder, and:
1. Install _PIL_ package for Python
2. Run `python grayscale.py "0101"` from the command prompt in the current folder
3. Wait until seeing `|=  Finish! (,,•ω•,,)  -=-------` message in the command prompt
4. A grayscale version will be produced, titled _0101gray.png_

### Erosion method
After the erosion method, all processed property tax maps are available in [Modmaps](/Mpdmaps).

To perform the erosion method on a grayscale map parcel, for example, _0101gray.png_, place the grayscale image, _python_functions.py_ and _python_commands.py_ (available in [Codes](/Codes)) in the same folder, and:
1. Install _PIL_, _numpy_ package for Python
2. In the current folder, open _python_commands.py_, and run ```cell #0``` (import functions, initialize variables) with the desired image index (in this case set ```filename = "0101"```)
3. Run ```cell #1```
4. The processed image will be produced, titled _mod0101.png_

### Persistence Diagrams and Persistence Images
Persistence diagrams and images are available in [H0](/H0), for H_0 results, and [H1](/H1), for H_0^* results. We use H1 instead of H_0^* particularly for the ease of coding. Persistence diagrams have _pd_ as suffixes. Persistence images are plotted using two color schemes: The first one is based on the maximum and minimum value of the persistence image for each sample (these have suffixes _upi_); the other one is based on the maximum and minimum value of all persistence images (these have suffixes _pi_). Persistence images plotted with the first color scheme are useful when one wants to observe characteristics within a single sample, and persistence images plotted with the second color scheme are useful when comparing different persistence images.

Numpy arrays containing the persistence diagrams and images are available in [data](/data). The folder includes the following: (1) _pdh0.npy_ persistence diagrams of the H_0 analysis, _pdh1.npy_ persistence diagrams of the H_0^* analysis; (2) _pih0.npy_ all persistence images of the H_0 analysis, _pih1.npy_ all persistence images of the H_0^* analysis; (3) _repih0.npy_ persistence images used in the H_0 analysis (all outliers have been removed, to reproduce this please check the next section), _repih1.npy_ persistence images used in the H_0^* analysis (all outliers removed).

To produce the persistence diagram based on the image after the erosion method was applied, place _mod0101.png_, _python_functions.py_ and _python_commands.py_ (available in [Codes](/Codes)) in the same folder, and:
1. Install _PIL_, _numpy_, _matplotlib_, _copy_, _ripser_ package for Python
2. In the current folder, open _python_commands.py_, and run ```cell #0``` (import functions, initialize variables) with the desired image index (in this case set ```filename = "0101"```)
3. Run ```cell #2```
4. The persistence diagram will be produced, titled _0101pd.png_

To produce the persistence image, place _mod0101.png_, _python_functions.py_ and _python_commands.py_ (available in [Codes](/Codes)) in the same folder, and:
1. Install _PIL_, _numpy_, _matplotlib_, _copy_, _ripser_, _persim_ package for Python
2. In the current folder, open _python_commands.py_, and run ```cell #0``` (import functions, initialize variables) with the desired image index (in this case set ```filename = "0101"```)
3. Run ```cell #3```
4. The persistence image with the color scheme based on the maximum and minimum value of sample 0101 will be produced, titled _0101upi.png_
5. Place _repih0.npy_ or _repih1.npy_ in the current folder based on the desired dimension
6. Run ```cell #4```
7. The persistence image with the color scheme based on the maximum and minimum value of all samples will be produced, titled _0101pi.png_

## Frequency matrix and PCA
Please check our paper for detailed reports. To reproduce the result, run _ClusterH0.mlx_(for H_0 analysis) or _ClusterH1.mlx_(for H_0^*) analysis) from [Codes](/Codes) in the same folder with _pi_vector.mat_ or _pi_vector1.mat_ (These Matlab datafiles contain the same persistence image data generated by the last section.)

## Representative Clustering Results

## Land Use Maps of Samples from California

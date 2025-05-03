__This Github site is currently under construction. It will provide data and code supporting [David Damiano, Anbo Wu, "Topological Data Analysis in an Economic Context: Property Tax Maps"](https://ieeexplore.ieee.org/document/9671276).__

__Table of contents by folders and files:__

[Codes](/Codes)  - [ _commands\_processing.py_](#samples-and-image-processing-related-files) and [_commands\_evaluation.py_](#clustering-and-evaluation)

[Data](/Data) - [_pdh0.npy_, _pdh1.npy_, _pdh0\_fin.npy_, _pdh1\_fin.npy_](#persistence-diagrams-and-persistence-images)

[H0](/H0)/[H1](/H1) - [Persistence Diagrams and Persistence Images](#persistence-diagrams-and-persistence-images)

[Landuse](/Landuse) - [Land Use Maps](#land-use-maps-of-samples-from-california)

[Maps](/Maps)/[Modmaps](/Modmaps) - [Persistence Diagrams and Persistence Images](#persistence-diagrams-and-persistence-images)

# Supplementary materials to Topological Data Analysis in an Economic Context: Property Tax Maps
This is a joint work by David Damiano (Department of Mathematics and Computer Science, College of the Holy Cross, ddamiano@holycross.edu) and Anbo Wu (Department of Mathematics, The Ohio State University, wu.3488@osu.edu).
## Samples and Image-processing-related Files
This section outlines where readers may find all the image-processing results, stored by different stages.
Readers may reproduce each stage of the image-processing results of our analysis pipeline using _commands\_processing.py_ and _python\_functions.py_ (available in [Codes](/Codes)).

### Raw property tax maps and grayscale images
Property tax maps sampled from PropertyShark are available in [Maps](/Maps), which contains 21 subfolders, each corresponding to one city, ordered by the total population. 

In each city folder, the file _00.txt_ records the Census tracts contained by each map parcel (a tract will be included if roughly over 3/4 of the area is contained in the parcel). Property tax map parcels are named in the format of _city index + sample index + .png_ (city index ranges from 1 to 21 and samples index ranges from 1 to 10), and the corresponding grayscale images are named in the format of _city index + sample index + gray.png_.

To reproduce the grayscale version of a map parcel, place _commands\_processing.py_ and _python\_functions.py_ in the same folder.
- __Required package(s)__: _PIL_
- __Pre-processing__: Run ```Cell[0]```
- __Processing__: Run ```Cell[1]```
- __Output__: grayscale image titled _city index + sample index + gray.png_

### Erosion method
After the erosion method, all processed property tax maps are available in [Modmaps](/Modmaps).

To perform the erosion method on a grayscale map parcel _city index + sample index + gray.png_:
- __Required package(s)__: _PIL_, _numpy_
- __Pre-processing__: Run ```Cell[0]```. Run all other cells up to ```Cell [2]``` or place _city index + sample index + gray.png_ in the same folder in ```PATH```
- __Processing__: Run ```Cell[2]```
- __Output__: processed image titled _mod + city index + sample index + .png_

### Persistence Diagrams and Persistence Images
Persistence diagrams and images are available in [H0](/H0), for PH<sub>0</sub> results, and [H1](/H1), for PH<sub>0</sub><sup>\*</sup> results. We use H1 instead of PH<sub>0</sub><sup>\*</sup> particularly for the ease of coding. Persistence diagrams have _-pd_ as suffixes. Persistence images are plotted using two color schemes: The first one is based on the maximum and minimum value of the persistence image for each sample (these have suffixes _-upi_); the other one is based on the maximum and minimum value of all persistence images (these have suffixes _-pi_). Persistence images plotted with the first color scheme are useful when one wants to observe characteristics within a single sample, and persistence images plotted with the second color scheme are useful when comparing different persistence images.

Numpy arrays containing the persistence diagrams and images are available in [Data](/Data). The folder includes the following: (1) _pdh0.npy_ persistence diagrams of the PH<sub>0</sub> analysis, _pdh1.npy_ persistence diagrams of the PH<sub>0</sub><sup>\*</sup> analysis; (2) _pih0.npy_ all persistence images of the PH<sub>0</sub> analysis, _pih1.npy_ all persistence images of the PH<sub>0</sub><sup>\*</sup> analysis; (3) _pih0\_fin.npy_ persistence images used in the PH<sub>0</sub> clustering and evaluation (_pih0.npy_ with all outliers removed), _pih1\_fin.npy_ persistence images used in the PH<sub>0</sub><sup>\*</sup> clustering and evaluation (_pih1.npy_ with all outliers removed).

To produce the persistence diagram based on the image after applying the erosion method:
- __Required package(s)__: _PIL_, _numpy_, _matplotlib_, _copy_, _ripser_
- __Pre-processing__: Run ```Cell[0]```. Run all other cells up to ```Cell [3]``` or place _mod + city index + sample index_ in the same folder in ```PATH```
- __Processing__: Run ```Cell[3]```
- __Output__: persistence diagram titled _city index + sample index + pd.png_

To produce the persistence image:
- __Required package(s)__: _PIL_, _numpy_, _matplotlib_, _copy_, _ripser_, _persim_
- __Pre-processing__: Run ```Cell[0]```. Run all other cells up to ```Cell [3]``` or place _mod + city index + sample index_ in the same folder in ```PATH```. To process (2), place _nameh + dimension + \_fin.npy_ in the same folder in ```PATH```
- __Processing__: (1) Run ```Cell[4]``` (2) Run ```Cell[5]```
- __Output__: (1) persistence image with color scheme based on maximum and minimum value of sample _city index + sample index_ titled _city index + sample index + upi.png_ (2) all persistence images with the color scheme based on the maximum and minimum value of all samples with suffixes _-pi_.

## Clustering and Evaluation

Readers may reproduce all the other plots presented in the paper, including a summary of iteration-based clustering results and visual presentations of the clustering results, using _commands\_evaluation.py_ and _python\_functions.py_ (available in [Codes](/Codes)).
Authors choose to omit the instructions. Please refer to the in-line comments of _commands\_evaluation.py_ for further details.

## Land Use Maps of Samples from California

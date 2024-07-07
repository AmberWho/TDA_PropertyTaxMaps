__This Github site is currently under construction. It will provide data and code supporting [David Damiano, Anbo Wu, Topological Data Analysis in an Economic Context: Property Tax Maps](https://ieeexplore.ieee.org/document/9671276)__

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

### Filtrations


## Representative Clustering Results

## Land Use Maps of Samples from California

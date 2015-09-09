# Texton-Map-Generation-
cmake .
make
./textonMap <FLAG> <k> <N>
 FLAG: 1 - to compute kmeans centers, 0 - to generate texton map
 k: no of dictionary words,
 N: Num of training images to use (here between 1 and 50)
# This code tests for an exact zero difference when we compare two images (Image1, Image2), i.e, Image, 'Image1',
# with a 180 degree rotated version of itself, 'Image2'.
# TM2, TM1 are the texton Maps generated for these images(Image1, Image2) where the texton map is flipped as well.
# The file tm.txt stores all the image values, texton Map values and the difference between the two texton maps. 
# The file FR.txt gives the filter responses for every training and testing image. 

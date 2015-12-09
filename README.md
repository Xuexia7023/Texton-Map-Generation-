# Texton-Map-Generation-
<ol>
<li>cmake .</li>
<li>make</li>
<li>./textonMap FLAG k N</li>
</ol>
<br> FLAG: 1 - to compute kmeans centers, 0 - to generate texton map
<br> k: no of dictionary words,
<br> N: Num of training images to use (here between 1 and 50) <br>
<em> <strong>Note: </strong>The folder in which training images are stored must be in "example_roads/im%d.jpg" format where %d varies from 1 to N(or less than N, but not more), given above</em>
<p> This code tests for an exact zero difference when we compare two images (Image1, Image2), i.e, Image, 'Image1',
 with a 180 degree rotated version of itself, 'Image2'.</p>
<p> TM2, TM1 are the texton Maps generated for these images(Image1, Image2) where the texton map is flipped as well.</p>
<p> The file tm.txt stores all the image values, texton Map values and the difference between the two texton maps. </p>
<p> The file FR.txt gives the filter responses for every training and testing image. </p>

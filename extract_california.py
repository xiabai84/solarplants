"""

Plan:
zip the lines from polygonVertices_PixelCoordinates.csv and polygonDataExceptVertices.csv

the entry from polygonDataExceptVertices.csv tells us which .tif file to load (so we do that).
Make sure that polygonDataExceptVertices.csv is ordered by filename so each image is only loaded once!

then take the average location of pixels from polygonVertices_PixelCoordinates and cut out a 300x300 pixel area around
this, and take a look at ca. 10 such images. is 300x300 too much or too little? adjust the value.

Try to find the exact pixel per meter for both Google and these images to make a better conversion.

"""
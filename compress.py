import numpy
from matplotlib import pyplot

#This function will take the flattened image data (DATA) as input, along with k, the number of principal
#components to use. This function will use PCA to find the principal components of the face images. It will
#return the compressed data. X compressed = Z ∗ U T . Where Z ∗ is the projected data and U T is the transpose
#of the principal components. This function will then output the compressed images into a directory named
#Output. You should use the os package to make sure that the directory is present and to create it if it does
#not exist. NOTE: Images have values from 0 to 255, so you will want to rescale them before saving them.
#You will also want to use the cmap=’gray’ option in pyplot.imsave to save them as grayscale images.
def compress_images(DATA,k):

#The above function takes the input directory as input, and outputs the DATA matrix. DATA will have one
#flattened image per column, so each column represents an image and one row represents the pixel values for
#every image at a particular location. This function will use pyplot.imread to load the images. Before you
#return DATA you will want to convert it to floating point.
def load_data(input_dir):

import compress
import numpy as np

# Real training
train = 'Data/Train/'
small = 'Data/small/'

X = compress.load_data(small)
compress.compress_images(X,100)

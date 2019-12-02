import matplotlib.pyplot as plot
import numpy as np
import os
from pca import *



def load_data (input_dir):
    
    images = os.listdir(input_dir)    
    reading_first_img = True
    count = 0
    
    for image in images:
        count += 1
        
        temp_img = plot.imread(input_dir+image)
        temp_img = temp_img.astype(float)
        temp_img = temp_img.flatten('C')
        temp_img  = np.reshape(temp_img, (1, temp_img.shape[0]))
        
        if reading_first_img:
            img_arrays = np.empty((0, temp_img.shape[0]), float)
            img_arrays = temp_img
            reading_first_img = False
        else:
            img_arrays = np.append(img_arrays, temp_img, axis=0)
    return img_arrays.transpose()
    


def compress_images(DATA, k):
    print(DATA.shape)
    Z = compute_Z(DATA)
    print('Z = ', Z.shape)    
    COV = compute_covariance_matrix(Z)
    print("COV = ",COV.shape)
    L, PCS = find_pcs(COV)
    print('Vals = ',L.shape,'--> Vects = ',PCS.shape)
    Z_star = np.dot(Z, PCS[:,:k])
    print(Z_star.shape)

    
    x_compress = np.dot(Z_star, PCS[:,:k].transpose())
    print(x_compress.shape)
    
    if not os.path.exists('Output_2'):
        os.makedirs('Output_2')

    
    for i in range(DATA.shape[1]):
        face = np.reshape(x_compress[:,i], (60,48))
        plot.imsave('Output_2/image_'+str(i+1)+'.png', face, cmap='gray')

    


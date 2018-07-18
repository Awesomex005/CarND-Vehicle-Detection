import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from extract_feature import *
from auto_subplot import *

cars = glob.glob('./hog_test_imgs/vehicles_smallset/*/*.jpeg')
notcars = glob.glob('./hog_test_imgs/non-vehicles_smallset/*/*.jpeg')
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

car_image = mpimg.imread(cars[car_ind])
noncar_image = mpimg.imread(notcars[notcar_ind])
car_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
noncar_image = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2YCrCb)

car_ch1 = car_image[:,:,0]
car_ch2 = car_image[:,:,1]
car_ch3 = car_image[:,:,2]
noncar_ch1 = noncar_image[:,:,0]
noncar_ch2 = noncar_image[:,:,1]
noncar_ch3 = noncar_image[:,:,2]
imgs = [car_ch1, noncar_ch1, car_ch2,  noncar_ch2, car_ch3, noncar_ch3]
imgs_names = ["car_ch1", "noncar_ch1", "car_ch2", "noncar_ch2", "car_ch3", "noncar_ch3"]

orient= 9
pix_per_cell= 8
cell_per_block= 2

out_img_names = []
out_imgs = []
out_img_camps = []
for ixd, img in enumerate(imgs):
    features, hog_image = get_hog_features(img, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)
                        
    out_img_names.append(imgs_names[ixd])
    out_imgs.append(img)
    out_img_camps.append('gray')
    out_img_names.append(imgs_names[ixd]+" HOG")
    out_imgs.append(hog_image)
    out_img_camps.append('gray')                    

multi_subplot(out_img_names, out_imgs, 4, out_img_camps)
plt.show()
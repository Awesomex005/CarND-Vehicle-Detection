import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from extract_feature import *

# Read in our vehicles
car_images = glob.glob('./train_data/*/*/*.png')
#car_images = glob.glob('./hog_test_imgs/*/*/*.jpeg')

# Generate a random index to look at a car image
ind = np.random.randint(0, len(car_images))
print("img ind: {}".format(ind))
ind=17433
# Read in the image
image = mpimg.imread(car_images[ind])
print("img min pixel value: {}".format(np.min(image)))
#image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
ch1 = image[:,:,0]
ch2 = image[:,:,1]
ch3 = image[:,:,2]
#gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

orient= 9
pix_per_cell= 8
cell_per_block= 2

# Call our function with vis=True to see an image output
features1, hog_image1 = get_hog_features(ch1, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)
print("sum of features1: {}".format(sum(features1)))
# Call our function with vis=True to see an image output
features2, hog_image2 = get_hog_features(ch2, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)
print("sum of features2: {}".format(sum(features2)))
                        # Call our function with vis=True to see an image output
features3, hog_image3 = get_hog_features(ch3, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)
print("sum of features3: {}".format(sum(features3)))

# Plot the examples
fig = plt.figure()
plt.subplot(231)
plt.imshow(ch1, cmap='gray')
plt.title('ch1')
plt.subplot(232)
plt.imshow(ch2, cmap='gray')
plt.title('ch2')
plt.subplot(233)
plt.imshow(ch3, cmap='gray')
plt.title('ch3')
plt.subplot(234)
plt.imshow(hog_image1, cmap='gray')
plt.title('HOG Visualization')
plt.subplot(235)
plt.imshow(hog_image2, cmap='gray')
plt.title('HOG Visualization')
plt.subplot(236)
plt.imshow(hog_image3, cmap='gray')
plt.title('HOG Visualization')

plt.show()
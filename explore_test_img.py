import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
#from skimage.feature import hog
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles


example_img = mpimg.imread("test_images/test4.jpg")
print("example_img.shape: {}".format(example_img.shape))
print("example_img.dtype: {}".format(example_img.dtype))
print("max pixel value: {}".format(np.max(example_img)))


# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(example_img)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(example_img[:,:,0])
plt.title('Example Not-car Image')

plt.show()
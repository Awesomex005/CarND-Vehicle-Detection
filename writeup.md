
**Vehicle Detection**

Mine steps of completing this project:

* Perform Histogram of Oriented Gradients (HOG) feature extraction, binned color feature extraction, histograms of color feature extraction on a labeled training set of images
* Normalize features and randomize a selection for training and testing.
* Train Linear SVM classifier with extracted features.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Build a pipeline on a video stream and create heat maps of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: https://raw.githubusercontent.com/Awesomex005/CarND-Vehicle-Detection/master/report_images/car_noncar.png
[image2]: https://raw.githubusercontent.com/Awesomex005/CarND-Vehicle-Detection/master/report_images/car_noncar_hog.png
[image3]: https://raw.githubusercontent.com/Awesomex005/CarND-Vehicle-Detection/master/report_images/sliding_windows.png
[image4]: https://raw.githubusercontent.com/Awesomex005/CarND-Vehicle-Detection/master/report_images/process_frame.png
[video1]: https://raw.githubusercontent.com/Awesomex005/CarND-Vehicle-Detection/master/output_videos/Vehicle_detection.mp4


# Feature Extraction
### Histogram of Oriented Gradients (HOG)

#### 1. extracted HOG features from the training images.

The code for this step is contained in lines 71 through 82 of the file called `extract_feature.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. HOG parameters

I choose to use YCrCb color space 3-channels since it with HOG could gives a stable shape information. I also tried other color space like RGB or HLV. It turns out that RGB 3-channels HOG is redundant, and HLV with HOG could not give a stable shape information.
I tried various combinations of parameters and found that `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` gives a really good result in both  distinguishable and feature size aspects.

#### 3. Train classifier

The code for this step is contained in `train_classifier.py`.

I trained a linear SVM using sclared combined features, HOG features combining with binned color feature and histograms of color feature.

### Sliding Window Search

#### 1. Sliding window

I decided to search on 1.4 & 1.5 scales with 75% overlap. A relativly high scales to reduce windows and search time. A relarivly high overlap to ensure we don't miss vehicles objects.

#### 2. Search with classifier

Ultimately I got a nice result.  Here are some example images:

![alt text][image3]
---

### Video Implementation

The code for this step is contained in `vehicle_detection.py`.

#### 1. Video result
Here's a [link to my video result](https://raw.githubusercontent.com/Awesomex005/CarND-Vehicle-Detection/master/output_videos/Vehicle_detection.mp4)


#### 2. Filter out false positives and combine overlapping bounding boxes for each frame

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap some frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps, labeled image, final result:

![alt text][image4]

#### 3. Filter out false positives over a serials of frames

I performed a similar mechanism again onto a serials of frames to obtain a more stable result.

The relative code is contained in 72 through 87 lines of `vehicle_detection.py`.


---

### Discussion

** Problems **

The bounding boxes on the detected vehicle is wobbly, and it is not tightly match to the shape of the vehicle, especially when two cars get close.

** Probable Solution **

I think I could constructe a Vehicle object for each detected car and track the bounding boxes of cars. When I am confident one car is real(detected over serveral filter cycle), I could start to extract the car's histograms of color information (especially H channel of HLV) and record it. With that I could finally more percisely draw out the bounding box of a car even two cars get close or overlap(ideally two car with different color).

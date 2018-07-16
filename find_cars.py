import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from extract_feature import *
from auto_subplot import *
from post_process import *
from hog_sampling_win_search import *
from scipy.ndimage.measurements import label 
import time

import glob
from random import shuffle

if __name__ == "__main__":
    verbose = False
    pickle_file='svc_acc_0.983400.p'
    # load a pe-trained svc model from a serialized (pickle) file
    dist_pickle = pickle.load( open(pickle_file, "rb" ) )

    # get attributes of our svc object
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    color_space = dist_pickle["color_space"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    spatial_feat = dist_pickle["spatial_feat"]
    hist_feat = dist_pickle["hist_feat"]

    if verbose:
        print("color_space : {}".format(color_space))
        print("orient : {}".format(orient))
        print("pix_per_cell : {}".format(pix_per_cell))
        print("cell_per_block : {}".format(cell_per_block))
        print("spatial_size : {}".format(spatial_size))
        print("hist_bins : {}".format(hist_bins))
        print("spatial_feat : {}".format(spatial_feat))
        print("hist_feat : {}".format(hist_feat))

    images = glob.glob('./test_images/*.jpg')
    shuffle(images)
    #images = ["test_images/test11.jpg"]
    out_img_names = []
    out_imgs = []
    out_img_camps = []
    for image_name in images[:]:
        img = mpimg.imread(image_name)
        out_img = img.copy()

        ystart = 380 # 350
        ystop = 636
        #scales = [1.0, 1.5, 2.0]
        scales = [1.0, 1.2, 1.5, 1.8]
        #scales = [1.4, 1.5]
        
        t = time.time()
        bboxes = []
        for scale in scales:
            boxes = pre_find_cars(img, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_feat, spatial_size, hist_feat, hist_bins)
            if boxes:
                bboxes.append(boxes)
        if bboxes:
            bboxes = np.vstack(bboxes)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to process img...')
        
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat, bboxes)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 3)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        
        labeled_bboxes = find_labeled_bboxes(labels)
        out_img = draw_boxes(out_img, labeled_bboxes)

        out_img_names.append(image_name)
        out_imgs.append(out_img)
        out_img_camps.append(None)
        out_img_names.append(image_name)
        out_imgs.append(heatmap)
        out_img_camps.append('hot')

    multi_subplot(out_img_names, out_imgs, 6, out_img_camps)

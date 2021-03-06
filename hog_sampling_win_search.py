import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from extract_feature import *
from auto_subplot import *
from post_process import *
import glob
from random import shuffle


# Extract features using hog sub-sampling and make predictions
''' The input img must be jpg/jpeg & RGB color space '''
def pre_find_cars(img, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_feat, spatial_size, hist_feat, hist_bins):
    img = img.astype(np.float32)
    
    img_tosearch = img[ystart:ystop,:,:]
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
            #print("extract YCrCb features")
    else: 
        ctrans_tosearch = img_tosearch
        #print("extract RGB features")
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step, whcih means 75% overlap.
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            features = []
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            features.append(hog_features)
            #features.append(hog_feat1)
            #print("hog_features shape: {}".format(hog_feat1.shape))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                features.append(spatial_features)
                #print("spatial_features shape: {}".format(spatial_features.shape))
            if hist_feat == True:
                hist_features = color_hist(subimg, nbins=hist_bins)
                features.append(hist_features)
                #print("hist_features shape: {}".format(hist_features.shape))

            # concatenate all three type of features
            features = np.concatenate(features).astype(np.float64)
            #print("features shape: {}".format(features.shape))
            test_features = X_scaler.transform([features])
            #print("test_features shape: {}".format(test_features.shape))            
            prediction = svc.predict(test_features)
            
            if prediction == 1:
                # scale back to normal size
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # ((x1,y1),(x2,y2))
                box = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                boxes.append(box)
                
    return boxes

if __name__ == "__main__":
    verbose = False
    pickle_file='svc_acc_0.994400.p'
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
    out_img_names = []
    out_imgs = []
    out_img_camps = []
    for image_name in images[:4]:
        img = mpimg.imread(image_name)
        out_img = img.copy()

        ystart = 386
        ystop = 642
        scales = [1.5]#[1.4, 1.5]
        
        bboxes = []
        for scale in scales:
            boxes = pre_find_cars(img, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_feat, spatial_size, hist_feat, hist_bins)
            if boxes:
                bboxes.append(boxes)
        
        if bboxes:
            bboxes = np.vstack(bboxes)

        out_img = draw_boxes(out_img, bboxes)
        
        #cv2.rectangle(out_img,(0, ystart),(out_img.shape[1],ystop),(0,255,0),6)
        #cv2.rectangle(out_img,(0, 0),(int(64*scale),int(64*scale)),(0,255,0),6)

        out_img_names.append(image_name)
        out_imgs.append(out_img)
        out_img_camps.append(None)

    multi_subplot(out_img_names, out_imgs, 2, out_img_camps)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time
from scipy.ndimage.measurements import label 

from extract_feature import *
from auto_subplot import *
from post_process import *
from hog_sampling_win_search import *

    
verbose = False
pickle_file='svc_acc_0.994400.p'
# load the pe-trained svc model
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

# how many frames were processed
frame_cnt = 0
# how many frames we take it as a car/non-car determine iteration
n_frame_mask = 5
# positives we found over last n_frame_mask frame
bboxes_over_frame = []
# high confidence positives
valid_bboxes = []

def FIND_CARS_PIPELINE(img):
    global frame_cnt
    global bboxes_over_frame
    global valid_bboxes
    frame_cnt += 1
    out_img = img.copy()
    ystart = 386
    ystop = 642
    scales = [1.4, 1.5]

    ''' Collect positives from different scale window search.  '''
    bboxes = []
    for scale in scales:
        boxes = pre_find_cars(img, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_feat, spatial_size, hist_feat, hist_bins)
        if boxes:
            bboxes.append(boxes)
    if bboxes:
        bboxes = np.vstack(bboxes)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, bboxes)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    bboxes = find_labeled_bboxes(labels)

    ''' filter out false positives over frames '''
    if bboxes:
        bboxes_over_frame.append(bboxes)
    # each n_frame_mask frames an iteration
    if 0 == frame_cnt % n_frame_mask:
        print("frame_cnt: {}".format(frame_cnt))
        if bboxes_over_frame:
            bboxes_over_frame = np.vstack(bboxes_over_frame)
            print("bboxes_over_frame: {}".format(bboxes_over_frame))
            heat = np.zeros_like(img[:,:,0]).astype(np.float)
            heat = add_heat(heat, bboxes_over_frame)
            # Apply threshold to help remove false positives
            heat = apply_threshold(heat, int(n_frame_mask*0.6))
            heatmap = np.clip(heat, 0, 255)
            labels = label(heatmap)
            valid_bboxes = find_labeled_bboxes(labels)
            # filter out false positives those who is with unreasonable appearence size
            valid_bboxes = filter_bbox_by_size(valid_bboxes)
            print("valid_bboxes: {}".format(valid_bboxes))
        bboxes_over_frame = []
        
    out_img = draw_boxes(out_img, valid_bboxes)
    return out_img

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # you should return the final output (image where lines are drawn on lanes)
    result = FIND_CARS_PIPELINE(image)
    return result

prj_output = 'output_videos/Vehicle_detection.mp4'

if __name__ == "__main__":
    if verbose:
        print("color_space : {}".format(color_space))
        print("orient : {}".format(orient))
        print("pix_per_cell : {}".format(pix_per_cell))
        print("cell_per_block : {}".format(cell_per_block))
        print("spatial_size : {}".format(spatial_size))
        print("hist_bins : {}".format(hist_bins))
        print("spatial_feat : {}".format(spatial_feat))
        print("hist_feat : {}".format(hist_feat))

    #clip_v = VideoFileClip("test_video.mp4")
    clip_v = VideoFileClip("project_video.mp4")#.subclip(14,16)#.subclip(27,32)#.subclip(14,16)#.subclip(27,30)#.subclip(12,20)#.subclip(14,16)#
    clip = clip_v.fl_image(process_image)
    t=time.time()    
    clip.write_videofile(prj_output, audio=False)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to process video...')
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math

# cmaps shall contains number of cols elements.
def multi_subplot(names, images, cols, cmaps):
    cnt = len(images)
    rows = math.ceil(cnt / cols)
    fig = plt.figure()
    for indx in range(cnt):
        plt.subplot(rows, cols, indx+1)
        plt.imshow(images[indx], cmap=cmaps[indx])
        plt.title(names[indx], fontsize=8)
        plt.tick_params(labelsize=4)
    #fig.tight_layout()
    #fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.3)
    plt.show()
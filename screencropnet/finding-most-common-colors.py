#!/usr/bin/env python

# SOURCE: https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL

def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()


img = cv2.imread("autocropped/cropped-ObjLocModelV1-2021-10-20_12-44-46_000.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_2 = cv2.imread("autocropped/cropped-ObjLocModelV1-2021-10-26_00-16-16_000.png")
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

dim = (500, 300)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img_2 = cv2.resize(img_2, dim, interpolation = cv2.INTER_AREA)

# show_img_compar(img, img_2)


img_temp = img.copy()
img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = np.average(img, axis=(0,1))

img_temp_2 = img_2.copy()
img_temp_2[:,:,0], img_temp_2[:,:,1], img_temp_2[:,:,2] = np.average(img_2, axis=(0,1))

show_img_compar(img, img_temp)
show_img_compar(img_2, img_temp_2)

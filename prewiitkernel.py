import cv2
from skimage import io, color

import matplotlib.pyplot as plt
import numpy as np

#input image
img1=cv2.imread('House1.jpg', cv2.CV_8UC1)
img1 = color.rgb2gray(img1)

img2=cv2.imread('House2.jpg', cv2.CV_8UC1)
img2 = color.rgb2gray(img2)


def convolve2d(image, kernel,kernel_height,kernel_width):
    output = np.zeros_like(image)
    for x in range(image.shape[1]-kernel_width+1):
        for y in range(image.shape[0]-kernel_height+1):
            output[y,x]=(kernel*image[y:y+kernel_height,x:x+kernel_width]).sum()/(kernel_width*kernel_height)
    return output


# Prewiit edge operator

# Prewiit Edge Operator ----> x <----
kernel = np.array([[+1,0,-1],[+1,0,-1],[1,0,-1]])
result1 = convolve2d(img1,kernel,3,3)
result2 = convolve2d(img2,kernel,3,3)

cv2.imwrite('Output image/Prewiit_edge_X_house1.jpg',result1)
cv2.imwrite('Output image/Prewiit_edge_X_house2.jpg',result2)


# Prewiit Edge Operator --->  y  <----
kernel = np.array([[+1,+1,+1],[0,0,0],[-1,-1,11]])
result1 = convolve2d(img1,kernel,3,3)
result2 = convolve2d(img2,kernel,3,3)

cv2.imwrite('Output image/Prewiit_edge_Y_house1.jpg',result1)
cv2.imwrite('Output image/Prewiit_edge_Y_house2.jpg',result2)

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


def gussiankern(sigma):
    x, y = np.meshgrid(np.linspace(-1,1,2*sigma+1), np.linspace(-1,1,2*sigma+1))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( d**2 / ( 2.0 * sigma**2 ) ) )
    return g

#gaussian kernel

sigma = 1
kernel = gussiankern(sigma)
result1 = convolve2d(img1,kernel,2*sigma+1,2*sigma+1)
result2 = convolve2d(img2,kernel,2*sigma+1,2*sigma+1)

cv2.imwrite('Output image/gussiankernelHouse1.jpg',result1)
cv2.imwrite('Output image/gussiankernelHouse2.jpg',result2)


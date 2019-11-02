import cv2
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
#input image from directory
img1=cv2.imread('NotreDame1.jpg', cv2.CV_8UC1)
img1 = rgb2gray(img1)

img2=cv2.imread('House2.jpg', cv2.CV_8UC1)
img2 = color.rgb2gray(img2)


def convolve2d(image, kernel,kernel_height,kernel_width):
    output = np.zeros_like(image)
    for x in range(image.shape[1]-kernel_width+1):
        for y in range(image.shape[0]-kernel_height+1):
            output[y,x]=(kernel*image[y:y+kernel_height,x:x+kernel_width]).sum()/(kernel_width*kernel_height)
    return output

# Average kernel 3by3

kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
result1 = convolve2d(img1, kernel, 3, 3)
result2 = convolve2d(img2, kernel, 3, 3)

cv2.imwrite('Output image/AveragekernelThreeByThreeHouse1.jpg',result1)
cv2.imwrite('Output image/AveragekernelThreeByThreeHouse2.jpg',result2)

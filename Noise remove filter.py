import cv2
from skimage import io, color
import numpy as np

#input image
img1=cv2.imread('Noisyimage1.jpg', cv2.CV_8UC1)
img1 = color.rgb2gray(img1)

img2=cv2.imread('Noisyimage2.jpg', cv2.CV_8UC1)
img2 = color.rgb2gray(img2)



def convolve2d(image, kernel,kernel_height,kernel_width):
    output = np.zeros_like(image)
    for x in range(image.shape[1]-kernel_width+1):
        for y in range(image.shape[0]-kernel_height+1):
            output[y,x]=(kernel*image[y:y+kernel_height,x:x+kernel_width]).sum()/(kernel_width*kernel_height)
    return output


def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final




#average jernel 5 by 5

kernel = np.array([[1,1,0,0,1],[1,1,1,1,0],[0,0,0,0,0],[0,1,1,1,1],[1,0,1,1,0]])
result1 = convolve2d(img1,kernel,5,5)
result2 = convolve2d(img2,kernel,5,5)

cv2.imwrite('Output image/noiseRemove1Byaveragefilter.jpg',result1)
cv2.imwrite('Output image/noiseRemove2Byaveragefilter.jpg',result2)

result1 = median_filter(img1,3)
result2 = median_filter(img2,3)

cv2.imwrite('Output image/noiseremoveByMedian1.jpg',result1)
cv2.imwrite('Output image/noiseremoveByMedian2.jpg',result2)



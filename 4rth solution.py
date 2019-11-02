import cv2

image_One = cv2.imread('walk_1.jpg')
image_two = cv2.imread('walk_2.jpg')

gray_image_one = cv2.cvtColor(image_One,cv2.COLOR_RGB2GRAY)
gray_image_two = cv2.cvtColor(image_two,cv2.COLOR_RGB2GRAY)

subtract_image = gray_image_one.copy()

for i in range(subtract_image.shape[0]):
    for j in range(subtract_image.shape[1]):
        subtract_image.itemset((i,j),abs(int(gray_image_one[i][j]) - int(gray_image_two[i][j])))


cv2.imwrite('output image/subtract_image.jpg',subtract_image)


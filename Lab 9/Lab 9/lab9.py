import cv2
import numpy as np
from matplotlib import pyplot as plt

nrows = 2
ncols = 4

img = cv2.imread('japan.jpg',)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# First Image Read IN to Column 1
plt.subplot(nrows, ncols, 1), plt.imshow(
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Second Image <GrayScale> Read into Column 2
plt.subplot(nrows, ncols, 2), plt.imshow(gray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

# Third Image <Blur 5x5> Read Into Column 3
imgOut = cv2.GaussianBlur(gray, (5, 5), 0)
plt.subplot(nrows, ncols, 3), plt.imshow(imgOut, cmap='gray')
plt.title('5x5'), plt.xticks([]), plt.yticks([])

# Fourth Image <Blur 13x13> Read Into Column 4
imgOut2 = cv2.GaussianBlur(gray, (13, 13), 0)
plt.subplot(nrows, ncols, 4), plt.imshow(imgOut2, cmap='gray')
plt.title('13x13'), plt.xticks([]), plt.yticks([])

# Fifth Image <Verticle Sobel> Read Into Column 5
sobelVertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # y dir
plt.subplot(nrows, ncols, 5), plt.imshow(sobelVertical, cmap='gray')
plt.title('Verticle Sobel'), plt.xticks([]), plt.yticks([])

# Sixth Image <Horizontal Sobel> Read Into Column 6
sobelHorizontal = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # x dir
plt.subplot(nrows, ncols, 6), plt.imshow(sobelHorizontal, cmap='gray')
plt.title('Horizontal Sobel'), plt.xticks([]), plt.yticks([])

# Seventh Image <Mixed Sobel> Read Into Column 7
sobelMixed = sobelHorizontal + sobelVertical
plt.subplot(nrows, ncols, 7), plt.imshow(sobelMixed, cmap='gray')
plt.title('Mixed Sobel'), plt.xticks([]), plt.yticks([])

# Canny Edge Detection Image Column 8
canny = cv2.Canny(gray, 300, 600)
plt.subplot(nrows, ncols, 8), plt.imshow(canny, cmap='gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])


plt.show()

cv2.imshow('Canny Image',canny)
cv2.waitKey(0);



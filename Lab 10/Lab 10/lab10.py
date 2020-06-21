import cv2
import numpy as np
from matplotlib import pyplot as plt

nrows = 1
ncols = 5

img = cv2.imread('hedge.jpg',)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# First Image Read IN to Column 1
plt.subplot(nrows, ncols, 1), plt.imshow(
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Second Image <GrayScale> Read into Column 2
plt.subplot(nrows, ncols, 2), plt.imshow(gray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

imgHarris = img.copy()
dst = cv2.cornerHarris(gray,2,3,0.04)

threshold = 0.3; #number between 0 and 1
for i in range(len(dst)):
 for j in range(len(dst[i])):
    if dst[i][j] > (threshold*dst.max()):
        cv2.circle(imgHarris,(j,i),3,(220, 22, 189),-1)
        


# Third Image <Image Harris> Read into Column 2
plt.subplot(nrows, ncols, 3), plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Image Harris'), plt.xticks([]), plt.yticks([])


imgShiTomasi = img.copy()
corners = cv2.goodFeaturesToTrack(gray,100,0.01,10)
for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),3,(0, 215, 255),-1)
    

# Fourth Image <ShiTomasi> Read into Column 2
plt.subplot(nrows, ncols, 4), plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Shi Tomasi GFFT'), plt.xticks([]), plt.yticks([])


#Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create(50)
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
#Draw keypoints
imgSift = cv2.drawKeypoints(img,kps,outImage=None,color=(200,80,40),flags=4)


# Fifth Image <Sift> 
plt.subplot(nrows, ncols, 5), plt.imshow(cv2.cvtColor(imgSift, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Sift'), plt.xticks([]), plt.yticks([])

plt.show()
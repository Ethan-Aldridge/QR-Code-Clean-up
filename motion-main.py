import cv2
import numpy as np
import sys

image = cv2.imread('qr-codes/JpbZGw_blur.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

edges = cv2.Canny(otsu_thresh, 0, 25)

# Calculate image prior
img_x1 = 255555555
img_x2 = -1
for i in range(int(gray_image.shape[0])):
    for j in range(int(gray_image.shape[1])):
        if gray_image[i, j] != 0:
            img_x1 = min(img_x1, j)
            img_x2 = max(img_x2, j)

d = (img_x1 + img_x2) / 2 - img_x1

# Calculate center of clear image
x1 = 255555555
x2 = -1
for i in range(int(edges.shape[0])):
    for j in range(int(edges.shape[1])):
        if edges[i, j] != 0:
            x1 = min(x1, j)
            x2 = max(x2, j)
    

x0 = x2 - x1
xc = x0 - d

# Calculated clear image center xc
print(f"Calculated clear image center xc: {xc}")

resized_img = cv2.resize(image, (640, 640))
resized_edges = cv2.resize(edges, (640, 640))
resized_otsu = cv2.resize(otsu_thresh, (640, 640))

# Calulate the two-step gradient using Sobel operator
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude of the gradient
magnitude = np.sqrt(sobelx**2 + sobely**2)

print(f"Sobel Gradient Magnitude - min: {np.min(magnitude)}, max: {np.max(magnitude)}")

# Find the position of the maximum gradient value
maxGradient = np.max(magnitude)
maxGradient_pos = (-1, -1)
for i in range(magnitude.shape[0]):
    for j in range(magnitude.shape[1]):
        if magnitude[i, j] == maxGradient:
            maxGradient_pos = (i, j)
            break

    if maxGradient_pos != (-1, -1):
        break

print(f"Max Gradient Position: {maxGradient_pos}")

# Calculate defocus radius R
R = xc - maxGradient_pos[1]
print(f"Calculated defocus radius R: {R}")

cv2.imshow('Original Image', resized_img)
cv2.imshow('Canny Edges', resized_edges)
cv2.imshow('Otsu Thresholding', resized_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import wiener_restore as wr

def calcPSF(shape, R):
    # Calculate a circular PSF (Point Spread Function) given the shape and radius R
    psf = np.zeros(shape, dtype=np.float32)
    center = (shape[1] // 2, shape[0] // 2)
    cv2.circle(psf, center, R, 1, thickness=-1)
    psf /= psf.sum()  # Normalize the PSF

    return psf

gray_image = cv2.imread('qr_codes/total1.jpg', cv2.IMREAD_GRAYSCALE)
gray_image = cv2.resize(gray_image, (300, 300))

ret, binarized = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

edges = cv2.Canny(gray_image, 0, 75, L2gradient=True)

# Calculate image edge of image
x1 = 255555555
x2 = -1
edges_x1 = 255555555
edges_y1 = -1
edges_x2 = -1
for i in range(int(binarized.shape[0]) // 2):
    for j in range(int(binarized.shape[1])):
        if binarized[i, j] == 0:
            x1 = min(x1, j)
            x2 = max(x2, j)

        if edges[i, j] != 0:
            edges_x1 = min(x1, j)
            edges_x2 = max(x2, j)
            edges_y1 = min(edges_y1, i)

if x1 == 255555555 or x2 == -1:
    raise ValueError("No black pixels found in the binarized image.")
if edges_x1 == 255555555 or edges_x2 == -1:
    raise ValueError("No edge pixels found in the edge image.")

# Calulate the two-step gradient using the Laplacian operator on the edge image
laplacian = np.abs(cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3))

# Visualize the laplacian for debugging
cv2.imshow('Laplacian', cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

# Find the column of the leftmost QR code edge in the binarized image (xc)
row_idx = gray_image.shape[0] // 2  # Use center row for robustness
row_bin = binarized[row_idx, :]
xc = np.argmax(row_bin == 0)  # First black pixel in the center row

# Find the column xQ where the Laplacian (two-step gradient) is maximized
row_lap = laplacian[row_idx, :]
xQ = np.argmax(row_lap)

print(f"Calculated QR code edge column xc: {xc}")
print(f"Calculated maximum gradient column position xQ: {xQ}")

# Calculate defocus radius R
R = min(abs(x2 - xQ), abs(x1 - xQ))
print(f"Calculated defocus radius R: {R}")

psf = calcPSF((gray_image.shape[0], gray_image.shape[1]), 10)

restored_img = wr.wiener_restore(gray_image, psf)

ret, restored_img_binarzied = cv2.threshold(restored_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

resize_factor = 300
resized_img = cv2.resize(gray_image, (resize_factor, resize_factor))
resized_edges = cv2.resize(edges, (resize_factor, resize_factor))
resized_otsu = cv2.resize(binarized, (resize_factor, resize_factor))
resized_restored_img = cv2.resize(restored_img, (resize_factor, resize_factor))
resized_restored_img_binarized = cv2.resize(restored_img_binarzied, (resize_factor, resize_factor))

cv2.imshow('Original Image', resized_img)
cv2.imshow('Canny Edges', resized_edges)
cv2.imshow('Otsu Thresholding', resized_otsu)
cv2.imshow('PSF', cv2.resize(psf / psf.max(), (resize_factor, resize_factor)))
cv2.imshow('Restored Image', resized_restored_img)
cv2.imshow('Restored Binarized Image', resized_restored_img_binarized)
cv2.waitKey(0)
cv2.destroyAllWindows()
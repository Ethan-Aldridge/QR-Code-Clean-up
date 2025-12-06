import cv2
import numpy as np

def calcPSF(shape, R):
    # Calculate a circular PSF (Point Spread Function) given the shape and radius R
    psf = np.zeros(shape, dtype=np.float32)
    center = (shape[1] // 2, shape[0] // 2)
    cv2.circle(psf, center, R, 255, thickness=-1)
    psf /= psf.sum()  # Normalize the PSF

    return psf

def quad_rearrange(img):
    # Rearrange the quadrants of the image
    h, w = img.shape
    cx, cy = w // 2, h // 2

    q0 = img[0:cy, 0:cx]      # Top-Left
    q1 = img[0:cy, cx:w]      # Top-Right
    q2 = img[cy:h, 0:cx]      # Bottom-Left
    q3 = img[cy:h, cx:w]      # Bottom-Right

    top = np.hstack((q3, q2))
    bottom = np.hstack((q1, q0))
    rearranged = np.vstack((top, bottom))

    return rearranged

image = cv2.imread('qr-codes/JpbZGw_300_blur5.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, binarized = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

edges = cv2.Canny(gray_image, 0, 10)

# Calculate image prior and edge of clear image
x1 = 255555555
x2 = -1
for i in range(int(binarized.shape[0]) // 2):
    for j in range(int(binarized.shape[1])):
        if binarized[i, j] == 0:
            x1 = min(x1, j)
            x2 = max(x2, j)

print(f"Binarized image x1: {x1}, x2: {x2}")
    
x0 = (x2 + x1) // 2 # Center of the edge image
xc = x0 # Edge of the clear image
print(f"Calculated clear image edge xc: {xc}")

# Calulate the two-step gradient using the Laplacian operator on the edge image
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)

# Find the maximum gradient column position
column_sums = np.sum(laplacian, axis=0)
xQ = np.argmax(column_sums)
print(f"Calculated maximum gradient column position xQ: {xQ} and value: {column_sums[xQ]}")

# Calculate defocus radius R and resulting PSF
R = abs(x1 - xQ)
print(f"Calculated defocus radius R: {R}")
psf = calcPSF((gray_image.shape[0], gray_image.shape[1]), R)
psf = quad_rearrange(psf)

# Compute the FFT of the PSF and the image
h_psf = np.fft.fftshift(np.fft.fft2(psf))
h_img = np.fft.fftshift(np.fft.fft2(gray_image))
# h_img = cv2.dft(gray_image.astype(np.float32))

# Apply Wiener restoration deconvolution
snr = abs(np.mean(h_img) / np.std(h_img)) # Signal to Noise Ratio for Wiener filter
print(f"Calculated SNR: {snr}")
h_wiener = h_psf / (np.abs(h_psf)**2 + 1/snr)
restored_img = np.fft.ifft2(np.fft.ifftshift(h_wiener * h_img)).real
# ret, restored_img_binarzied = cv2.threshold(restored_img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Restored image shape: {restored_img.shape}")

resize_factor = 300
resized_img = cv2.resize(image, (resize_factor, resize_factor))
resized_edges = cv2.resize(edges, (resize_factor, resize_factor))
resized_otsu = cv2.resize(binarized, (resize_factor, resize_factor))
resized_restored_img = cv2.resize(restored_img, (resize_factor, resize_factor))

cv2.imshow('Original Image', resized_img)
cv2.imshow('Canny Edges', resized_edges)
cv2.imshow('Otsu Thresholding', resized_otsu)
cv2.imshow('PSF', cv2.resize(psf / psf.max(), (resize_factor, resize_factor)))
cv2.imshow('Restored Image', resized_restored_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
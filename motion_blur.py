import cv2
import numpy as np

def calcPSF(shape, len, theta):
    # Calculate a motion blur PSF (Point Spread Function) given the shape, length, and angle theta
    psf = np.zeros(shape, dtype=np.float32)
    center = (shape[1] // 2, shape[0] // 2)
    angle = theta * np.pi / 180.0
    half_len = len / 2

    x1 = int(center[0] - half_len * np.cos(angle))
    y1 = int(center[1] - half_len * np.sin(angle))
    x2 = int(center[0] + half_len * np.cos(angle))
    y2 = int(center[1] + half_len * np.sin(angle))

    cv2.line(psf, (x1, y1), (x2, y2), 1, thickness=1)
    psf /= psf.sum()  # Normalize the PSF

    return psf

def applyBlur(image, psf):
    # Apply blur to the image using the given PSF
    h_img = np.fft.fft2(image)
    h_psf = np.fft.fft2(np.fft.ifftshift(psf))
    blurred_img = np.fft.ifft2(h_img * h_psf).real
    return blurred_img

image = cv2.imread('qr_codes/JpbZGw_300.jpg', cv2.IMREAD_GRAYSCALE)
len = 5  # Example motion blur length
theta = 10  # Example motion blur angle in degrees
psf = calcPSF(image.shape, len, theta)
blurred_image = applyBlur(image, psf)
# Normalize the blurred image to 0-255
blurred_image = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX)

# Save image
cv2.imwrite(f'qr_codes/JpbZGw_300_blur{len}-{theta}_generated.png', blurred_image.astype(np.uint8))

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
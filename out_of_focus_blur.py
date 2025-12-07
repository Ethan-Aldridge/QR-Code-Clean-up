import cv2
import numpy as np

def calcPSF(shape, R):
    # Calculate a circular PSF (Point Spread Function) given the shape and radius R
    psf = np.zeros(shape, dtype=np.float32)
    center = (shape[1] // 2, shape[0] // 2)
    cv2.circle(psf, center, R, 1, thickness=-1)
    psf /= psf.sum()  # Normalize the PSF

    return psf

def applyBlur(image, psf):
    # Apply blur to the image using the given PSF
    h_img = np.fft.fft2(image)
    h_psf = np.fft.fft2(np.fft.ifftshift(psf))
    blurred_img = np.fft.ifft2(h_img * h_psf).real
    return blurred_img

image = cv2.imread('qr_codes/JpbZGw_300.jpg', cv2.IMREAD_GRAYSCALE)
R = 20  # Example defocus radius
psf = calcPSF(image.shape, R)
blurred_image = applyBlur(image, psf)
# Normalize the blurred image to 0-255
blurred_image = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX)

# Save image
cv2.imwrite(f'qr_codes/JpbZGw_300_blur{R}_generated.png', blurred_image.astype(np.uint8))

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
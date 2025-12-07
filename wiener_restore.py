import cv2
import numpy as np

def wiener_restore(image, psf):
    # Compute the FFT of the PSF and the image
    h_psf = np.fft.fft2(np.fft.ifftshift(psf))
    h_img = np.fft.fft2(image.astype(np.float32))

    # Compute Wiener filter
    eps = 1e-6  # To avoid division by zero
    snr = abs(np.mean(h_img) / np.std(h_img)) # Signal to Noise Ratio for Wiener filter
    h_wiener = np.conj(h_psf) / (np.abs(h_psf)**2 + snr + eps)

    # Apply Wiener restoration deconvolution
    restored_img = np.fft.ifft2(h_wiener * h_img).real
    restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)

    return restored_img
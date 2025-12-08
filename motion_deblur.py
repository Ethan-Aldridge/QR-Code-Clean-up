import cv2
import numpy as np
import wiener_restore as wr

def calcPSF(shape, len, theta):
    # Calculate a motion blur PSF (Point Spread Function) given the shape, length, and angle theta
    psf = np.zeros(shape, dtype=np.float32)
    center = (shape[1] // 2, shape[0] // 2)
    theta_rad = np.deg2rad(theta)
    x_offset = int((len / 2) * np.cos(theta_rad))
    y_offset = int((len / 2) * np.sin(theta_rad))
    cv2.line(psf, (center[0] - x_offset, center[1] - y_offset), 
                  (center[0] + x_offset, center[1] + y_offset), 1, thickness=1)
    psf /= psf.sum()  # Normalize the PSF

    return psf

gray_image = cv2.imread('qr_codes/total3.jpg', cv2.IMREAD_GRAYSCALE)
gray_image = cv2.resize(gray_image, (300, 300))

ret, binarized = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(binarized, 0, 75, L2gradient=True)

# Calculate corners of the QR code from the edges image
# (x1, y1) - top-left; (x2, y2) - top-right; (x3, y3) - bottom-left; a,b - additional points due to edge splitting
x = [255555555, 255555555, -1, -1]  # x1a, x1b, x2a, x2b
y = [255555555, 255555555, 255555555, 255555555, -1]  # y1a, y1b, y2a, y2b, y3
for i in range(int(edges.shape[0]) // 10):
    for j in range(int(edges.shape[1])):
        if edges[i, j] != 0:
            if j < edges.shape[1] // 10:
                if j < x[0]:
                    x[0] = j
                    y[0] = i
                elif i <= y[1] and j < x[1]:
                    x[1] = j
                    y[1] = i
            elif j > 9 * edges.shape[1] // 10:
                if j > x[2]:
                    x[2] = j
                    y[2] = i
                elif i <= y[3] and j > x[3]:
                    x[3] = j
                    y[3] = i
            
            y[4] = max(y[4], i)

if any(val == 255555555 or val == -1 for val in x) or any(val == 255555555 or val == -1 for val in y):
    raise ValueError("Could not find all corners of the QR code.")

print(f"Detected corners: Top-left-a ({x[0]}, {y[0]}), Top-left-b ({x[1]}, {y[1]}), Top-right-a ({x[2]}, {y[2]}), Top-right-b ({x[3]}, {y[3]})")

# Calculate the center of the image
w = (x[1] - x[0]) // 2
h = (y[4] - y[0]) // 2
center = (w, h)

# Define the rotation angle and scale
angle = np.rad2deg(np.arctan2((y[3] - y[1]), (x[3] - x[1])))  # Angle based on top edge
scale = 1.0  # No scaling

# Get the 2D rotation matrix
M = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the affine transformation to rotate the image
rotated_edges = cv2.warpAffine(edges, M, (edges.shape[0], edges.shape[1]))
cv2.imshow('Rotated Edges', rotated_edges)

# Redefine QR code corners after rotation
x = [255555555, 255555555, -1, -1]  # x1a, x1b, x2a, x2b
y = [255555555, 255555555, 255555555, 255555555, -1]  # y1a, y1b, y2a, y2b, y3
for i in range(int(rotated_edges.shape[0]) // 10):
    for j in range(int(rotated_edges.shape[1])):
        if rotated_edges[i, j] != 0:
            if j < rotated_edges.shape[1] // 10:
                if j < x[0]:
                    x[0] = j
                    y[0] = i
                elif i <= y[1] and j < x[1]:
                    x[1] = j
                    y[1] = i
            elif j > 9 * rotated_edges.shape[1] // 10:
                if j > x[2]:
                    x[2] = j
                    y[2] = i
                elif i <= y[3] and j > x[3]:
                    x[3] = j
                    y[3] = i
            
            y[4] = max(y[4], i)

if any(val == 255555555 or val == -1 for val in x) or any(val == 255555555 or val == -1 for val in y):
    raise ValueError("Could not find all corners of the QR code.")

print(f"Detected corners post-rotation: Top-left-a ({x[0]}, {y[0]}), Top-left-b ({x[1]}, {y[1]}), Top-right-a ({x[2]}, {y[2]}), Top-right-b ({x[3]}, {y[3]})")

# Estimate motion blur parameters
err = 1.2  # Error adjustment factor
length_est = np.ceil(np.max([np.hypot(x[1] - x[0], y[1] - y[0]), np.hypot(x[3] - x[2], y[3] - y[2])])) * err
mode = np.argmax([np.hypot(x[1] - x[0], y[1] - y[0]), np.hypot(x[3] - x[2], y[3] - y[2])])
if mode == 0: # top-left edge used for angle estimation
    print(f"Mode 0 selected for angle estimation.")
    theta_est = -1 * np.rad2deg(np.arctan2(abs(y[1] - y[0]), max(abs(x[1] - x[0]), 1))) * err
else: # top-right edge used for angle estimation
    print(f"Mode 1 selected for angle estimation.")
    theta_est = np.rad2deg(np.arctan2(abs(y[3] - y[2]), max(abs(x[3] - x[2]), 1))) * err
print(f"Estimated motion blur length: {length_est}, angle: {theta_est}")

psf = calcPSF((gray_image.shape[0], gray_image.shape[1]), length_est, theta_est)
cv2.imshow('Estimated PSF', cv2.normalize(psf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

h_psf = np.fft.fft2(np.fft.ifftshift(psf))
cv2.imshow('PSF FFT Magnitude', cv2.normalize(np.log(np.abs(h_psf) + 1), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

restored_img = wr.wiener_restore(gray_image, psf)
ret, restored_img_binarzied = cv2.threshold(restored_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Resize images for better visualization
scale_factor = 1
gray_image = cv2.resize(gray_image, (0, 0), fx=scale_factor, fy=scale_factor)
binarized = cv2.resize(binarized, (0, 0), fx=scale_factor, fy=scale_factor)
edges = cv2.resize(edges, (0, 0), fx=scale_factor, fy=scale_factor)
restored_img = cv2.resize(restored_img, (0, 0), fx=scale_factor, fy=scale_factor)
restored_img_binarzied = cv2.resize(restored_img_binarzied, (0, 0), fx=scale_factor, fy=scale_factor)

# Display results
cv2.imshow('Blurred Image', gray_image)
cv2.imshow('Binarized Image', binarized)
cv2.imshow('Edges', edges)
cv2.imshow('Restored Image', restored_img)
cv2.imshow('Restored Binarized Image', restored_img_binarzied)
cv2.waitKey(0)
cv2.destroyAllWindows()
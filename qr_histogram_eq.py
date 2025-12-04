import cv2 as cv
import numpy as np

# image strings for easier editing
img_name = "qr_photo1"
file_ext = ".jpg"
img_path = "qr_codes\\" + img_name
out_path = "qr_out\\" + img_name

img = cv.imread(f"{img_path}{file_ext}", cv.IMREAD_GRAYSCALE)
# error handling
assert img is not None, "file could not be read, check with os.path.exists()"

# contrast adjustments via histogram equalization 
equ = cv.equalizeHist(img)
#stacking images side-by-side to compare before and after
res = np.hstack((img,equ)) 
cv.imwrite(f"{out_path}_he{file_ext}",res)

# more advanced histogram equalization, CLAHE:
# Contrast Limited Adaptive Histogram Equalization
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))
cl1 = clahe.apply(img)
res = np.hstack((img,cl1))
cv.imwrite(f"{out_path}_clahe4{file_ext}",res)
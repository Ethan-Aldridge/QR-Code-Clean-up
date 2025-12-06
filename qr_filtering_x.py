import cv2 as cv
import numpy as np

# image strings for easier editing
img_name = "qr_photo1_clahe"
file_ext = ".jpg"
img_path = "qr_codes\\" + img_name
out_path = "qr_out\\" + img_name

# the image object to be modified
img = cv.imread(f"{img_path}{file_ext}", cv.IMREAD_GRAYSCALE)
# error handling
assert img is not None, "file could not be read, check with os.path.exists()"

# do gaussian blurs
out_img = cv.GaussianBlur(img, (0,0), 0.7)
out_img2 = cv.GaussianBlur(img, (0,0), 2.5)
# difference of gaussian
diff_img = out_img2 - out_img
# write images
cv.imwrite(f"{out_path}gaussian_l.jpg",out_img)
cv.imwrite(f"{out_path}gaussian_h.jpg",out_img2)
cv.imwrite(f"{out_path}gaussian_d.jpg",diff_img)

# from https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# Otsu's thresholding after Gaussian filtering (binarization)
blur = cv.GaussianBlur(img,(5,5),0)
ret3,otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# cv.imwrite(f"{out_path}_otsu_gauss.jpg",otsu)

adp_gauss = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,int(2*np.round(img.shape[0]/80)+3),2) # 83 = height/50->rounded to odd+2? for complex image
cv.imwrite(f"{out_path}_adp_gauss.jpg",adp_gauss)

kernel = np.ones((3,3),np.uint8)
morphology_ver = cv.morphologyEx(adp_gauss, cv.MORPH_CLOSE, kernel)
morphology_ver = cv.morphologyEx(morphology_ver, cv.MORPH_OPEN, kernel)
cv.imwrite(f"{out_path}_morphology.jpg",morphology_ver)

# bilateral = cv.bilateralFilter(img, 7, 75, 75)
# cv.imwrite(f"{out_path}_bilateral.jpg", bilateral)
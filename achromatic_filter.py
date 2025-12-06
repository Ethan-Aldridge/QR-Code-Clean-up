import cv2 as cv
import numpy as np

# image strings for easier editing
img_name = "qr_photo1.jpg"
img_path = f"qr_codes\\{img_name}"
out_img_name = f"achro_{img_name}"

# open original image
col_img = cv.imread(img_path)
# make it smaller to make computing faster (takes forever with 4k image)
col_img = cv.resize(col_img, (int(col_img.shape[1]/4), int(col_img.shape[0]/4)))
# reduce noise with blur
col_img = cv.GaussianBlur(col_img, (5,5), 0)
# create an empty results array
res = np.ndarray(col_img.shape)

# closeness in value, k
k = 20

# https://stackoverflow.com/questions/52278546/how-can-i-iterate-through-numpy-3d-array
# set pixels that are too colorful to white, retain pixel otherwise
# should erase the rest of the image basically
for ij in np.ndindex(col_img.shape[:2]):
    # print(f"{col_img[ij][0]} - {col_img[ij][1]} = {int(col_img[ij][0]) - int(col_img[ij][1])}")
    res[ij] = col_img[ij] if ((np.abs(int(col_img[ij][0]) - int(col_img[ij][1])) <= k) &
        (np.abs(int(col_img[ij][2]) - int(col_img[ij][1])) <= k) &
        (np.abs(int(col_img[ij][0]) - int(col_img[ij][2])) <= k)) else [255,255,255]

# create result image    
cv.imwrite(out_img_name,res)

out_img = cv.imread(out_img_name, cv.IMREAD_GRAYSCALE)
ret3,th3 = cv.threshold(out_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

cv.imwrite(f"{out_img_name.replace(".jpg", "")}_otsu_gauss.jpg",th3)

def achro_filter(img):
    # make it smaller to make computing faster (takes forever with 4k image)
    col_img = cv.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
    # reduce noise with blur
    col_img = cv.GaussianBlur(col_img, (5,5), 0)
    # create an empty results array
    res = np.ndarray(col_img.shape)

    # closeness in value, k
    k = 20

    # https://stackoverflow.com/questions/52278546/how-can-i-iterate-through-numpy-3d-array
    # set pixels that are too colorful to white, retain pixel otherwise
    # should erase the rest of the image basically
    for ij in np.ndindex(col_img.shape[:2]):
        # print(f"{col_img[ij][0]} - {col_img[ij][1]} = {int(col_img[ij][0]) - int(col_img[ij][1])}")
        res[ij] = col_img[ij] if ((np.abs(int(col_img[ij][0]) - int(col_img[ij][1])) <= k) &
            (np.abs(int(col_img[ij][2]) - int(col_img[ij][1])) <= k) &
            (np.abs(int(col_img[ij][0]) - int(col_img[ij][2])) <= k)) else [255,255,255]
    return res

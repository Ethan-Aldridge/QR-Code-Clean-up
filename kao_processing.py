import cv2 as cv
import numpy as np
import math
from skimage import measure # if you want to do more, like finding 
import os

img_names = ["darkened", "lightened", "kao_multi", "complexBG_tilted"]
file_ext = ".jpg"

# Get Histogram Equalized images stacked with original images
for name in img_names:
    img_path = "kao_qr\\" + name
    out_path = "kao_qr\\he_basic\\" + name

    img = cv.imread(f"{img_path}{file_ext}", cv.IMREAD_GRAYSCALE)
    # error handling
    assert img is not None, "file could not be read, check with os.path.exists()"

    # contrast adjustments via histogram equalization 
    equ = cv.equalizeHist(img)
    if name == "kao_multi":
        # stack veritically
        res = np.vstack((img,equ)) 
    else:
        #stacking images side-by-side to compare before and after
        res = np.hstack((img,equ)) 
    cv.imwrite(f"{out_path}_HE{file_ext}",res)

    # more advanced histogram equalization, CLAHE:
    # Contrast Limited Adaptive Histogram Equalization
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    if name == "kao_multi":
        # stack veritically
        res = np.vstack((img,cl1)) 
    else:
        #stacking images side-by-side to compare before and after
        res = np.hstack((img,cl1)) 
    cv.imwrite(f"{out_path}_CLAHE{file_ext}",res)

# Run through various other image processing steps in different orders 
print("Progress Check\nPress Enter to continue...")
input()  # Pauses until the user presses Enter
print("Image Processing...")

# for each image just created, the first step is always to grayscale the image,
# so it doesn't matter that the histogram equalized images are in grayscale
img_names = []
out_names = []

for filename in os.listdir("kao_qr\\he_basic\\"):
    if filename.lower().endswith((".png",".jpg")):
        filename = filename.replace(file_ext, "")
        img_names.append(os.path.join("kao_qr","he_basic", filename))
        out_names.append(os.path.join("kao_qr","img_proc", filename))

k = 0 # index for out_names

for name in img_names:
    img = cv.imread(f"{name}{file_ext}", cv.IMREAD_GRAYSCALE)

    # from https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    # Otsu's thresholding after Gaussian filtering (binarization)
    blur = cv.GaussianBlur(img,(5,5),0)
    ret3,otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    cv.imwrite(f"{out_names[k]}_otsu_gauss{file_ext}",otsu)

    adp_gauss = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,int(2*np.round(img.shape[0]/80)+3),2) # height/40->rounded to odd+2? for complex image
    cv.imwrite(f"{out_names[k]}_adp_gauss{file_ext}",adp_gauss)

    kernel = np.ones((5,5),np.uint8)
    morphology_ver = cv.morphologyEx(adp_gauss, cv.MORPH_CLOSE, kernel)
    morphology_ver = cv.morphologyEx(morphology_ver, cv.MORPH_OPEN, kernel)
    cv.imwrite(f"{out_names[k]}_morphology{file_ext}",morphology_ver)
    k = k + 1

print("Progress Check\nPress Enter to continue...")
input()  # Pauses until the user presses Enter
print("Tilt Correction...")

# tilt correction

img_names = []
out_names = []

for filename in os.listdir("kao_qr\\img_proc\\"):
    if filename.lower().endswith((".png",".jpg")):
        img_names.append(os.path.join("kao_qr","img_proc", filename))
        out_names.append(os.path.join("kao_qr","rotated", filename))

k = 0 # indexing for out_names

for name in img_names:
    img = cv.imread(name, cv.IMREAD_GRAYSCALE)

    dst = cv.Canny(img, 50, 180, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 40, None, 50, 10)
    
    if linesP is not None:
        # find the average angle of all the lines, lines are oriented left to right
        # so you can use inverse tangent, opposite (y_diff) over adjacent (x_diff)
        angles = np.ndarray(len(linesP))
        
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            angles[i] = 90 if (l[2]-l[0]) == 0 else math.atan((l[3]-l[1])/(l[2]-l[0]))*180/np.pi
        # just getting the horizontal angle to fix
        # this is the angle that keeps track of whether or not the line was originally
        # horizontal or vertical if the image wasn't tilted
        h_angles = angles[np.abs(angles) < 45]
        h_angle = np.mean(h_angles)
        print(h_angle) # for testing

        # from google ai
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, h_angle, 1.0) # -18 is close to ideal value after testing
        rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        cv.imwrite(f"{out_names[k].replace(file_ext, "")}_rotated{file_ext}", rotated)


        cdstP = cv.resize(cdstP, (int(np.round(cdstP.shape[1]/5)), int(np.round(cdstP.shape[0]/5))) )
        print(out_names[k])
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        cv.waitKey()
    else:
        print("No tilt detectable")

    k = k + 1

# end
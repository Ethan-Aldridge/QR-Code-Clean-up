import cv2 as cv
import numpy as np
import math

image = cv.imread(f"kao_tilt_crop.jpg")
# error handling
assert image is not None, "file could not be read, check with os.path.exists()"
# https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html 
img = cv.imread(f"kao_tilt_crop.jpg", cv.IMREAD_GRAYSCALE)

dst = cv.Canny(img, 50, 200, None, 3)

# Copy edges to the images that will display the results in BGR
cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 40, None, 50, 10)
# find the average angle of all the lines, lines are oriented left to right
# so you can use inverse tangent, opposite (y_diff) over adjacent (x_diff)
angles = np.ndarray(len(linesP))

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
        angles[i] = 90 if (l[2]-l[0]) == 0 else math.atan((l[3]-l[1])/(l[2]-l[0]))*180/np.pi
    # just getting the horizontal angle to fix
    h_angles = angles[np.abs(angles) < 45]
    h_angle = np.mean(h_angles)
    print(h_angle)

    # from google ai
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, h_angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    cv.imwrite("untilt_kao_tilt_crop.jpg", rotated)
else:
    print("No tilt detectable")
import numpy as np
import cv2
minValue = 70
def func(path):    
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ##rgb to grayscale
    blur = cv2.GaussianBlur(gray,(5,5),2) ##blurring to reduce image noise 

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) ##helps in segmenting images in different lighting condition
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # used for seperating foreground for background
    return res



#Import librarries and dependencies
import py_compile
import cv2
import easyocr
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import sys

#Read in image,Greyscale and Blur
img = cv2.imread("img2.jpg")  #Read image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Color conversion to gray
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)) #Color conversion to RGB
plt.show() #Show image using matplotlib

#Apply filter and find edges for localization
bfilter = cv2.bilateralFilter(gray, 11,17,17) #Noise reduction
edged = cv2.Canny(bfilter, 30,200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.show()

#Find Contours and Apply Mask
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
#Find shape(contour) - simplified version of contour 
contours = imutils.grab_contours(keypoints) #Return contour 
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #sorting contour  

location = None
for contour in contours:
    approx=cv2.approxPolyDP(contour,10,True)
    if len(approx) == 4:
        location = approx
        break
#print(location)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255,-1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

(x,y) = np.where(mask==255)
(x1,y1) = (np.min(x) , np.min(y))
(x2,y2) = (np.max(x) , np.max(y))
cropped_image = gray[x1:x2+1 , y1:y2+1]
plt.imshow(cv2.cvtColor(cropped_image , cv2.COLOR_BGR2RGB))
plt.show()

#Use Easy OCR to Read text
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
#print(result)

#Render Result
text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img , text=text, org=(100,300), fontFace=font , fontScale=1 ,color=(255,0,0),thickness=2 , lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]) , tuple(approx[2][0]) , (0,255,0),3)
plt.imshow(cv2.cvtColor(res , cv2.COLOR_BGR2RGB))
plt.show()

data = "C:\\01_Data\\PythonProgram\\Carrecord.csv"

def reading_csv():
    with open(data) as csv_stream:
        csv_reader = csv.reader(csv_stream,delimiter=',')
        next(csv_reader)

        for col in csv_reader:
            owner = col[0]    
            flat = col[1]
            np = col[2]
            
            if text in np:

                print("Approval granted")
                print(owner, flat, np)
                break
            
            
reading_csv()













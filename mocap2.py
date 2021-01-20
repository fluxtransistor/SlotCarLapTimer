#!/usr/bin/env python
 

from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import math
import time
FMT = '%H:%M:%S'
 
f=0
smin = 8000
smax = 100000
flag=( (580, 130), (850, 600) )
cars=[(0,0,0)]*4
best_laps=[0]*4
nums_laps=[0]*4
last_logs=[0]*4

DIFFERENCE = 80
TIME_THRESH = 0.5

def hsv_to_rgb(h, s, v):
    if s == 0.0: v*=255; return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

def rgb_to_hsv(o):
    r, g, b = o
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def difference(clr1, clr2):
    dif = 0
    for i in range(3):
        dif += abs(clr1[i]-clr2[i])*100
    return dif

def findcar(clr):
    min, minindex = -1,-1
    for i in range(len(cars)):
        d = difference(cars[i],clr)
        if d < min or min < 0:
            if difference < DIFFERENCE:
                min = d
                minindex = i
    return i

def timerecorded(clr,timestamp):
    
            

def translate(value, leftMin, leftMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return valueScaled / 3
     
def main():

    # Create a VideoCapture object
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)
    # Create the background subtractor object
    # Use the last 700 video frames to build the background
    back_sub = cv2.createBackgroundSubtractorMOG2(history=150, 
        varThreshold=25, detectShadows=True)
 
    # Create kernel for morphological operation
    # You can tweak the dimensions of the kernel
    # e.g. instead of 20,20 you can try 30,30.
    kernel = np.ones((30,30),np.uint8)
    phist=[((0,0),0)]
    clr=(0,0,0)
    while(True):
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        ret, frame = cap.read()
        # Use every frame to calculate the foreground mask and update
        # the background
        
        fg_mask = back_sub.apply(frame)
        
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        cv2.imshow("bs",fg_mask)
        # Close dark gaps in foreground object using closing
        #fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove salt and pepper noise with a median filter
        
        
         
        # Threshold the image to make it either black or white
        _, fg_mask = cv2.threshold(fg_mask,127,255,cv2.THRESH_BINARY)
 
        # Find the index of the largest contour and draw bounding box
        fg_mask_bb = fg_mask
        contours, hierarchy = cv2.findContours(fg_mask_bb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        for i in range(len(areas)):
            if areas[i] < smin or areas[i] > smax:
                contours[i]=None
     
        for c in contours:
            if c is None:
                continue
            # Draw the bounding box
            cnt = c
            x,y,w,h = cv2.boundingRect(cnt)
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
     
            # Draw circle in the center of the bounding box
            x2 = x + int(w/2)
            y2 = y + int(h/2)
            
            if (x2,y2) > flag[0]:
                if (x2,y2) < flag[1]:
                    currtime=time.time()
                    laptime=(int((currlap-prevlap)*1000))/1000
                    col= back_sub.apply(frame)
                    col, col2 = col[y:y+h, x:x+w], frame[y:y+h, x:x+w]
                    cv2.imshow("clip",col2)
                    countpx=0
                    ra,ga,ba=(0,0,0)
                    for y in range(h//10,h//2,4):
                        for x in range(w//4,3*w//4,4):
                            if col[y, x]==255:
                                ra += col2[y,x][0]
                                ga += col2[y,x][1]
                                ba += col2[y,x][2]
                                countpx+=1
                    ra=int(ra/(countpx+1))
                    ga=int(ga/(countpx+1))
                    ba=int(ba/(countpx+1))
                    clr = (ra,ga,ba)
                    timerecorded(clr,currtime)
                cv2.circle(frame,(x2,y2),8,clr,-1)
            # Print the centroid coordinates (we'll use the center of the
##            # bounding box) on the image
##            x8,y8=phist[len(phist)-1][0]
##            d = math.sqrt(((x2-x8)**2)+((y2-y8)**2))
##            if d == 0:
##                continue
##            phist.append(((x2,y2),d,time.time()*1000))
##            if len(phist)>100000:
##                phist.pop(0)
##        for i in range(len(phist)):
##            if i < len(phist)-2 and phist[i][1] > 40:
##                phist.pop(i)
##                continue
##        v=0
##        v2=0
##        v3=0
##        v4=0
##        for i in range(2,len(phist)):
##            if phist[i][1]<50:
##                if phist[i][2]==phist[i-1][2]:
##                    continue
##                v5,v4,v3,v2=v4,v3,v2,v
##                v=phist[i][1]/(phist[i][2]-phist[i-1][2])
##                #cv2.line(frame, phist[i][0], phist[i-1][0], hsv_to_rgb((v*1.5)%1,1,1), thickness=2)
##                cv2.circle(frame,phist[i][0],4,hsv_to_rgb(((3+v+v2+v3+v4+v5)/4)%1,1,1),-1) 
            
        # Display the resulting frame
    
        
        frame = cv2.rectangle(frame,flag[0],flag[1],clr)
        frame = cv2.rectangle(frame,(0,620),(320,720),cars[1],-1)
        frame = cv2.rectangle(frame,(320,620),(640,720),cars[2],-1)
        frame = cv2.rectangle(frame,(640,620),(720,960),cars[3],-1)
        frame = cv2.rectangle(frame,(640,620),(960,1280),cars[4],-1)
        frame = cv2.putText(frame, str(laptime), (8,705), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)
 
        # If "q" is pressed on the keyboard, 
        # exit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    print(__doc__)
    main()

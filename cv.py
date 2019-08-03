import cv2
import numpy as np 
from pynput.mouse import Button, Controller
import wx

def is_contour_bad(c):
	x,y,w,h = cv2.boundingRect(conts[i])
	return (w < 12 and h < 12)

mouse = Controller()

cap = cv2.VideoCapture(-1)

app = wx.App(False)
sx,sy = wx.GetDisplaySize()
camx,camy = (320,240)

cap.set(3,camx)
cap.set(4,camy)

lowerBound = np.array([50,100,117])
upperBound = np.array([90,240,220])

kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))

while(True):
    
    fixedConts = []
    ret, img = cap.read()
   
    #img=cv2.resize(img,(768,640))
    #cv2.imshow("Frame",img)

    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV,lowerBound,upperBound)
    maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelClose)

    maskFinal = maskClose
    conts,h = cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    

    for i in range(len(conts)) : 
        if not is_contour_bad(conts[i]):
            print("contour " + str(i) + " is not Bad!")
            fixedConts.append(conts[i])

    
        
    if(len(fixedConts) == 2):
        mouse.release(Button.left)
        x1,y1,w1,h1 = cv2.boundingRect(fixedConts[0])
        x2,y2,w2,h2 = cv2.boundingRect(fixedConts[1])
        
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        
        cx1 = x1 + w1/2
        cy1 = y1 + h1/2
        cx2 = x2 + w2/2
        cy2 = y2 + h2/2
        cx = (cx1 + cx2)/2
        cy = (cy1 + cy2)/2
        
        cv2.line(img,(int(cx1),int(cy1)),(int(cx2),int(cy2)),(255,0,0),2)
        cv2.circle(img,(int(cx),int(cy)),2,(0,0,255),2)

        mouse.position = (sx - (cx*sx/camx/2),cy*sy/camy)
        # while mouse.position != (cx*sx/camx,cy*sy/camy):
        #     mouse.position = (cx*sx/camx,cy*sy/camy)
        #     pass


    elif(len(fixedConts) == 1 ):
        x,y,w,h = cv2.boundingRect(fixedConts[0])
        
        if(w>20 and h>20):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cx = x+w/2
            cy = y+h/2
            cv2.circle(img,(int(cx),int(cy)),int((w+h)/2),(0,255,0),2)
            mouse.position = (sx - (cx*sx/camx) , cy*sy/camy) 
            mouse.press(Button.left)
            # while mouse.position != (cx*sx/camx,cy*sy/camy):
            #     pass


    cv2.imshow("mask",mask)
    cv2.imshow("com",img)


    if cv2.waitKey(1) & 0xFF == ord('q') :
        break



# cv2.imshow("maskClose",maskClose)
# cv2.imshow("maskOpen",maskOpen)

# cap.release()
# cv2.waitKey(10)

# cap.release()
# cv2.destroyAllWindows()



import cv2 #for better image manipulation and operations
import pickle #for storing data and reload data(pickle is used to store and reload data)
import numpy as np #for array operations
import os
from tqdm import tqdm #To see progress of the process
import random #for shuffeling data and frame color randomization


CATEGORIES = ["INCLUDE YOUR LABELS TO FACES IN THE FORM OF LIST"]


IMG_SIZE = 50



#test models
import cv2
import os
import numpy as np


cap = cv2.VideoCapture(0)
#If you want to test on Video in your directory, place path to that video instead of 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('dummyopencv.yml')

while True:
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_COMPLEX
   
    face_cascade = cv2.CascadeClassifier('face.xml')
 
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_faces = face_cascade.detectMultiScale(gray,1.3,5)
    
  


    if faces == ():
        pass

    else:      
        for (x,y,w,h) in faces:
            B = random.randint(0,255)
            G = random.randint(0,255)
            R = random.randint(0,255)
       
              
            
            roi_color = gray[y:y+h, x:x+w]
            org = (x,y)

            new_array = cv2.resize(roi_color, (50, 50)) 
            X = np.array(new_array).reshape(-1, 50, 50, 1)
        
            predictions, conf = recognizer.predict(new_array)
        
            probability = round(100 - float(conf)/2)
            if probability>20:
                label = CATEGORIES[predictions] + ' : ' +str(probability) 
            else:
                label = 'Unknown'

            if label=='Unknown':
                color = (0,0,R)
            else:
                color = (B,G,0)        
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)    
          
                
            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
                
        
            
            # fontScale 
            fontScale = 1
                
            # Blue color in BGR 
            
                
            # Line thickness of 2 px 
            thickness = 2
                
            # Using cv2.putText() method 
            cv2.putText(frame, label, org, font,  
                                fontScale, color, thickness, cv2.LINE_AA) 
         
    cv2.imshow("WebCam", frame)
 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


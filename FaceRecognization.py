
import cv2 #for better image manipulation and operations
import pickle #for storing data and reload data(pickle is used to store and reload data)
import numpy as np #for array operations
import os
from tqdm import tqdm #To see progress of the process
import random #for shuffeling data and frame color randomization


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

CATEGORIES = ["INCLUDE YOUR LABELS TO FACES IN THE FORM OF LIST"]


#To Generate Data


training_data = []
IMG_SIZE = 50


DATADIR = "PATH TO IMAGE DATA"
def create_training_data():
    for category in CATEGORIES:  # Labeled data in your directory

        path = os.path.join(DATADIR,category)  # create path to Your data
        class_num = CATEGORIES.index(category)  # get the classification  (0 or 1 Upto the length of your CATEGORIES list)
        count = 0
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (50, 50))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
                count+=1
              
            except Exception as e:
                print(e)  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))



random.shuffle(training_data)


for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()




#TO reuse generated data

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

y = np.array(y)



#Model One and Model two are CNN(Obtional for opencv recognizer)
#model one 
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))



model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=30, epochs=10, validation_split=0.3)
'''



#model 2
'''
#*******************************************#
#Defining the architechture

model= Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=X.shape[1:]))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dense(3,activation='softmax'))



model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])	


model.fit(X, y, batch_size=30, epochs=10, validation_split=0.3)



model.save("dummy1.h5")

'''


#openCv model

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(X,y)
recognizer.save('dummyopencv.yml')




#test models
import cv2
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


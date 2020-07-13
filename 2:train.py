
import cv2 #for better image manipulation and operations
import pickle #for storing data and reload data(pickle is used to store and reload data)
import numpy as np #for array operations
import os
from tqdm import tqdm #To see progress of the process
import random #for shuffeling data and frame color randomization


CATEGORIES = ["INCLUDE YOUR LABELS TO FACES IN THE FORM OF LIST"]


IMG_SIZE = 50



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




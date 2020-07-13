import cv2 #for better image manipulation and operations
import pickle #for storing data and reload data(pickle is used to store and reload data)
import numpy as np #for array operations
import os
from tqdm import tqdm #To see progress of the process
import random #for shuffeling data and frame color randomization

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

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#!/usr/bin/env python
# coding: utf-8

# In[25]:


# import the necessary packages
from skimage import feature
import numpy as np
from sklearn.decomposition import PCA

 
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist


# In[26]:


# import the necessary packages
#from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
#from imutils import paths
import argparse
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

desc = LocalBinaryPatterns(22, 8)
data = []
labels = []

DATADIR = "C:/Users/Tanmay/Downloads/Training/"
Cat = ["vikas","tanmay"]
i = 0
for category in Cat:
    path = os.path.join(DATADIR, category)
    calls_num = Cat.index(category)
    for img in os.listdir(path):
        gray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        #gray = cv2.resize(gray,(100,100))
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)
        hist = desc.describe(gray)
        labels.append(category)
        #print(labels[i])
        
        data.append(hist)
            # initialize the local binary patterns descriptor along with
# the data and label lists


# In[27]:


n_components = 14
pca = PCA(n_components=n_components, whiten=True).fit(data)
data1_pca = pca.transform(data)

m = LinearSVC(C=160.0, random_state=60)
m.fit(data1_pca, labels)


# In[28]:


from sklearn.neural_network import MLPClassifier


pca = PCA(n_components=n_components, whiten=True).fit(data)
data_pca = pca.transform(data)
clf = MLPClassifier(hidden_layer_sizes=(2048,),max_iter =20 ,batch_size=74, verbose=True, early_stopping=True).fit(data_pca, labels)


# In[30]:


DATADIR = "C:/Users/Tanmay/Downloads/Test/test1"
Category = ["vikas","tanmay"]
i = 0
detector= cv2.CascadeClassifier('C:/Users/Tanmay/Downloads/haarcascades/haar/haarcascade_frontalface_default.xml')
path = os.path.join(DATADIR)
calls_num = Cat.index(category)
for img in os.listdir(path):
    img = cv2.imread(os.path.join(path,img))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray,(100,100))
    faces = detector.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cimg =  img[ y:y+h, x:x+w ]
        g = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("g",g)
        hist = desc.describe(g)
        
        pred = m.predict(pca.transform(hist.reshape(1, -1)))
        prob = clf.predict_proba(pca.transform(hist.reshape(1, -1)),)
        print(prob[0,0])
        #prediction = clf.predict(g)
    # display the image and the prediction
    #if(conf > 0.6):
        cv2.putText(img, pred[0], (x,y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 3)
        #print(conf)
    #else:
    #cv2.putText(img, "unknown", (x,y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (255, 0, 0), 3)
    
    img = cv2.resize(img, (500,600))
    cv2.imshow("Image", img)
    cv2.waitKey(0)


# In[29]:


import numpy as np
import cv2

detector= cv2.CascadeClassifier('C:/Users/Tanmay/Downloads/haarcascades/haar/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    #hist = desc.describe(gray)
    #prediction = clf.predict(pca.transform(hist.reshape(1, -1)))
    #prob = clf.predict_proba(pca.transform(hist.reshape(1, -1)))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        f = img[y:y+h, x:x+w]
        gry = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        his = desc.describe(gry)
        #print(clf.predict_proba(pca.transform(his.reshape(1, -1))))
        #prediction,conf = recog.predict(gry)
        pred = clf.predict(pca.transform(his.reshape(1, -1)))
        prob = clf.predict_proba(pca.transform(hist.reshape(1, -1)),)
        print(prob[0,0])
        #if(prob[0,0] > 0.2):
        cv2.putText(img, pred[0], (x,y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 3)
        #else:
         #   cv2.putText(img, "unknown", (x,y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (255, 0, 0), 3)
        
        #print(conf)
            
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
#print(prob)
#print(prediction)


# In[ ]:





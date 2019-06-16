# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 11:13:08 2019

@author: Sathish
"""

import cv2
from imutils import paths
import numpy as np

rows = open('synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ')+1:].split(',')[0] for r in rows]
#print(rows[0])
#print([r for r in rows[0].split(' ')][1:].join())

net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.prototxt','bvlc_googlenet.caffemodel')

#print(list(paths.list_images('images/')))
#print(sorted(list(paths.list_images('images/'))))

imagepaths = list(paths.list_images('images/'))

for i in imagepaths:
    
    img = cv2.imread(i)
    sized = cv2.resize(img,(224,224))
    
    blob = cv2.dnn.blobFromImage(sized,1,(224,224),(104,117,123))
    
    net.setInput(blob)
    prediction = net.forward()
    #print(prediction[0].shape)
    print(classes[np.argsort(prediction[0])[::-1][0]])
    
    txt = classes[np.argsort(prediction[0])[::-1][0]]
    cv2.putText(img,txt,(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
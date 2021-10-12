import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import gdown
import math
from PIL import Image
import base64
import bz2
from detect_align import detect_face
from mtcnn import MTCNN #0.1.0


def get_opencv_path():
	opencv_home = cv2.__file__
	folders = opencv_home.split(os.path.sep)[0:-1]
	
	path = folders[0]
	for folder in folders[1:]:
		path = path + "/" + folder
	
	return path+"/data/"

if __name__=="__main__" :
    """if os.path.isfile(get_opencv_path()) == True:
        print(get_opencv_path())
        print('a')
    
    home = str(Path.home())
    print(home.replace('\\','/'))
    if os.path.exists(home+'/.deepface') == True:
        print('dfa')"""
    
    image_path='../media/photo/2021/10/11/gg.jpg'
    img=cv2.imread(image_path)
    h,w,_= img.shape
    h1 = int(h * 0.1)
    h2 = int(h * 0.7)
    w1 = int(w * 0.62)
    w2 = int(w * 0.96)
    img= img[h1:h2,w1:w2]


   
    opencv_path = get_opencv_path()
    face_detector_path = "../haarcascade_frontalface_default.xml"
    #face_detector_path = opencv_path+"haarcascade_frontalface.xml"

    print(face_detector_path)
    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")
    
    face_detector = cv2.CascadeClassifier(face_detector_path)


    opencv_path = get_opencv_path()
    eye_detector_path = opencv_path+"haarcascade_eye.xml"
    print(eye_detector_path)
    if os.path.isfile(eye_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ",eye_detector_path," violated.")
    
    eye_detector = cv2.CascadeClassifier(eye_detector_path)


   
    faces = []
    
    try: 
        faces = face_detector.detectMultiScale(img, 1.3, 5)
    except:
        print("pass")
        pass
    
    if len(faces) > 0:
        x,y,w,h = faces[0] #focus on the 1st face found in the image
        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        print(detected_face)

    
    else: print(faces)
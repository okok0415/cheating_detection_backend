import numpy as np
import cv2

from users.apps import facenetConfig, livenessConfig
from .detect_align import detect_face, preprocess_face

#from connect_db import load_info
from .verification import verify
from users.models import User




def authentication(image,ID,confidence=0.5):

    embedding = User.objects.filter(username=ID).only("face_embedding")
    face = detect_face(image, confidence)

    face_to_recog = face
    face = cv2.resize(face, (32, 32))

    face = face.astype('float') / 255.0
    face = np.array(face,dtype='float32')
    face = np.expand_dims(face, axis=0)


    preds = livenessConfig.model.predict(face)[0]
    j = np.argmax(preds)
    label_name =  livenessConfig.le.classes_[j]  # get label of predicted class
    result = label_name
    
    if label_name == 'real' :
        target_img = preprocess_face(face_to_recog)
        target = facenetConfig.model.predict(target_img)[0]
        result = verify(embedding, target)

    return result
    
    
     
     
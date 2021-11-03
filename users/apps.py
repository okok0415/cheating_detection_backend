from django.apps import AppConfig
from pathlib import Path

#import sys, os
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from authentication.facenet import loadFacenet
from authentication.liveness import *

class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'

class livenessConfig(AppConfig):
    name ='liveness'
    MODEL_PATH = Path('./authentication/models/liveness.model')
    LABEL_PATH = Path('./authentication/models/label_encoder.pickle')
    model=loadLiveness(MODEL_PATH)
    le = loadLabel(LABEL_PATH)

class facenetConfig(AppConfig):
    name = 'facenet'
    WEIGHTS_PATH = Path('./authentication/models/facenet_weights.h5')
    model = loadFacenet(WEIGHTS_PATH)

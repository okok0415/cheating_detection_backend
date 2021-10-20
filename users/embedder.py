import warnings
from .detect_align import preprocess_face
from .facenet import *
from .ocr import ocr
from matplotlib import pyplot as plt


def img_embedding(image):

    model = loadModel()

    #detect and align
    facial_img = preprocess_face(image, info=True, target_size = (160, 160))

    #represent
    embedding = model.predict(facial_img)[0].tobytes()

    return embedding


def info_extractor(image) :
    
    name, birth = ocr(image)

    return name, birth

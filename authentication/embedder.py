
from .detect_align import preprocess_face
from .ocr import ocr

from users.apps import facenetConfig

def img_embedding(image):

    #detect and align
    facial_img = preprocess_face(image, info=True, target_size = (160, 160))

    #represent
    embedding = facenetConfig.model.predict(facial_img)[0].tobytes()

    return embedding


def info_extractor(image) :
    
    name, birth = ocr(image)

    return name, birth

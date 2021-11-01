import pytesseract as pyt
from PIL import Image
import cv2
import numpy as np

def ocr(image):
    pyt.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract'
    config_name = ('-l kor --oem 3 --psm 12')
    config_birth = ('-l kor --oem 3 --psm 12 -c tessedit_char_whitelist=0123456789')


    original_img = image.copy()
    w,h=original_img.size 
    original_img = original_img.crop((0,h*0.2,w,h*0.5))
    img = np.array(original_img)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.medianBlur(img, 5)

  
    name = ((pyt.image_to_string(img, config=config_name).split('\n\n'))[0].replace(' ', ''))[:3]
    birth = ((pyt.image_to_string(img, config=config_birth).split('\n\n'))[1].replace(' ', ''))[:6]

    return name, birth


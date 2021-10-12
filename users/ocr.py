import pytesseract as pyt
from PIL import Image
import cv2
import numpy as np

def ocr(path):
    pyt.pytesseract.tesseract_cmd = r'..\Tesseract-OCR\tesseract'
    config_name = ('-l kor --oem 3 --psm 12')
    config_birth = ('-l kor --oem 3 --psm 12 -c tessedit_char_whitelist=0123456789')

    '''
    이미지가 텍스트를 잘 인식할 수 있도록 해야함.
    OpenCV를 이용해 적절하게 그림을 조정해 텍스트 인식을 잘 할수 있도록 전처리하는과정이 필요
    이 부분에 대해서는 다양한 시도와 조사가 필요해 보인다.
    '''

    original_img = Image.open(path)
    original_img = np.array(original_img)
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.medianBlur(img, 5)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow('1', img)
    # cv2.waitKey(0)

    name = ((pyt.image_to_string(img, config=config_name).split('\n\n'))[1].replace(' ', ''))[:3]
    birth = ((pyt.image_to_string(img, config=config_birth).split('\n\n'))[2].replace(' ', ''))[:6]

    return name, birth




#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
MIT License

Copyright (c) 2019 Sefik Ilkin Serengil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import math
from PIL import Image
import base64


import tensorflow as tf

tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
    import keras
    from keras.preprocessing.image import load_img, save_img, img_to_array
    from keras.applications.imagenet_utils import preprocess_input
    from keras.preprocessing import image
elif tf_version == 2:
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image

# --------------------------------------------------


def initialize_input(img1_path, img2_path=None):

    if type(img1_path) == list:
        bulkProcess = True
        img_list = img1_path.copy()
    else:
        bulkProcess = False

        if (
            type(img2_path) == str and img2_path != None
        ) or (  # exact image path, base64 image
            isinstance(img2_path, np.ndarray) and img2_path.any()
        ):  # numpy array
            img_list = [[img1_path, img2_path]]
        else:  # analyze function passes just img1_path
            img_list = [img1_path]

    return img_list, bulkProcess


def initialize_detector(detector_backend):

    # eye detector is common for opencv and ssd

    eye_detector_path = "./cvdata/haarcascade_eye.xml"

    if os.path.isfile(eye_detector_path) != True:
        raise ValueError(
            "Confirm that opencv is installed on your environment! Expected path ",
            eye_detector_path,
            " violated.",
        )

    global eye_detector
    eye_detector = cv2.CascadeClassifier(eye_detector_path)

    # ------------------------------
    # face detectors
    global face_detector
    # opencv_path = get_opencv_path()
    face_detector_path = "./cvdata/haarcascade_frontalface_default.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError(
            "Confirm that opencv is installed on your environment! Expected path ",
            face_detector_path,
            " violated.",
        )

    face_detector = cv2.CascadeClassifier(face_detector_path)


def initializeFolder():

    home = str(Path.home())

    if not os.path.exists("/.deepface"):
        os.mkdir("/.deepface")
        print("Directory ", "/.deepface created")

    if not os.path.exists("/.deepface/weights"):
        os.mkdir("/.deepface/weights")
        print("Directory ", "/.deepface/weights created")


def loadBase64Img(uri):
    encoded_data = uri.split(",")[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_opencv_path():
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    return path + "/data/"


def load_image(img, info=False):
    num_img = img.copy()
    """exact_image = False
	if type(img).__module__ == np.__name__:
		exact_image = True
	
	base64_img = False
	if len(img) > 11 and str(img[0:11]) == "data:image/":
		base64_img = True
	
	#---------------------------
	
	if base64_img == True:
		img = loadBase64Img(img)
		
	if exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")
		"""
    num_img = np.array(num_img)

    # ---------------------------
    if info == True:
        h, w, _ = num_img.shape
        h1 = int(h * 0.1)
        h2 = int(h * 0.7)
        w1 = int(w * 0.62)
        w2 = int(w * 0.96)
        num_img = num_img[h1:h2, w1:w2]

    return num_img


def detect_face(
    img, detector_backend="opencv", grayscale=False, enforce_detection=True
):

    img_region = [0, 0, img.shape[0], img.shape[1]]

    # if functions.preproces_face is called directly, then face_detector global variable might not been initialized.
    if not "face_detector" in globals():
        initialize_detector(detector_backend=detector_backend)

    # detector_backend=opencv
    faces = []

    try:
        faces = face_detector.detectMultiScale(img, 1.3, 5)
    except:
        pass

    if len(faces) > 0:
        x, y, w, h = faces[0]  # focus on the 1st face found in the image
        detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

        return detected_face, [x, y, w, h]

    else:  # if no face detected

        if enforce_detection != True:
            return img, img_region

        else:
            raise ValueError(
                "Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False."
            )


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(img, left_eye, right_eye):

    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if (
        b != 0 and c != 0
    ):  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    # -----------------------

    return img  # return img anyway


def align_face(img, detector_backend="opencv"):

    detected_face_gray = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )  # eye detector expects gray scale image

    eyes = eye_detector.detectMultiScale(detected_face_gray)

    if len(eyes) >= 2:

        # find the largest 2 eye

        base_eyes = eyes[:, 2]

        items = []
        for i in range(0, len(base_eyes)):
            item = (base_eyes[i], i)
            items.append(item)

        df = pd.DataFrame(items, columns=["length", "idx"]).sort_values(
            by=["length"], ascending=False
        )

        eyes = eyes[df.idx.values[0:2]]  # eyes variable stores the largest 2 eye

        # -----------------------
        # decide left and right eye

        eye_1 = eyes[0]
        eye_2 = eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        # -----------------------
        # find center of eyes

        left_eye = (
            int(left_eye[0] + (left_eye[2] / 2)),
            int(left_eye[1] + (left_eye[3] / 2)),
        )
        right_eye = (
            int(right_eye[0] + (right_eye[2] / 2)),
            int(right_eye[1] + (right_eye[3] / 2)),
        )

        img = alignment_procedure(img, left_eye, right_eye)

    return img  # return img anyway


def preprocess_face(
    img,
    info=False,
    target_size=(224, 224),
    grayscale=False,
    enforce_detection=True,
    detector_backend="opencv",
    return_region=False,
):

    # img_path = copy.copy(img)
    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    if info == True:
        img = load_image(img, info)

    img = load_image(img)
    base_img = img.copy()

    img, region = detect_face(
        img=img,
        detector_backend=detector_backend,
        grayscale=grayscale,
        enforce_detection=enforce_detection,
    )

    # --------------------------

    if img.shape[0] > 0 and img.shape[1] > 0:
        img = align_face(img=img, detector_backend=detector_backend)
    else:

        if enforce_detection == True:
            raise ValueError(
                "Detected face shape is ",
                img.shape,
                ". Consider to set enforce_detection argument to False.",
            )
        else:  # restore base image
            img = base_img.copy()

    # --------------------------

    # post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, target_size)
    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]

    if return_region == True:
        return img_pixels, region
    else:
        return img_pixels


def find_input_shape(model):

    # face recognition models have different size of inputs
    # my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.

    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    if (
        type(input_shape) == list
    ):  # issue 197: some people got array here instead of tuple
        input_shape = tuple(input_shape)

    return input_shape

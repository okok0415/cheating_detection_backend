from django.contrib.auth.models import AbstractUser
from django.db import models
from datetime import date
from PIL import Image

from authentication.embedder import img_embedding, info_extractor


class User(AbstractUser):
    schoolID = models.CharField(max_length=255, default="C000000")
    name = models.CharField(max_length=10, default="lim")
    birth = models.CharField(max_length=10, default="20000000")
    password = models.CharField(max_length=255)
    image = models.ImageField(blank=True, upload_to="photo/%Y/%m/%d", default="example.jpg")
    face_embedding = models.BinaryField(blank=True)

    def save(self, *args, **kwargs):
        self.info_extraction()
        super(User, self).save(*args, **kwargs)

    def info_extraction(self, *args, **kwargs):
        img=Image.open(self.image)
        embedding_vector = self.embedding_image(img)
        name_,birth_ = self.extraction(img)
        self.face_embedding = embedding_vector
        self.name= name_
        self.birth = birth_

    def extraction(self,img) :
        name_,birth_ = info_extractor(img)
        return name_,birth_

    def embedding_image(self,img):  
        embedding_vector = img_embedding(img)
        return embedding_vector



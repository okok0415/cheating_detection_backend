from django.contrib.auth.models import AbstractUser
from django.db import models
from datetime import date
from PIL import Image
from .embedder import img_embedding


class User(AbstractUser):
    schoolID = models.CharField(max_length=255, default="C000000")
    name = models.CharField(max_length=10, default="lim")
    birth = models.CharField(max_length=10, default="20000000")
    password = models.CharField(max_length=255)
    image = models.ImageField(blank=True, upload_to="photo/%Y/%m/%d", default="example.jpg")
    face_embedding = models.BinaryField(blank=True)

    def save(self, *args, **kwargs):
        self.embedding_image()
        super(User, self).save(*args, **kwargs)

    def embedding_image(self, *args, **kwargs):
        img=Image.open(self.image)
        embedding_vector = img_embedding(img)
        self.face_embedding = embedding_vector

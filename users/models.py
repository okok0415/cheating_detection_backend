from django.contrib.auth.models import AbstractUser
from django.db import models
from datetime import date


class User(AbstractUser):
    schoolID = models.CharField(max_length=255, default="C000000")
    name = models.CharField(max_length=10, default="lim")
    birth = models.CharField(max_length=10, default="20000000")
    password = models.CharField(max_length=255)
    image = models.ImageField(blank=True, upload_to="photo/%Y/%m/%d")


# Create your models here.

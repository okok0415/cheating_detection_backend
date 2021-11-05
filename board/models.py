from django.db import models 
from django.contrib.auth.models import User 
from django.db.models.signals import post_save 
from django.dispatch import receiver 
from django.urls import reverse 
# user model 사용 
from django.contrib.auth import get_user_model 
from taggit.managers import TaggableManager 
# Create your models here. 
# # 게시판 
class Board(models.Model): 
    objects = models.Manager() 
    title = models.CharField('title', max_length=200) 
    text = models.TextField('text', max_length=4096) 
    owner = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, verbose_name='OWNER', blank=True, null=True) 
    tags = TaggableManager(blank=True) 
    created_at = models.DateTimeField('CREATED DATE', auto_now=False, auto_now_add=True) 
    modified_at = models.DateTimeField('MODIFIED DATE', auto_now=True) 
    
    def __str__(self): 
        return self.title


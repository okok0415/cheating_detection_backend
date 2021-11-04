from django.contrib import admin 
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin 
from django.contrib.auth.models import User 
from .models import Board 

@admin.register(Board) 
class BoardAdmin(admin.ModelAdmin): 
    list_display = ('id', 'title', 'modified_at', 'tag_list') 
    # prefetch_related : database에서 쿼리 호출 수를 줄이기위해 
    # # 보드 레코드를 가져올때 태그레코드도 같이 가져온다 
    # # 1:N = select_related 
    # # N:N = prefetch_related 
    def get_queryset(self, request): 
        return super().get_queryset(request).prefetch_related('tags') 
    def tag_list(self, obj): 
        return u", ".join(o.name for o in obj.tags.all())
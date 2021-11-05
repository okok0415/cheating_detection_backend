from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    url(r"^list$", views.BoardListAPIView.as_view()),
    url(r"^<int:pk>$", views.BoardDetailAPIView.as_view()),
]

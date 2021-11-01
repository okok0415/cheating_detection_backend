from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    url(r"^register$", views.Register.as_view()),
    url(r"^login", views.LoginView.as_view()),
    url(r"^user", views.UserView.as_view()),
    url(r"^logout", views.LogoutView.as_view()),
    url(r"^update$", views.UpdateView.as_view()),
    url(r"^update-name$", views.UpdateNameView.as_view()),
]

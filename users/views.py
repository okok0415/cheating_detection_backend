from corsheaders.middleware import ACCESS_CONTROL_ALLOW_ORIGIN
from rest_framework import serializers
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import UserSerializer


class Register(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


class LoginView(APIView):
    def post(self, request):
        pass

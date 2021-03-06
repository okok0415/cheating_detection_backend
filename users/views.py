from corsheaders.middleware import ACCESS_CONTROL_ALLOW_ORIGIN
from rest_framework import serializers, status
from rest_framework import permissions
from rest_framework import response
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.permissions import IsAuthenticated
import jwt
import datetime
from .models import User
from .serializers import UserSerializer


class Register(APIView):
    """ Register API /user/register"""

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


class LoginView(APIView):
    """ Login API /user/login"""

    def post(self, request):
        username = request.data["username"]
        password = request.data["password"]

        user = User.objects.filter(username=username).first()

        if user is None:
            raise AuthenticationFailed("User not found!")

        if not user.check_password(password):
            raise AuthenticationFailed("Incorrect password")
        #
        payload = {
            "id": user.id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=1000),
            "iat": datetime.datetime.utcnow(),
        }

        token = jwt.encode(payload, "secret", algorithm="HS256")

        response = Response()
        response.set_cookie(key="jwt", value=token)
        response.data = {"jwt": token}

        return response


class UserView(APIView):
    """ User API /user/user"""

    def get(self, request):
        token = request.COOKIES.get("jwt")

        if not token:
            raise AuthenticationFailed("Unauthenticated")

        try:
            payload = jwt.decode(token, "secret", algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed("Unauthenticated")

        user = User.objects.filter(id=payload["id"]).first()
        serializer = UserSerializer(user)
        return Response(serializer.data)


class LogoutView(APIView):
    """ Logout API /user/logout"""

    def post(self, request):
        response = Response()
        response.delete_cookie("jwt")
        response.data = {"message": "success"}

        return response


class UpdateView(APIView):
    """ Update API /user/update"""

    def post(self, request):
        username = request.data["username"]
        user = User.objects.get(username=username)

        serializer = UserSerializer(instance=user, data=request.data)
        if serializer.is_valid():
            serializer.save()
        return Response(serializer.data)


class UpdateNameView(APIView):
    """ Update Name API /user/updatename"""

    def post(self, request):
        username = request.data["username"]
        User.objects.filter(username=username).update(
            name=request.data["name"], birth=request.data['birth'])
        return Response({"message": "ok"})

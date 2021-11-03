
from rest_framework import serializers
from .models import User



class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id","username","password", "image","face_embedding", "name", "birth", "date_joined", "supervisor")
        extra_kwargs = {"password": {"write_only": True}}
    """"""
    def create(self, validated_data):
        password = validated_data.pop("password", None)
        instance = self.Meta.model(**validated_data)
        if password is not None:
            instance.set_password(password)
        instance.save()
        return instance

    def update(self,instance, validated_data):
        if validated_data.get("password"):
            password = validated_data.get("password")
            instance.set_password(password)
        if validated_data.get("birth") or validated_data.get("name"):
            instance.birth = validated_data.get("birth", instance.birth)
            instance.name = validated_data.get("name", instance.name)
        instance.save()
        return instance




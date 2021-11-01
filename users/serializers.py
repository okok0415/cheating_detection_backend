
from rest_framework import serializers
from .models import User



class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id","username","password", "image","face_embedding", "name", "birth", "date_joined")
        extra_kwargs = {"password": {"write_only": True}}
    """"""
    def create(self, validated_data):
        print("@22")
        password = validated_data.pop("password", None)
        instance = self.Meta.model(**validated_data)
        if password is not None:
            instance.set_password(password)
        instance.save()
        return instance

    def update(self,instance, validated_data):
        if validated_data.get("password"):
            print("2")
            password = validated_data.get("password")
            instance.set_password(password)
        if validated_data.get("birth") or validated_data.get("name"):
            print(validated_data.get("birth"))
            print(instance.birth)

            instance.birth = validated_data.get("birth", instance.birth)
            instance.name = validated_data.get("name", instance.name)
            print(instance.birth)
        instance.save()
        return instance



class PasswordSerializer(serializers.Serializer):
    """
    Serializer for password change endpoint.
    """
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)
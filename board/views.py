from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.response import status
from django.shortcuts import get_object_or_404

from .models import Board
from .serializers import BoardSerializer
# 게시판 리스트 API 
class BoardListAPIView(APIView): 
    def get(self, request): 
        qs = Board.objects.all() 
        serializer = BoardSerializer(qs, many=True) 
        return Response(serializer.data) 
        
    def post(self, request): 
        serializer = BoardSerializer(data=request.data) 
        if serializer.is_valid(): 
            serializer.save() 
            return Response(serializer.data, status=status.HTTP_201_CREATED) 
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 

# 게시판 디테일 API 
class BoardDetailAPIView(APIView): 
    def get_object(self, pk): 
        qs = get_object_or_404(Board, pk=pk) 
        return qs 

    # 글 디테일 
    def get(self, request, pk): 
        qs = self.get_object(pk) 
        serializer = BoardSerializer(qs) 
        return Response(serializer.data) 
    
    # 글 수정 
    def put(self, request, pk): 
        qs = self.get_object(pk) 
        serializer = BoardSerializer(qs, data=request.data) 
        if serializer.is_valid(): 
            serializer.save() 
            return Response(serializer.data) 
    # 글 삭제 
    def delete(self, request, pk):
        qs = self.get_object(pk) 
        qs.delete() 
        return Response(status=status.HTTP_204_NO_CONTENT)

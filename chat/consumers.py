import json
from channels.generic.websocket import AsyncWebsocketConsumer
import base64, cv2
import numpy as np
#import imutils
import asyncio
import os
from authentication.face_authentication import authentication
#from users.models import User


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):

        self.room_group_name = "Test-Room"

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):

        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

        print("Disconnected!")

    # Receive message from WebSocket
    async def receive(self, text_data):
        receive_dict = json.loads(text_data)
        peer_username = receive_dict["peer"]
        action = receive_dict["action"]
        message = receive_dict["message"]

        # print('unanswered_offers: ', self.unanswered_offers)

        print("peer_username: ", peer_username)
        print("action: ", action)
        print("self.channel_name: ", self.channel_name)

        if (action == "new-offer") or (action == "new-answer"):
            # in case its a new offer or answer
            # send it to the new peer or initial offerer respectively

            receiver_channel_name = receive_dict["message"]["receiver_channel_name"]

            print("Sending to ", receiver_channel_name)

            # set new receiver as the current sender
            receive_dict["message"]["receiver_channel_name"] = self.channel_name

            await self.channel_layer.send(
                receiver_channel_name,
                {
                    "type": "send.sdp",
                    "receive_dict": receive_dict,
                },
            )

            return

        # set new receiver as the current sender
        # so that some messages can be sent
        # to this channel specifically
        receive_dict["message"]["receiver_channel_name"] = self.channel_name

        # send to all peers
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "send.sdp",
                "receive_dict": receive_dict,
            },
        )

    async def send_sdp(self, event):
        receive_dict = event["receive_dict"]

        this_peer = receive_dict["peer"]
        action = receive_dict["action"]
        message = receive_dict["message"]

        await self.send(
            text_data=json.dumps(
                {
                    "peer": this_peer,
                    "action": action,
                    "message": message,
                }
            )
        )
"""
class TrainConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("connect!")
    
    async def disconnect(self, code):
        print("disconnect!")

    async def receive(self, text_data):
        receive_dict = json.loads(text_data)
        img = cv2.imdecode(np.fromstring(base64.b64decode(text_data.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('image', img)
        cv2.waitKey(1)
"""
class TrainConsumer(AsyncWebsocketConsumer):
    async def connect(self):

        self.room_group_name = "Test-Room"

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):

        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

        print("Disconnected!")

    # Receive message from WebSocket
    async def receive(self, text_data):
        json_data=json.loads(text_data)
        msg = json_data['frame']
        img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        """
        receive_dict = json.loads(text_data)
        top = receive_dict["top"]
        x = receive_dict["x"]
        frame = receive_dict["frame"]
        print(top)
        print(x)
        print(frame)
        """
        #msg = text_data
        #img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        #cv2.imshow('image', img)
        #cv2.waitKey(1)
        # print('unanswered_offers: ', self.unanswered_offers)



class AuthenticationConsumer(AsyncWebsocketConsumer):
    async def connect(self):

        self.room_group_name = "Test-Room"

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):

        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

        print("Disconnected!")

    # Receive message from WebSocket
    async def receive(self, text_data):
        json_data=json.loads(text_data)
        print(json_data)
        msg = json_data['frame']
        #embedding = User.objects.filter(id=json_data["id"]).only("face_embedding")
        ID = json_data["username"] 
        img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        result = authentication(img,ID)
        print(result)
        """
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
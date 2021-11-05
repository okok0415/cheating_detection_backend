import torch
import cv2
import numpy as np
import base64
import json
import pickle
import os
import sys
import random
from users.apps import EyeConfig, device
from channels.generic.websocket import AsyncWebsocketConsumer
from few_shot_gaze.demo.new_frame_processor import frame_processer
from few_shot_gaze.demo.monitor2 import monitor
from few_shot_gaze.src.losses import GazeAngularLoss

from authentication.face_authentication import authentication
from users.models import User

from asgiref.sync import sync_to_async, async_to_sync


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.mon = monitor()
        self.room_group_name = "Test-Room"
        # self.cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
        # 이것도 디비에 저장?
        self.cam_calib = pickle.load(open("calib_cam0.pkl", "rb"))
        self.frame_processer = frame_processer(self.cam_calib)
        # 여기에 디비에 저장된 모델을 갖고와야됨 원래
        ted_parameters_path = 'Gang_gaze_network.pth.tar'
        ted_weights = torch.load(ted_parameters_path)
        self.gaze_network = EyeConfig.vanila_gaze_network.to(device)
        self.gaze_network.load_state_dict(ted_weights)
        self.subject = 'Gang'
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):

        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

        torch.cuda.empty_cache()

        print('Disconnected!')

    # Receive message from WebSocket
    async def receive(self, text_data):
        try:
            receive_dict = json.loads(text_data)
        except:
            self.send(text_data, 'Data Error')
            return
        peer_username = receive_dict['peer']
        action = receive_dict['action']
        message = receive_dict['message']

        if action == 'new-offer' or action == 'new-answer':
            # in case its a new offer or answer
            # send it to the new peer or initial offerer respectively

            receiver_channel_name = receive_dict['message']['receiver_channel_name']

            print('Sending to ', receiver_channel_name)

            # set new receiver as the current sender
            receive_dict['message']['receiver_channel_name'] = self.channel_name

            await self.channel_layer.send(
                receiver_channel_name,
                {
                    'type': 'send.sdp',
                    'receive_dict': receive_dict,
                }
            )

            return

        if action == "get-frame":
            msg = receive_dict['frame']
            frame = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
            x_hat, y_hat = self.frame_processer.process('Gang', frame, self.mon, device, self.gaze_network,
                                                        por_available=False, show=True, target=None)
            if x_hat > self.mon.w_pixels or x_hat < 0 or y_hat < self.mon.display_to_cam or y_hat > self.mon.display_to_cam + self.mon.h_pixels:
                await self.send(
                    text_data=json.dumps(
                        {
                            'peer': peer_username,
                            'action': action,
                            'message': message,
                            'x': x_hat,
                            'y': y_hat,
                        }
                    )
                )

            return

        # set new receiver as the current sender
        # so that some messages can be sent
        # to this channel specifically
        receive_dict['message']['receiver_channel_name'] = self.channel_name

        # send to all peers
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'send.sdp',
                'receive_dict': receive_dict,
            }
        )

    async def send_sdp(self, event):
        receive_dict = event['receive_dict']

        this_peer = receive_dict['peer']
        action = receive_dict['action']
        message = receive_dict['message']

        await self.send(text_data=json.dumps({
            'peer': this_peer,
            'action': action,
            'message': message,
        }))


class ScreenConsumer(AsyncWebsocketConsumer):
    async def connect(self):

        self.room_group_name = 'Test-Room'

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):

        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

        print('Disconnected!')

    async def receive(self, text_data):
        receive_dict = json.loads(text_data)
        peer_username = receive_dict['peer']
        action = receive_dict['action']
        message = receive_dict['message']

        # print('unanswered_offers: ', self.unanswered_offers)

        # print('Message received: ', message)

        print('peer_username: ', peer_username)
        # print('action: ', action)
        # print('self.channel_name: ', self.channel_name)

        if (action == 'new-offer') or (action == 'new-answer'):
            # in case its a new offer or answer
            # send it to the new peer or initial offerer respectively

            receiver_channel_name = receive_dict['message']['receiver_channel_name']

            print('Sending to ', receiver_channel_name)

            # set new receiver as the current sender
            receive_dict['message']['receiver_channel_name'] = self.channel_name

            await self.channel_layer.send(
                receiver_channel_name,
                {
                    'type': 'send.sdp',
                    'receive_dict': receive_dict,
                }
            )

            return

        if (action == "get-frame"):
            msg = receive_dict['frame']
            img = cv2.imdecode(np.fromstring(base64.b64decode(
                msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
            # receiver_channel_name = receive_dict['message']['receiver_channel_name']
            await self.send(
                text_data=json.dumps(
                    {
                        'peer': peer_username,
                        'action': action,
                        'message': message,
                        'x': 27,
                        'y': 100
                    }
                )
            )

            return

        # set new receiver as the current sender
        # so that some messages can be sent
        # to this channel specifically
        receive_dict['message']['receiver_channel_name'] = self.channel_name

        # send to all peers
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'send.sdp',
                'receive_dict': receive_dict,
            }
        )

    async def send_sdp(self, event):
        receive_dict = event['receive_dict']

        this_peer = receive_dict['peer']
        action = receive_dict['action']
        message = receive_dict['message']

        await self.send(text_data=json.dumps({
            'peer': this_peer,
            'action': action,
            'message': message,
        }))


class TrainConsumer(AsyncWebsocketConsumer):

    async def connect(self):

        self.mon = monitor()
        self.room_group_name = "Test-Room"
        # self.cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
        self.cam_calib = pickle.load(open("calib_cam0.pkl", "rb"))
        self.frame_processer = frame_processer(self.cam_calib)
        self.data = {'image_a': [], 'gaze_a': [], 'head_a': [], 'R_gaze_a': [], 'R_head_a': []}
        self.cnt = 0
        self.gaze_network = EyeConfig.gaze_network.to(device)
        self.subject = 'Gang'
        self.target = (0, 0)

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):

        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        torch.cuda.empty_cache()

        print('Disconnected!')

    async def receive(self, text_data):
        json_data = json.loads(text_data)
        if json_data['message'] == 'screen-size':
            self.mon.set_monitor(json_data['height'], json_data['width'])
        else:
            self.cnt += 1
            msg = json_data['frame']
            if json_data['message'] == 'clicked':
                g_x = json_data['x']
                g_y = json_data['y']
                g_x, g_y, _ = self.mon.monitor_to_camera(g_x, g_y)
                self.target = (g_x, g_y)

            frame = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
            processed_patch, g_n, h_n, R_gaze_a, R_head_a = self.frame_processer.process(self.subject, frame, self.mon,
                                                                                         device, self.gaze_network,
                                                                                         por_available=True, show=False,
                                                                                         target=self.target)
            self.data['image_a'].append(processed_patch)
            self.data['gaze_a'].append(g_n)
            self.data['head_a'].append(h_n)
            self.data['R_gaze_a'].append(R_gaze_a)
            self.data['R_head_a'].append(R_head_a)

            if self.cnt == 140:
                n = len(self.data['image_a'])
                k = 9
                lr = 1e-5
                steps = 5000
                cnt = 0
                _, c, h, w = self.data['image_a'][0].shape
                img = np.zeros((n, c, h, w))
                gaze_a = np.zeros((n, 2))
                head_a = np.zeros((n, 2))
                R_gaze_a = np.zeros((n, 3, 3))
                R_head_a = np.zeros((n, 3, 3))
                for i in range(n):
                    img[i, :, :, :] = self.data['image_a'][i]
                    gaze_a[i, :] = self.data['gaze_a'][i]
                    head_a[i, :] = self.data['head_a'][i]
                    R_gaze_a[i, :, :] = self.data['R_gaze_a'][i]
                    R_head_a[i, :, :] = self.data['R_head_a'][i]

                # create data subsets
                train_indices = []
                for i in range(0, k * 10, 10):
                    train_indices.append(random.sample(range(i, i + 10), 3))
                train_indices = sum(train_indices, [])

                valid_indices = []
                for i in range(k * 10, n, 10):
                    valid_indices.append(random.sample(range(i, i + 10), 1))
                valid_indices = sum(valid_indices, [])

                input_dict_train = {
                    'image_a': img[train_indices, :, :, :],
                    'gaze_a': gaze_a[train_indices, :],
                    'head_a': head_a[train_indices, :],
                    'R_gaze_a': R_gaze_a[train_indices, :, :],
                    'R_head_a': R_head_a[train_indices, :, :],
                }

                input_dict_valid = {
                    'image_a': img[valid_indices, :, :, :],
                    'gaze_a': gaze_a[valid_indices, :],
                    'head_a': head_a[valid_indices, :],
                    'R_gaze_a': R_gaze_a[valid_indices, :, :],
                    'R_head_a': R_head_a[valid_indices, :, :],
                }

                for d in (input_dict_train, input_dict_valid):
                    for k, v in d.items():
                        d[k] = torch.FloatTensor(v).to(device).detach()

                #############
                # Finetuning
                #################

                loss = GazeAngularLoss()
                optimizer = torch.optim.SGD(
                    [p for n, p in self.gaze_network.named_parameters() if n.startswith('gaze')],
                    lr=lr,
                )
                self.gaze_network.eval()
                output_dict = self.gaze_network(input_dict_valid)
                valid_loss = loss(input_dict_valid, output_dict).cpu()
                print('%04d> , Validation: %.2f' % (0, valid_loss.item()))

                for i in range(steps):
                    # zero the parameter gradient
                    self.gaze_network.train()
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    output_dict = self.gaze_network(input_dict_train)
                    train_loss = loss(input_dict_train, output_dict)
                    train_loss.backward()
                    optimizer.step()

                    if i % 100 == 99:
                        self.gaze_network.eval()
                        output_dict = self.gaze_network(input_dict_valid)
                        valid_loss = loss(input_dict_valid, output_dict).cpu()
                        message = '%04d> Train: %.2f, Validation: %.2f' % (i + 1, train_loss.item(), valid_loss.item())
                        await self.send(message)
                        print(message)
                torch.save(self.gaze_network.state_dict(), '%s_gaze_network.pth.tar' % self.subject)
                torch.cuda.empty_cache()


class AuthenticationConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.room_group_name = "Test-Room"

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

        print("Disconnected!")

    @sync_to_async
    def receive(self, text_data):
        json_data = json.loads(text_data)

        ID = json_data["username"]

        embedding_bytes = User.objects.filter(
            username=ID).values_list("face_embedding", flat=True)[0]
        embedding = np.frombuffer(embedding_bytes, dtype='float32')

        msg = json_data['frame']
        img = cv2.imdecode(np.fromstring(base64.b64decode(
            msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)

        result = authentication(img, embedding)
        print(result)

        async_to_sync(
            self.send)(
            text_data=json.dumps(
                {
                    'result': result
                }
            )
        )


class CalibrateConsumer(AsyncWebsocketConsumer):
    async def connect(self):

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.pts = np.zeros((6 * 9, 3), np.float32)
        self.pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        # capture calibration frames
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane.
        self.frames = []
        self.cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
        self.room_group_name = "Test-Room"

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):

        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

        print("Disconnected!")

    # Receive message from WebSocket
    async def receive(self, text_data):
        if len(self.frames) >= 20:
            self.send('Stop it!')
        msg = text_data
        frame = cv2.imdecode(np.fromstring(base64.b64decode(
            msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        frame_copy = frame.copy()
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        retc, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if retc:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            # Draw and display the corners
            cv2.drawChessboardCorners(frame_copy, (9, 6), corners, True)
            '''
            cv2.imshow('points', frame_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            # s to save, c to continue, q to quit
            if len(self.frames) < 20:
                self.img_points.append(corners)
                self.obj_points.append(self.pts)
                self.frames.append(frame)
                await self.send('echo : image get')
            else:
                # compute calibration matrices
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points,
                                                                   self.frames[0].shape[0:2], None,
                                                                   None)
                # check
                error = 0.0
                for i in range(len(self.frames)):
                    proj_imgpoints, _ = cv2.projectPoints(self.obj_points[i], rvecs[i], tvecs[i], mtx, dist)
                    error += (cv2.norm(self.img_points[i], proj_imgpoints, cv2.NORM_L2) / len(proj_imgpoints))
                print("Camera calibrated successfully, total re-projection error: %f" % (error / len(self.frames)))

                self.cam_calib['mtx'] = mtx
                self.cam_calib['dist'] = dist
                print("Camera parameters:")
                print(self.cam_calib)
                pickle.dump(self.cam_calib, open("calib_cam.pkl", "wb"))
                await self.send('echo : finish calibrate')

                return

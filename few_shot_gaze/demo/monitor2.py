#from win32api import GetSystemMetrics
import numpy as np


class monitor:

    def __init__(self):
        #self.h_pixels = GetSystemMetrics(1)
        #self.w_pixels = GetSystemMetrics(0)
        self.h_pixels = 1080
        self.w_pixels = 1920
        self.display_to_cam = 30
        self.h_mm = self.h_pixels * 0.26458333333333
        self.w_mm = self.w_pixels * 0.26458333333333

    def monitor_to_camera(self, x_pixel, y_pixel):

        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        # desktop은 10mm보다 더 크게 해야한다
        x_cam_mm = ((int(self.w_pixels/2) - x_pixel)/self.w_pixels) * self.w_mm
        y_cam_mm = self.display_to_cam + (y_pixel/self.h_pixels) * self.h_mm
        z_cam_mm = 0.0

        return x_cam_mm, y_cam_mm, z_cam_mm

    def camera_to_monitor(self, x_cam_mm, y_cam_mm):
        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_mon_pixel = np.ceil(int(self.w_pixels/2) - x_cam_mm * self.w_pixels / self.w_mm)
        y_mon_pixel = np.ceil((y_cam_mm - self.display_to_cam) * self.h_pixels / self.h_mm)

        return x_mon_pixel, y_mon_pixel

    def set_monitor(self, h_pixels, w_pixels):
        self.h_pixels = h_pixels
        self.w_pixels = w_pixels
        self.h_mm = self.h_pixels * 0.26458333333333
        self.w_mm = self.w_pixels * 0.26458333333333


#!/usr/bin/python3

import time
import numpy as np
import pandas as pd
import cv2.aruco as aruco
import cv2
import math
from numpy.lib.function_base import average
from numpy.linalg.linalg import _tensorsolve_dispatcher
from numpy.testing._private.utils import measure
from scipy import spatial
from timeit import default_timer as timer


class arucoTrack():

    def __init__(self):

        self.var = 0.0

        self.markerLength = 20 # in cm - marker
        self.rmatWht = None
        self.tvecWht = None
        self.HWht = None
        self.saveRotTrans = None
        # Import calibration files
        # self.mfs = cv2.FileStorage('/home/junlin/catkin_ws/src/track_cam/cam_wht_calib.yaml', cv2.FILE_STORAGE_READ)
        self.mfs = cv2.FileStorage('/home/imsel/camera/cam_wht_calib.yaml', cv2.FILE_STORAGE_READ)
        self.lfs = cv2.FileStorage('/home/imsel/camera/cam_gnt_calib.yaml', cv2.FILE_STORAGE_READ)
        self.rfs = cv2.FileStorage('/home/imsel/camera/cam_blk_new_2_calib.yaml', cv2.FILE_STORAGE_READ)

    def findRobot(self, color, cam):
        # undistort the image from different camera using different camera matrixes
        if cam == 0:
            camera_matrix = self.lfs.getNode('mtx').mat()
            dist_coeffs = self.lfs.getNode('dist').mat()

        elif cam == 1:
            camera_matrix = self.mfs.getNode('mtx').mat()
            dist_coeffs = self.mfs.getNode('dist').mat()

        else:
            camera_matrix = self.rfs.getNode('mtx').mat()
            dist_coeffs = self.rfs.getNode('dist').mat()


        # Convert the image to gray
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        # Set ArUco dictionary and detect markers
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)  # Use 4x4 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters
       
        corners, ids_gnt, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # ret = aruco.estimatePoseSingleMarkers(corners, self.markerLength, camera_matrix, dist_coeffs)

        Angle = 0
        if corners == []:
        # if len(corners) < 6:
            X = None
            Y = None
            Z = None
            Angle = None


        else:
            #marker #0
            # corner coordinates
            # print('corners:',corners)
            corners_subpix = cv2.cornerSubPix(gray, np.array(corners[0], dtype=np.float32).reshape(4,1,2), (3,3), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            undistorted_corners = cv2.fisheye.undistortPoints(corners_subpix, camera_matrix, dist_coeffs)
            undistorted_corners = undistorted_corners.reshape(-1,2)
            # print('undistorted_corners:',undistorted_corners)
            pixel_width_1 = math.sqrt((undistorted_corners[0,0]-undistorted_corners[1,0])**2+(undistorted_corners[0,1]-undistorted_corners[1,1])**2)
            pixel_width_2 = math.sqrt((undistorted_corners[1,0]-undistorted_corners[2,0])**2+(undistorted_corners[1,1]-undistorted_corners[2,1])**2)
            pixel_width_3 = math.sqrt((undistorted_corners[2,0]-undistorted_corners[3,0])**2+(undistorted_corners[2,1]-undistorted_corners[3,1])**2)
            pixel_width_4 = math.sqrt((undistorted_corners[3,0]-undistorted_corners[0,0])**2+(undistorted_corners[3,1]-undistorted_corners[0,1])**2)
            # average pixel width
            pixel_width = (pixel_width_1 + pixel_width_2 + pixel_width_3 + pixel_width_4)/4
            z = self.markerLength/pixel_width
            pixel_center = np.mean(undistorted_corners,0)

            coordinates = self.markerLength/pixel_width*pixel_center
            x = coordinates[0]
            y = coordinates[1]
            # print('coordinates #0: ', X, Y, Z)
            if cam == 0:
                X = - 9.691e-6*x**2 + 7.27e-5*x*y - 0.0002653*x*z + 1.046*x + 0.000289*y**2 + 0.001401*y*z - 0.3018*y - 0.01214*z**2 + 4.856*z - 252.4;
                Y = 2.204e-5*x**2 - 1.06e-5*x*y - 0.0001837*x*z + 0.03463*x + 7.35e-5*y**2 + 0.0004064*y*z - 1.079*y - 0.0008385*z**2 + 0.3309*z + 61.81;
                Z = - 3.527e-5*x**2 - 5.852e-5*x*y + 1.746e-5*x*z - 0.007777*x - 0.0003209*y**2 + 0.0003067*y*z - 0.07423*y + 0.001698*z**2 - 1.631*z + 272.7;

            elif cam == 1:
                X = - 1.72e-5*x**2 - 1.5e-5*x*y - 0.001238*x*z + 1.241*x + 3.8e-5*y**2 - 0.0009945*y*z + 0.2141*y + 0.002916*z**2 - 1.241*z + 373.1;
                Y = - 1.455e-5*x**2 - 1.146e-6*x*y - 5.011e-5*x*z + 0.01122*x + 2.009e-5*y**2 + 0.001912*y*z - 1.398*y + 0.001723*z**2 - 0.709*z + 302.6;
                Z = 3.597e-5*x**2 + 1.975e-5*x*y + 0.0001918*x*z - 0.04227*x - 0.0003327*y**2 - 0.0002002*y*z + 0.04885*y + 0.009779*z**2 - 4.864*z + 596.5;

            else:
                X = - 8.052e-6*x**2 + 3.832e-5*x*y - 0.001318*x*z + 1.25*x - 7.139e-5*y**2 + 0.0008202*y*z - 0.1807*y + 0.01086*z**2 - 4.475*z + 705.2;
                Y = - 5.03e-7*x**2 - 7.217e-6*x*y + 2.346e-5*x*z + 0.009031*x + 8.186e-6*y**2 + 0.001486*y*z - 1.295*y - 0.004481*z**2 + 1.841*z + 182.6;
                Z = 0.0001131*x**2 + 7.102e-5*x*y - 0.0003735*x*z + 0.0926*x - 0.0002808*y**2 - 0.0007797*y*z + 0.1733*y + 0.007465*z**2 - 4.033*z + 525.1;

            # coordinate
            X = np.round(X/100,4)
            Y = np.round(Y/100,4)
            Z = np.round(Z/100,4)
            Angle = -np.round(math.atan2(undistorted_corners[0,1]-undistorted_corners[3,1],undistorted_corners[0,0]-undistorted_corners[3,0]),4)
            
        return [X, Y, Z, Angle]
        

arTrck = arucoTrack()

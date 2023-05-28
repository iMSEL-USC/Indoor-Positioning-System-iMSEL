#!/usr/bin/env python

# import the necessary packages
from track_poly import arucoTrack
import numpy as np
import time
import cv2
from timeit import default_timer as timer
import math
from math import sin, cos, pi
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import imutils
import scipy.io as sio

class cameraFeed():

    def __init__(self):

        # initialize the video streams and allow them to warmup
        print("[INFO] starting cameras...")
        
        self.leftStream = cv2.VideoCapture('/dev/video2')
        self.middleStream = cv2.VideoCapture('/dev/video0')
        self.rightStream = cv2.VideoCapture('/dev/video4')

        # set resolution for images
        self.leftStream.set(3, 1920)
        self.middleStream.set(3, 1920)
        self.rightStream.set(3, 1920)
        
        self.leftStream.set(4, 1080)
        self.middleStream.set(4, 1080)
        self.rightStream.set(4, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.leftStream.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.middleStream.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.rightStream.set(cv2.CAP_PROP_FOURCC, fourcc)

        time.sleep(2.0)

        # set time date stamp for saved files
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        
        # set framerate for video feed
        #self.leftStream.set(5, 30)
        #self.middleStream.set(5, 30)
        #self.rightStream.set(5, 30)
        # Import calibration files
        self.mfs = cv2.FileStorage('/home/imsel/camera/cam_wht_calib.yaml', cv2.FILE_STORAGE_READ)
        self.lfs = cv2.FileStorage('/home/imsel/camera/cam_gnt_calib.yaml', cv2.FILE_STORAGE_READ)
        self.rfs = cv2.FileStorage('/home/imsel/camera/cam_blk_new_2_calib.yaml', cv2.FILE_STORAGE_READ)
        
        # Set up the video recording
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('/home/imsel/camera/'+self.timestr+'.avi', fourcc, 5.0, (1620, 960))

        # Some initial variables
        self.posMatPrev = np.array([0, 0, 0])
        self.distancePrev = 0.0

        # initialize the aruco tracker
        self.arucoTrack = arucoTrack()

        self.camera()

    def camera(self):
        x_1,y_1,th_1 = 0,0,0
        vx,vy,vth = 0,0,0
        # loop over frames from the video streams
        while True:
            
            # grab the frames from their respective video streams
            start = timer()
            retV_left, left_raw = self.leftStream.read()
            retV_middle, middle_raw = self.middleStream.read()
            retV_right, right_raw = self.rightStream.read()
            

            # update time date stamp for saved files
            self.timestr = time.strftime("%Y%m%d-%H%M%S")

            if retV_left == True and retV_middle == True and retV_right == True:
                
                x = None
                y = None
                z = None
                Angle = None
                # Feed the distorted image into the aruco tracker
                x_right, y_right, z_right, Angle_right = self.arucoTrack.findRobot(right_raw, 2)
                if z_right != None:
                    x = x_right
                    y = y_right
                    z = z_right
                    Angle = Angle_right
                    
                x_middle, y_middle, z_middle, Angle_middle = self.arucoTrack.findRobot(middle_raw, 1)
                if z_middle != None and y_middle <= 3.00:
                    x = x_middle
                    y = y_middle
                    z = z_middle
                    Angle = Angle_middle
                    
                x_left, y_left, z_left, Angle_left = self.arucoTrack.findRobot(left_raw, 0)
                if z_left != None and y_left <= 1.65:
                    x = x_left
                    y = y_left
                    z = z_left
                    Angle = Angle_left
                    
                Aruco = np.zeros(4)
                Aruco[0] = x
                Aruco[1] = y
                Aruco[2] = z
                Aruco[3] = Angle
                print('coordinates:', Aruco)
                middle = middle_raw
                time_A = timer() - start
                print('time_ArUco: ',time_A)

                text1 = 'X: ' + str(Aruco[0])
                text2 = 'Y: ' + str(Aruco[1])
                
                text3 = 'Dir: ' + str(Aruco[3])

                font = cv2.FONT_HERSHEY_SIMPLEX

                org1 = (20, 40)
                org2 = (260, 40)
                org3 = (500, 40)

                pt1 = (0, 0)
                pt2 = (700, 60)

                fontScale = 1.0

                color1 = (0, 0, 255)
                color2 = (79, 69, 54)

                thickness = 2
                imageA = imutils.rotate_bound(left_raw, 90)
                imageB = imutils.rotate_bound(middle_raw, 90)
                imageC = imutils.rotate_bound(right_raw, 90)
                left_min = cv2.resize(imageA, (540,960))
                middle_min = cv2.resize(imageB, (540,960))
                right_min = cv2.resize(imageC, (540,960))

                result_display = np.concatenate((left_min,middle_min,right_min),axis=1)
                result = cv2.rectangle(result_display, pt1, pt2, color2, cv2.FILLED, cv2.LINE_8, shift = 0)
                result = cv2.putText(result, text1, org1, font, fontScale, color1, thickness, cv2.LINE_AA, False)
                result = cv2.putText(result, text2, org2, font, fontScale, color1, thickness, cv2.LINE_AA, False)
                result = cv2.putText(result, text3, org3, font, fontScale, color1, thickness, cv2.LINE_AA, False)


                cv2.imshow("Global Position", result)
                #self.out.write(result_display)
                
                # Record video
                self.out.write(result)
                #while not rospy.is_shutdown():
                
                time_co = timer() - start
                print('time_total: ',time_co)
                key = cv2.waitKey(1) & 0xFF

                # If the 's' key is pressed, save the binary matrix and ArUco tracking
                if key == ord("s"):
                    cv2.imwrite('/home/tristankyzer/Pictures/junlin_calibration/black/'+self.timestr+'_all.png', result)

                    
                # if the 'q' key was pressed, break from the loop
                if key == ord("q"):
                    print("[INFO] killing windows...")
                    break
                
            else:
                print("[INFO] not retV...")
                break
            
        # do a bit of cleanup
        print("[INFO] cleaning up...")
        self.leftStream.release()
        self.middleStream.release()
        self.rightStream.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

cam = cameraFeed()

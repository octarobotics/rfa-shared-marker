# Copyright (c) 2024- Octa Robotics, Inc. All Rights Reserved.

import os
import cv2
import copy
import numpy as np
from cv2 import aruco
from usb_camera import UsbCamera
import sys
import yaml
from packaging import version

if __name__ == '__main__':

    """
    Camera calibrator.
    It will output the camera parameter files as mtx.npy and dist.npy under ./camera_param/
    """

    if len(sys.argv) != 2:
        print('Usage: {0} <device setting yaml file>')
        exit()

    with open(sys.argv[1], 'r') as file:
        device_settings = yaml.safe_load(file)

    bus_index = device_settings['camera']['bus_index']

    if device_settings['camera']['resolution'] == '1280x720':
        resolution = [1280, 720]
    else:
        resolution = [640, 480]

    print(
        f'Use bus_index: {bus_index}, resolution: {resolution}')

    output_dir = f"camera_param/{resolution[0]}-{resolution[1]}"
    os.makedirs(output_dir, exist_ok=True)

    # ArUco API changes in these versions
    opencv_version = version.parse(cv2.__version__)
    version_4_6_0 = version.parse('4.6.0')
    version_4_7_0 = version.parse('4.7.0')

    is_opencv_4_7_0_and_later = (version_4_7_0 <= opencv_version)

    if is_opencv_4_7_0_and_later:
        # ArUco API is renewed from 4.7
        marker_dictionary = aruco.getPredefinedDictionary(
            aruco.DICT_6X6_250)
        marker_parameters = aruco.DetectorParameters()
    else:
        # INTENT: for marker detection
        marker_dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
        marker_parameters = aruco.DetectorParameters_create()

    # INTENT: ChAruco board variables
    squares_x = 10
    squares_y = 7

    CHARUCO_BOARD = aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=0.023,
        markerLength=0.012,
        dictionary=marker_dictionary
    )

    corners_all = []
    ids_all = []
    image_size = None

    cam = UsbCamera(bus_index,
                    width=resolution[0],
                    height=resolution[1],
                    fps=30,
                    )

    validCaptures = 0
    show_count = 0

    # USAGE: Show description.
    print("1. Get sample/charuco.jpg inside the camera frame.")
    print("2. Hit 's' key 10 times to take pictures.")

    # Loop through frames
    while cam._capture.isOpened():

        ret, img = cam._capture.read()
        if ret is False:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if is_opencv_4_7_0_and_later:
            marker_detector = aruco.ArucoDetector(
                marker_dictionary)
            corners, ids, _ = marker_detector.detectMarkers(
                image=gray)
        else:
            corners, ids, _ = aruco.detectMarkers(
                image=gray,
                dictionary=marker_dictionary
            )
        if ids is None:
            # Show the image even when no marker was detected.
            if show_count % 3 == 0:
                cv2.imshow('Charuco board', cv2.resize(
                    img, (int(resolution[0]/4), int(resolution[1]/4))))
            show_count += 1
            cv2.waitKey(10)
            continue

        lined_img = copy.deepcopy(img)
        lined_img = aruco.drawDetectedMarkers(
            image=lined_img,
            corners=corners
        )

        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD
        )

        if show_count % 3 == 0:
            cv2.imshow('Charuco board', cv2.resize(
                lined_img, (int(resolution[0]/4), int(resolution[1]/4))))
        show_count += 1
        k = cv2.waitKey(10) & 0xFF

        # INTENT: If a Charuco board was found, collect image/corner points
        #         Requires at least 20 squares for a valid calibration image
        if (response > 20) and (k == ord('s')):
            corners_all.append(charuco_corners)
            ids_all.append(charuco_ids)

            cv2.imwrite(output_dir + f'/{validCaptures}.png', img)

            lined_img = aruco.drawDetectedCornersCharuco(
                image=lined_img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids
            )

            if not image_size:
                image_size = gray.shape[::-1]

            # proportion = max(lined_img.shape) / 1000.0
            # lined_img = cv2.resize(
            #     img, (int(lined_img.shape[1]/proportion), int(lined_img.shape[0]/proportion)))

            print(f'{validCaptures+1}/10')
            validCaptures += 1
            if validCaptures == 10:
                break

    cv2.destroyAllWindows()

    print(f"{validCaptures} valid captures")
    if validCaptures < 10:
        print("Calibration was failed. Enough charucoboards was not detected in the video.")
        print("Perform a better capture or reduce the minimum number of valid captures required.")
        exit()

    if len(corners_all) == 0:
        print(
            "Calibration was failed. No charucoboard was detected in the video.")
        print("Make sure that the calibration pattern is the same as the one we are looking for (ARUCO_DICT).")
        exit()

    print("Calibrating...")

    # NOTE: Perform the camera calibration
    calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print(f"Camera intrinsic parameters matrix:\n{cameraMatrix}\n")
    print(f"Camera distortion coefficients:\n{distCoeffs}\n")

    np.save(f'{output_dir}/mtx.npy', cameraMatrix)
    np.save(f'{output_dir}/dist.npy', distCoeffs)
    print(f'Calibration data were saved in {output_dir}')

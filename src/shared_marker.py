# Copyright (c) 2024- Octa Robotics, Inc. All Rights Reserved.

import math
import yaml
import usb_camera
import os
import sys
from typing import Final
from packaging import version
import collections
import numpy as np
import cv2
aruco = cv2.aruco


class SharedMarkerInfo:
    """
    Shared Marker Information class
    """

    # All final variables will be set in __init__() by SharedMarkerDetector.find()
    shared_marker_id: Final[str]
    upper_marker_corners: Final[np.ndarray]
    lower_marker_corners: Final[np.ndarray]
    short_side_length_in_image: Final[float]
    long_side_length_in_image: Final[float]

    elevation_angle: float | None
    azimuth_angle: float | None

    def __init__(self,
                 shared_marker_id: str,
                 upper_marker_corners: np.ndarray,
                 lower_marker_corners: np.ndarray,
                 short_side_length_in_image: float,
                 long_side_length_in_image: float) -> None:
        """
        Constructor to be called in SharedMarkerDetector.find()

        Args:
            shared_marker_id (str): "xxx-yyy" format ID
            upper_marker_corners (np.ndarray): 4 corners of the uppoer ArUco maker
            lower_marker_corners (np.ndarray): 4 corners of the lower ArUco maker
            short_side_length_in_image (float): The short side length of the Shared Marker
            long_side_length_in_image (float): The long side length of the Shared Marker
        """

        self.shared_marker_id = shared_marker_id
        self.upper_marker_corners = upper_marker_corners
        self.lower_marker_corners = lower_marker_corners
        self.short_side_length_in_image = short_side_length_in_image
        self.long_side_length_in_image = long_side_length_in_image

        self.elevation_angle = None
        self.azimuth_angle = None

    def __lt__(self, other: 'SharedMarkerInfo') -> bool:
        # Less-than function to sort. The bigger is the closer.
        return self.long_side_length_in_image * self.short_side_length_in_image < other.long_side_length_in_image * self.short_side_length_in_image

    def calculate_elevation_and_azimuth(self, camera_matrix: cv2.UMat, distortion_coeff: cv2.UMat) -> None:
        """
        Calculate the elevation and azimuth from the camera coordinate system to the Shared Marker.
        In the camera coordinate system, x-axis+: front, y-axis+: left, z-axis+: up.
        The camera calibraion by using camera_calibrator.py is required in advance.

        Args:
            camera_matrix (cv2.UMat): The camera intrinsic parameter matrix
            distortion_coeff (cv2.UMat): The camera distortion coefficient matrix
        """

        # Undistort the corners to the canonical screen coordinate system
        # In the canonical screen coordinate system,
        # x-axis+: right, y-axis+: down, z-axis+: front
        upper_marker_corners = cv2.undistortPoints(
            np.array(self.upper_marker_corners), camera_matrix, distortion_coeff)
        lower_marker_corners = cv2.undistortPoints(
            np.array(self.lower_marker_corners), camera_matrix, distortion_coeff)

        # Calculate the marker origin.
        # After undistortion, the shape of Shared Marker is trapezoid.
        # Then, its origin is the intersection of diagonals of the trapezoid.
        left_edge_length = lower_marker_corners[3][0][1] - \
            upper_marker_corners[0][0][1]
        right_edge_length = lower_marker_corners[2][0][1] - \
            upper_marker_corners[1][0][1]

        dir_vector = lower_marker_corners[2] - upper_marker_corners[0]
        dir_vector = dir_vector / np.linalg.norm(dir_vector)

        marker_origin_in_image = left_edge_length / \
            (left_edge_length + right_edge_length) * \
            dir_vector + upper_marker_corners[0]

        # elevator_angle and azimuth_angle are in the camera coordinate system.
        # x-axis+: front, y-axis+: left, z-axis+: up
        self.elevation_angle = math.atan(-marker_origin_in_image[0, 1])
        self.azimuth_angle = math.atan(-marker_origin_in_image[0, 0])


class SharedMarkerDetector:
    """
    Detector for Shared Marker
    """

    def __init__(self) -> None:

        # ArUco API changes in these versions
        opencv_version = version.parse(cv2.__version__)
        version_4_6_0 = version.parse('4.6.0')
        version_4_7_0 = version.parse('4.7.0')

        self._is_opencv_4_7_0_and_later = (version_4_7_0 <= opencv_version)
        self._is_opencv_4_6 = (
            version_4_6_0 <= opencv_version) and not self._is_opencv_4_7_0_and_later

        if self._is_opencv_4_7_0_and_later:
            # ArUco API is renewed from 4.7
            self._marker_dictionary = aruco.getPredefinedDictionary(
                aruco.DICT_4X4_1000)
            self._marker_parameters = aruco.DetectorParameters()
        else:
            # INTENT: for marker detection
            self._marker_dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
            self._marker_parameters = aruco.DetectorParameters_create()

        # CORNER_REFINE_CONTOUR outputs the corner positions in the pixel order (int).
        # CORNER_REFINE_SUBPIX does them in subpixel order (float).
        # CORNER_REFINE_CONTOUR is more stable than CORNER_REFINE_SUBPIX.
        self._marker_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        # self._marker_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        # 反転したマーカーも検出するか
        # self._marker_parameters.detectInvertedMarker = True

    def find(self, orig_image: cv2.UMat, draw_marker_shape: bool = False, detect_inverted_marker: bool = False) -> list[SharedMarkerInfo]:
        """
        Find Shared Marker patterns in the image.

        Args:
            orig_image (cv2.UMat): Original image. It will be overdrawn for debugging if draw_marker_shape is True.
            draw_marker_shape (bool, optional): If True, the orig_image will be overdrawn to indicate the detected ArUco markers. Defaults to False.
            detect_inverted_marker (bool, optional): If True, the contrast inverted Shared Markers will also be detected. Defaults to False.

        Raises:
            e: Exception rasied from OpenCV functions

        Returns:
            list[SharedMarkerInfo]: The list of the found Shared Marker information
        """

        self._marker_parameters.detectInvertedMarker = detect_inverted_marker
        try:
            if self._is_opencv_4_7_0_and_later:
                marker_detector = aruco.ArucoDetector(
                    self._marker_dictionary, self._marker_parameters)
                corners, ids, _rejectedImgPoints = marker_detector.detectMarkers(
                    image=orig_image)
            else:
                corners, ids, _rejectedImgPoints = aruco.detectMarkers(
                    image=orig_image,
                    dictionary=self._marker_dictionary,
                    parameters=self._marker_parameters
                )
        except Exception as e:
            raise e

        if ids is None:
            # No Shared Marker is found.
            return []

        if draw_marker_shape:
            aruco.drawDetectedMarkers(
                image=orig_image,
                corners=corners,
                ids=ids,
                borderColor=(0, 255, 0)
            )

            for i, corner in enumerate(corners):
                points = corner[0].astype(np.int32)
                cv2.polylines(orig_image, [
                    points], True, (0, 255, 255))
                cv2.putText(orig_image, str(ids[i][0]), tuple(
                    points[0]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 1)

        id_list = ids.flatten()

        ret: list = []
        neglect_index: list[int] = []

        # Check if every two ArUco markers are Shared Marker.
        # When the two ArUco markers are deteremined as a Shared Marker, they are marked to skip
        for i in range(len(corners)):
            if i in neglect_index:
                continue

            for j in range(i+1, len(corners)):
                if j in neglect_index:
                    continue

                # Pick the top left corner and the bottom left corner of an ArUco marker.
                # Ideally, the 4 corners from two ArUco markers consisnting of Shared Marker satisfy the collinearity condition.
                p_i0 = np.array(corners[i][0][0])
                p_i3 = np.array(corners[i][0][3])
                p_j0 = np.array(corners[j][0][0])
                p_j3 = np.array(corners[j][0][3])

                # 3 displacement vectors from the top left corner of ArUco marker "i"
                d_to_i3 = p_i3 - p_i0
                d_to_j0 = p_j0 - p_i0
                d_to_j3 = p_j3 - p_i0

                norm_d_to_i3 = np.linalg.norm(d_to_i3)
                norm_d_to_j0 = np.linalg.norm(d_to_j0)
                norm_d_to_j3 = np.linalg.norm(d_to_j3)

                if 2.5 * norm_d_to_i3 < norm_d_to_j0 or 2.5 * norm_d_to_i3 < norm_d_to_j3:
                    # In this case, the ratio of displacement is clearly too large to satisfy the colinearity condition AND the adjacency condition.
                    continue

                # The angle between two lines using the inner product
                inner_d_i3_j0 = np.inner(d_to_i3, d_to_j0)
                inner_d_i3_j3 = np.inner(d_to_i3, d_to_j3)

                cos_d_i3_j0 = inner_d_i3_j0 / (norm_d_to_i3 * norm_d_to_j0)
                cos_d_i3_j3 = inner_d_i3_j3 / (norm_d_to_i3 * norm_d_to_j3)

                if 0.995 < cos_d_i3_j0 and 0.995 < cos_d_i3_j3:
                    # Under this condtion, ArUco marker "j" is positioned under ArUco marker "i"
                    # norm_d_to_j3 is the length of the long side of the Shared Marker.
                    # Together with the length of the short side, the area is usable for an indicator of confidence.
                    norm_d_to_i1 = np.linalg.norm(
                        np.array(corners[i][0][1]) - p_i0)
                    ret.append(SharedMarkerInfo(f'{id_list[i]:03}-{id_list[j]:03}',
                                                np.array(corners[i][0][:]),
                                                np.array(corners[j][0][:]),
                                                norm_d_to_i1,
                                                norm_d_to_j3))
                    neglect_index.extend([i, j])

                elif cos_d_i3_j0 < -0.995 and cos_d_i3_j3 < -0.995:
                    # Under this condtion, ArUco marker "j" is positioned above ArUco marker "i"
                    # "norm_d_to_i3 + norm_d_to_j0" is the length of the long side of the Shared Marker.
                    # Together with the length of the short side, the area is usable for an indicator of confidence.
                    norm_d_to_i1 = np.linalg.norm(
                        np.array(corners[i][0][1]) - p_i0)
                    ret.append(SharedMarkerInfo(f'{id_list[j]:03}-{id_list[i]:03}',
                                                np.array(corners[j][0][:]),
                                                np.array(corners[i][0][:]),
                                                norm_d_to_i1,
                                                norm_d_to_i3 + norm_d_to_j0))
                    neglect_index.extend([i, j])

        return ret


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {0} <device setting yaml file>')
        exit()

    frame_queue = collections.deque(maxlen=3)

    def _test_callback(timestamp: float, image, actual_fps: float):
        print(f'Time: {timestamp} @ {actual_fps} fps')
        frame_queue.append((timestamp, image, actual_fps))

    with open(sys.argv[1], 'r') as file:
        device_settings = yaml.safe_load(file)

    bus_index = device_settings['camera']['bus_index']

    resolution = device_settings['camera']['resolution'].split('x')
    resolution[0] = int(resolution[0])
    resolution[1] = int(resolution[1])

    param_dir = device_settings['camera']['param_dir']

    cam_mtx: cv2.UMat | None = None
    cam_dist: cv2.UMat | None = None

    if os.path.exists(f'{param_dir}/mtx.npy') and os.path.exists(f'{param_dir}/dist.npy'):
        try:
            cam_mtx = np.load(f'{param_dir}/mtx.npy')
            cam_dist = np.load(f'{param_dir}/dist.npy')
        except Exception as e:
            print(f'Error: {e}')

    if cam_mtx is not None and cam_dist is not None:
        print(f'Load camera params in {param_dir}')
    else:
        print('No camera params')

    sm_detector = SharedMarkerDetector()

    camera = usb_camera.UsbCamera(
        bus_index, resolution[0], resolution[1], 30)
    camera.start(_test_callback)
    print(f'Camera (bus_index: {bus_index}, resolution: {resolution}) start.')

    while camera.is_capturing():
        if len(frame_queue) != 0:
            current_image = frame_queue.popleft()[1]
            sm_list = sm_detector.find(
                current_image,
                draw_marker_shape=True,
                detect_inverted_marker=True)

            for sm in sm_list:
                if cam_mtx is not None and cam_dist is not None:
                    sm.calculate_elevation_and_azimuth(cam_mtx, cam_dist)
                    print(
                        f'Shared Marker {sm.shared_marker_id} is detected (EL: {math.degrees(sm.elevation_angle):.1f}, AZ: {math.degrees(sm.azimuth_angle):.1f}).')
                else:
                    print(f'Shared Marker {sm.shared_marker_id} is detected.')

            cv2.imshow('image', current_image)
            cv2.waitKey(20)

    camera.stop()
    print('Camera stopping.')
    print('Finish!')

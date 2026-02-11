# Copyright (c) 2024- Octa Robotics, Inc. All Rights Reserved.

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

import rplidar
import usb_camera
import shared_marker

import sys
import yaml
import cv2
import threading
import numpy as np
import os
import math
import random
import signal

"""
This demo shows an exmaple implementation of a relative self-localization.

The relative position and rotation of the camera will be calculated by using the largest Shared Marker in the camera view as the referece coordinate system.

Expectation:
  The Shared Marker are placed on the wall upright.
  The camera axis and the LiDAR X-axis are aligned and horizontal.
"""


class SharedMarkerLocalization:

    _shared_marker_detector: shared_marker.SharedMarkerDetector
    _camera: usb_camera.UsbCamera
    _lidar: rplidar.RPLidar

    _cam_mtx: cv2.UMat | None
    _cam_dist: cv2.UMat | None

    _current_camera_image: cv2.UMat | None
    _image_capture_cond: threading.Condition

    _current_camera_image_to_show: cv2.UMat | None
    _image_show_lock: threading.Lock

    _current_scan_data: list[tuple[float, float]] | None
    _scan_data_lock: threading.Lock

    _camera_origin_from_lidar_origin_x: float

    _stop_flag: bool
    _thread: threading.Thread | None

    shared_marker_id_of_concern: str | None
    shared_marker_position_in_camera_coord: list[float] | None
    shared_marker_normal_in_camera_coord: list[float] | None
    camera_position_in_shared_marker_coord: list[float] | None
    camera_direction_in_shared_marker_coord: list[float] | None

    def __init__(self) -> None:
        self._image_capture_cond = threading.Condition(threading.Lock())
        self._image_show_lock = threading.Lock()
        self._scan_data_lock = threading.Lock()

        self._current_camera_image = None
        self._current_camera_image_to_show = None

        self.shared_marker_id_of_concern = None
        self.shared_marker_position_in_camera_coord = None
        self.shared_marker_normal_in_camera_coord = None
        self.camera_position_in_shared_marker_coord = None
        self.camera_direction_in_shared_marker_coord = None

    def initialize(self, device_setting_file: str) -> None:
        with open(sys.argv[1], 'r') as file:
            device_settings = yaml.safe_load(file)

        # Load relative position of camera from lidar
        self._camera_origin_from_lidar_origin_x = float(device_settings[
            'lidar']['camera_origin_from_lidar_origin_x'])

        # Load camera settings
        camera_bus_index = device_settings['camera']['bus_index']

        camera_resolution = device_settings['camera']['resolution'].split('x')
        camera_resolution[0] = int(camera_resolution[0])
        camera_resolution[1] = int(camera_resolution[1])

        camera_param_dir = device_settings['camera']['param_dir']

        self._cam_mtx = None
        self._cam_dist = None

        if os.path.exists(f'{camera_param_dir}/mtx.npy') and os.path.exists(f'{camera_param_dir}/dist.npy'):
            try:
                self._cam_mtx = np.load(f'{camera_param_dir}/mtx.npy')
                self._cam_dist = np.load(f'{camera_param_dir}/dist.npy')
            except Exception as e:
                print(f'Failed to load camera params: {e}')

        if self._cam_mtx is not None and self._cam_dist is not None:
            print(f'Load camera params in {camera_param_dir}')
        else:
            print('No camera params')

        self._shared_marker_detector = shared_marker.SharedMarkerDetector()
        self._current_camera_image = None

        # Start camera capturing
        self._camera = usb_camera.UsbCamera(
            camera_bus_index, camera_resolution[0], camera_resolution[1], 30)
        self._camera.start(self._camera_capture_callback)
        print(
            f'Camera (bus_index: {camera_bus_index}, resolution: {camera_resolution}) start.')

        # Load LiDAR settings
        lidar_path = device_settings['lidar']['path']
        lidar_model = device_settings['lidar']['model']

        self._current_scan_data = None

        # Start LiDAR scanning
        self._lidar = rplidar.RPLidar(lidar_path, lidar_model)
        self._lidar.start(self._lidar_scan_callback)
        print(f'LiDAR (path: {lidar_path}, model: {lidar_model}) start.')

    def _camera_capture_callback(self, timestamp: float, image, actual_fps: float) -> None:
        with self._image_capture_cond:
            self._current_camera_image = image
            self._image_capture_cond.notify_all()

    def _lidar_scan_callback(self, timestamp: float, scan_data: list[tuple[float, float]]) -> None:
        with self._scan_data_lock:
            self._current_scan_data = scan_data

    def start(self) -> None:
        self._stop_flag = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag = True
        if self._thread is not None and self.is_alive():
            self._thread.join(1)
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _worker(self) -> None:
        while not self._stop_flag:

            camera_image: cv2.UMat | None = None
            with self._image_capture_cond:
                while self._current_camera_image is None:
                    self._image_capture_cond.wait(0.1)

                camera_image = self._current_camera_image
                self._current_camera_image = None

            sm_list = self._shared_marker_detector.find(
                camera_image,
                draw_marker_shape=True,
                detect_inverted_marker=True)

            with self._image_show_lock:
                self._current_camera_image_to_show = camera_image
                camera_image = None

            # pick the largest Shared Marker
            if 0 < len(sm_list):
                sm_list.sort(reverse=True)
                # print(sm_list[0].shared_marker_id)

                if self._cam_mtx is not None and self._cam_dist is not None:
                    sm_list[0].calculate_elevation_and_azimuth(
                        self._cam_mtx, self._cam_dist)

                    self._calculate_relative_position_direction(sm_list[0])

                    if self.shared_marker_id_of_concern is not None:
                        print(
                            f'ID: {self.shared_marker_id_of_concern}, pos: {self.shared_marker_position_in_camera_coord}, normal: {self.shared_marker_normal_in_camera_coord}')

    def _calculate_relative_position_direction(self, shared_marker_info: shared_marker.SharedMarkerInfo) -> None:

        scan_data: list[tuple[float, float]] | None = None
        with self._scan_data_lock:
            if self._current_scan_data is not None:
                scan_data = self._current_scan_data.copy()

        if scan_data is None:
            return

        if shared_marker_info.azimuth_angle is None:
            return

        # extract neighbor points from lidar data
        lo_limit_azimuth = shared_marker_info.azimuth_angle - 0.35  # -20 deg
        hi_limit_azimuth = shared_marker_info.azimuth_angle + 0.35  # +20 deg

        # LiDAR data is clockwise from x-axis+ so split and join them to keep the sorted order
        tmp_scan_data = scan_data[int(
            len(scan_data) / 2) - 1: len(scan_data)] + scan_data[0: int(len(scan_data) / 2)]

        points_from_camera: list[tuple[float, float]] = []

        for sd in tmp_scan_data:
            x_from_camera = sd[1] * math.cos(sd[0]) - \
                self._camera_origin_from_lidar_origin_x
            y_from_camera = sd[1] * math.sin(sd[0])

            point_azimuth = math.atan2(y_from_camera, x_from_camera)
            if lo_limit_azimuth < point_azimuth and point_azimuth < hi_limit_azimuth:
                points_from_camera.append((x_from_camera, y_from_camera))

        # split points into two groups: left and rigth with the Shared Marker center.

        # Left group
        line_p, line_dir = self._line_by_ransac(
            points_from_camera[0:int(len(points_from_camera)/2) + 2])

        # skip the reference points is far from the marker
        if line_p is not None:
            j_length = math.sqrt(
                (points_from_camera[int(len(points_from_camera)/2)][0] - line_p[0]) ** 2 + (points_from_camera[int(len(points_from_camera)/2)][1] - line_p[1]) ** 2)
            distance = 1.0
            if 0.0001 < j_length:
                distance = abs((points_from_camera[int(len(points_from_camera)/2)][0] - line_p[0]) * line_dir[1] - (
                    points_from_camera[int(len(points_from_camera)/2)][1] - line_p[1]) * line_dir[0]) / j_length
            if 0.03 < distance:
                line_p = None

        if line_p is None:
            # Right group
            line_p, line_dir = self._line_by_ransac(
                points_from_camera[int(len(points_from_camera)/2) - 1:len(points_from_camera)])

        if line_p is not None:
            j_length = math.sqrt(
                (points_from_camera[int(len(points_from_camera)/2)][0] - line_p[0]) ** 2 + (points_from_camera[int(len(points_from_camera)/2)][1] - line_p[1]) ** 2)
            distance = 1.0
            if 0.0001 < j_length:
                distance = abs((points_from_camera[int(len(points_from_camera)/2)][0] - line_p[0]) * line_dir[1] - (
                    points_from_camera[int(len(points_from_camera)/2)][1] - line_p[1]) * line_dir[0]) / j_length
            if 0.03 < distance:
                line_p = None

        if line_p is not None:
            # Intersection between calculated line and the camera axis
            r_dir = (math.cos(shared_marker_info.azimuth_angle),
                     math.sin(shared_marker_info.azimuth_angle))

            num = (line_p[0] * line_dir[1] - line_p[1] * line_dir[0])
            den = (r_dir[0] * line_dir[1] - r_dir[1] * line_dir[0])

            if 0.001 < abs(den):
                l = num / den
                self.shared_marker_position_in_camera_coord = [
                    l * r_dir[0], l * r_dir[1]]

                # The extracted line dir means the surface of the Shared Marker.
                # -90 deg rotation applies to the line dir
                self.shared_marker_normal_in_camera_coord = [
                    line_dir[1], -line_dir[0]]

                # Calculate camera position and direction in Shared Marker coordinate system
                dx_in_sm_coord = [-line_dir[0], -line_dir[1]]
                dy_in_sm_coord = self.shared_marker_normal_in_camera_coord

                self.camera_position_in_shared_marker_coord = [
                    -self.shared_marker_position_in_camera_coord[0] * dx_in_sm_coord[0] -
                    self.shared_marker_position_in_camera_coord[1] *
                    dx_in_sm_coord[1],
                    -self.shared_marker_position_in_camera_coord[0] * dy_in_sm_coord[0] -
                    self.shared_marker_position_in_camera_coord[1] *
                    dy_in_sm_coord[1],
                ]

                self.camera_direction_in_shared_marker_coord = [
                    1.0 * dx_in_sm_coord[0] + 0.0 * dx_in_sm_coord[1],
                    1.0 * dy_in_sm_coord[0] + 0.0 * dy_in_sm_coord[1]
                ]

                self.shared_marker_id_of_concern = shared_marker_info.shared_marker_id
                return

            self.shared_marker_id_of_concern = None

    def _line_by_ransac(self, points: list) -> tuple:
        max_iteration = 20
        in_threshold = 0.03
        in_count_rate = 0.8

        max_in_count = 0
        max_p = None
        max_dir = None

        for _ in range(max_iteration):
            if len(points) < 2:
                return None, None
            sample_index = random.sample(range(len(points)), 2)

            # keep the order of the point to ease the decision of the direction
            if sample_index[0] < sample_index[1]:
                p1 = points[sample_index[0]]
                p2 = points[sample_index[1]]
            else:
                p1 = points[sample_index[1]]
                p2 = points[sample_index[0]]

            # Equation of the line. p1 is the reference point. Directional vector
            length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            if length < 0.0001:
                return None, None
            dir = ((p2[0] - p1[0])/length, (p2[1] - p1[1])/length)

            # Calculate the distance from each points to the line. Outer product applies. Count the number of points nearer than a threshold
            in_count = 0
            for j in range(len(points)):
                pj = points[j]
                j_length = math.sqrt(
                    (pj[0] - p1[0]) ** 2 + (pj[1] - p1[1]) ** 2)
                if 0 < j_length:
                    distance = abs((pj[0] - p1[0]) * dir[1] -
                                   (pj[1] - p1[1]) * dir[0]) / j_length
                    if distance < in_threshold:
                        in_count += 1

            if max_in_count < in_count:
                max_in_count = in_count
                max_p = p1
                max_dir = dir

        if len(points) * in_count_rate < max_in_count:
            return max_p, max_dir
        else:
            return None, None


class MyArrowItem(pg.ArrowItem):
    def paint(self, p, *args):
        p.translate(-self.boundingRect().center())
        pg.ArrowItem.paint(self, p, *args)


class DemoViewer:
    def __init__(self, sm_loc: SharedMarkerLocalization):
        self.sm_loc = sm_loc

        self.win = pg.GraphicsLayoutWidget(
            show=True, title="Shared Marker Localization Demo")

        viewport = self.win.viewport()
        if viewport is not None:
            viewport.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)

        self.win.resize(1000, 600)
        self.win.setWindowTitle('Shared Marker Localization Demo')

        pg.setConfigOptions(antialias=True)

        self.view_1 = self.win.addViewBox(row=0, col=0, colspan=2)
        self.view_1.setAspectLocked(True)
        self.view_1.invertY(True)
        self.img_1 = pg.ImageItem(border='w')
        self.img_1.setOpts(axisOrder='row-major')
        self.view_1.addItem(self.img_1)

        self.plot_1 = self.win.addPlot(title="LiDAR", row=1, col=0)
        self.plot_1.showGrid(x=True, y=True)
        self.plot_1.enableAutoRange('xy', False)
        self.plot_1.setAspectLocked(True)
        self.plot_1.setRange(xRange=(-3, 3), yRange=(-3, 3))
        self.plot_1.setMinimumSize(200, 300)
        self.plot_1_data = self.plot_1.plot(
            pen=None, symbol="o", symbolPen=(255, 137, 6), symbolSize=8, symbolBrush=(237,  237, 237))

        self.plot_2 = self.win.addPlot(
            title="Camera position & direction in Shared Marker Coord", row=1, col=1)
        self.plot_2.showGrid(x=True, y=True)
        self.plot_2.enableAutoRange('xy', False)
        self.plot_2.setAspectLocked(True)
        self.plot_2.setRange(xRange=(-3, 3), yRange=(-3, 3))
        self.plot_2.setMinimumSize(200, 300)

        # angle == 0 means leftward
        self.plot_2_arrow_camera = MyArrowItem(
            angle=0, tipAngle=60, headLen=15, tailLen=15, tailWidth=15, brush=pg.mkBrush('r'), pen={'color': 'w', 'width': 2})

        self.plot_2_arrow_marker = MyArrowItem(
            angle=90, tipAngle=80, headLen=5, tailLen=5, tailWidth=40, brush=pg.mkBrush('b'), pen={'color': 'w', 'width': 2})

        self.plot_2.addItem(self.plot_2_arrow_marker)
        self.plot_2.addItem(self.plot_2_arrow_camera)

    def update(self):
        camera_image = None
        with self.sm_loc._image_show_lock:
            if self.sm_loc._current_camera_image_to_show is not None:
                camera_image = self.sm_loc._current_camera_image_to_show

        if camera_image is not None:
            self.img_1.setImage(cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB))
            self.view_1.autoRange()

        scan_data: list[tuple[float, float]] | None = None
        with self.sm_loc._scan_data_lock:
            if self.sm_loc._current_scan_data is not None:
                scan_data = self.sm_loc._current_scan_data.copy()

        if scan_data is not None:
            lidar_data_x = [d[1] * math.cos(d[0]) for d in scan_data]
            lidar_data_y = [d[1] * math.sin(d[0]) for d in scan_data]
            self.plot_1_data.setData(x=lidar_data_x, y=lidar_data_y)

        if self.sm_loc.shared_marker_id_of_concern is not None:
            camera_pos = self.sm_loc.camera_position_in_shared_marker_coord  # カメラの位置を取得
            if camera_pos is None or self.sm_loc.camera_direction_in_shared_marker_coord is None:
                return

            self.plot_2_arrow_camera.setPos(camera_pos[0],
                                            camera_pos[1])
            delta_x = -self.sm_loc.camera_direction_in_shared_marker_coord[0]
            delta_y = self.sm_loc.camera_direction_in_shared_marker_coord[1]
            angle = math.degrees(math.atan2(delta_y, delta_x))
            self.plot_2_arrow_camera.setStyle(angle=angle)

        self.plot_2.update()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {0} <device setting yaml file>')
        exit()

    sm_loc = SharedMarkerLocalization()
    sm_loc.initialize(sys.argv[1])
    sm_loc.start()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = pg.mkQApp("Shared Marker Localization Demo")

    viewer = DemoViewer(sm_loc)

    timer = QtCore.QTimer()
    timer.timeout.connect(viewer.update)
    timer.start(30)
    app.exec()

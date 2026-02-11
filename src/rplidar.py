# Copyright (c) 2024- Octa Robotics, Inc. All Rights Reserved.

import pyrplidar
import pyrplidar_protocol
import time
import sys
import yaml
import collections
import threading
from typing import Callable

import atexit
import math

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


class RPLidar:
    """
    RPLidar wrapper class
    """
    _device_path: str
    _model: str

    _lidar: pyrplidar.PyRPlidar
    _info: pyrplidar.PyRPlidarDeviceInfo
    _health: pyrplidar.PyRPlidarHealth
    _samplerate: pyrplidar.PyRPlidarSamplerate
    _scan_modes: list

    _scan_callback: Callable[[float, list[tuple[float, float]]], None]

    _stop_flag: bool
    _thread: threading.Thread | None

    def __init__(self, device_path: str, model: str) -> None:

        self._lidar = pyrplidar.PyRPlidar()

        self._device_path = device_path

        match model:
            case 'S2' | 'S2L':
                self._baudrate = 1000000
            case _:
                self._baudrate = 256000

        self._lidar.connect(port=self._device_path,
                            baudrate=self._baudrate, timeout=3)
        if self._lidar.lidar_serial is not None and self._lidar.lidar_serial._serial is not None:
            self._lidar.lidar_serial._serial.dsrdtr = False

        self._info = self._lidar.get_info()
        self._health = self._lidar.get_health()
        self._samplerate = self._lidar.get_samplerate()
        self._scan_modes = self._lidar.get_scan_modes()

        for scan_mode in self._scan_modes:
            print(scan_mode)

        self._stop_flag = False
        self._thread = None

        atexit.register(self.__del__)

    def __del__(self) -> None:
        self.stop()
        self._lidar.disconnect()

    def start(self, scan_callback: Callable[[float, list[tuple[float, float]]], None]) -> None:
        self._lidar.disconnect()
        self._lidar.connect(port=self._device_path,
                            baudrate=self._baudrate, timeout=3)

        self._scan_callback = scan_callback
        self._stop_flag = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag = True
        if self._thread is not None and self.is_alive():
            self._thread.join(1)
        self._thread = None

    def is_scanning(self) -> bool:
        return self.is_alive() and not self._stop_flag

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _worker(self) -> None:

        self._lidar.set_motor_pwm(660)
        time.sleep(2)

        while not self._stop_flag:
            """
            Python serial (macOS) and RPLiDAR S3 seems incompatible.
            It may be because high baudrate on macOS is unstable.
            The communication synchronization often breaks down.

            Because this script is just for demo, it simply skips suspicious scan data and then reset the serial communication.

            For deployment, please use the genuine rplidar_sdk written in C++.
            """

            self._lidar.send_command(pyrplidar_protocol.RPLIDAR_CMD_SCAN)
            try:
                discriptor = self._lidar.receive_discriptor()
            except Exception as e:
                print(f'Bad init: {e}')
                self._lidar.disconnect()
                self._lidar.connect(port=self._device_path,
                                    baudrate=self._baudrate, timeout=3)
                self._lidar.set_motor_pwm(660)
                time.sleep(2)
                continue

            scan_buf: list[tuple[float, float]] = []

            while not self._stop_flag:
                try:
                    recv_data = self._lidar.receive_data(discriptor)
                except Exception as e:
                    print(f'Bad scan: {e}')
                    break

                scan = pyrplidar.PyRPlidarMeasurement(recv_data)

                # RPLiDAR's angle is clockwise [deg] and distance is [mm]
                # Convert them to counter-clockwise [rad] and [m] here

                if scan.start_flag:
                    if 0 < len(scan_buf) < 100:
                        print(f'Bad scan: {len(scan_buf)}')
                        break

                    self._scan_callback(time.time(), scan_buf)

                    if 30 < scan.distance:
                        scan_buf = [
                            (-math.radians(scan.angle), scan.distance * 0.001)]
                    else:
                        scan_buf = []
                else:
                    if 30 < scan.distance:
                        scan_buf.append(
                            (-math.radians(scan.angle), scan.distance * 0.001))

            self._lidar.send_command(pyrplidar_protocol.RPLIDAR_CMD_STOP)
            time.sleep(0.1)
            if self._lidar.lidar_serial is not None and self._lidar.lidar_serial._serial is not None:
                self._lidar.lidar_serial._serial.reset_input_buffer()

        self._lidar.set_motor_pwm(0)
        time.sleep(2)
        self._lidar.disconnect()


class RPLidarTestViewer:
    def __init__(self) -> None:
        self.win = pg.GraphicsLayoutWidget(
            show=True, title="RPLidar Test Viewer")

        viewport = self.win.viewport()
        if viewport is not None:
            viewport.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)

        self.win.resize(1000, 600)
        self.win.setWindowTitle('RPLidar Test Viewer')

        pg.setConfigOptions(antialias=True)

        self.plot_1 = self.win.addPlot(title="LiDAR", row=0, col=0)
        self.plot_1.showGrid(x=True, y=True)
        self.plot_1.enableAutoRange('xy', False)
        self.plot_1.setAspectLocked(True)
        self.plot_1.setRange(xRange=(-3, 3), yRange=(-3, 3))
        self.plot_1.setMinimumSize(200, 300)
        self.plot_1_data = self.plot_1.plot(
            pen=None, symbol="o", symbolPen=(255, 137, 6), symbolSize=8, symbolBrush=(237,  237, 237))

    def update(self, scan_data: list[tuple[float, float]]) -> None:
        lidar_data_x = [d[1] * math.cos(d[0]) for d in scan_data]
        lidar_data_y = [d[1] * math.sin(d[0]) for d in scan_data]
        self.plot_1_data.setData(x=lidar_data_x, y=lidar_data_y)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {0} <device setting yaml file>')
        exit()

    frame_queue: collections.deque[tuple[float, list[tuple[float, float]]]] = collections.deque(
        maxlen=3)

    def _test_callback(timestamp: float, scan_data: list[tuple[float, float]]):
        frame_queue.append((timestamp, scan_data))

    with open(sys.argv[1], 'r') as file:
        device_settings = yaml.safe_load(file)

    lidar_path = device_settings['lidar']['path']
    lidar_model = device_settings['lidar']['model']

    lidar = RPLidar(lidar_path, lidar_model)
    lidar.start(_test_callback)
    print(f'LiDAR (path: {lidar_path}, model: {lidar_model}) start.')

    app = pg.mkQApp("RPLidar Test Viewer")

    viewer = RPLidarTestViewer()

    def draw_latest():
        scan_data = None
        if len(frame_queue) > 0:
            scan_data = frame_queue.popleft()
            viewer.update(scan_data[1])

    timer = QtCore.QTimer()
    timer.timeout.connect(draw_latest)
    timer.start(30)
    app.exec()

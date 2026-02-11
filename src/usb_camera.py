# Copyright (c) 2024- Octa Robotics, Inc. All Rights Reserved.

import collections
import threading
import time
import sys
import cv2
from typing import Callable
import yaml


class UsbCamera:
    """ 
    USB Camera handler class using OpenCV
    """

    _capture: cv2.VideoCapture
    _capture_callback: Callable[[float, cv2.UMat, float], None] | None

    _prev_capture_time: float
    _actual_fps: float

    _resume_flag: threading.Event
    _stop_flag: bool
    _thread: threading.Thread | None

    def __init__(self, bus_index: int | str, width: int, height: int, fps: int) -> None:
        """
        Constructor

        Args:
            usb_video_path (str): The path of the camera
            width (int): Horizontal resolution of the camera
            height (int): Vertical resolution of the camera
            fps (int): Frame per seconds to capture
            bus_index (int, optional): The bus index used in OpneCV. Defaults to None (auto) 

        Raises:
            IOError: Failed to open the camera
        """
        if isinstance(bus_index, str) and bus_index.isdecimal():
            bus_index = int(bus_index)
        elif not isinstance(bus_index, int):
            bus_index = 0

        self._capture = cv2.VideoCapture(bus_index)
        if not self._capture.isOpened():
            raise IOError

        self._capture.set(cv2.CAP_PROP_FOURCC,
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        self._capture_callback = None

        self._prev_capture_time = 0
        self._actual_fps = 0

        self._resume_flag = threading.Event()
        self._stop_flag = False
        self._thread = None

    def __del__(self) -> None:
        # terminate the capturing thread
        self.stop()
        self._capture.release()

    def start(self, capture_callback: Callable[[float, cv2.UMat, float], None]) -> None:
        """
        Start the capturing thread 

        Args:
            capture_callback (Callable[[float, cv2.UMat, float], None]): Callback function which will be called soon after capturing a frame
        """
        self._capture_callback = capture_callback
        self._stop_flag = False
        self._resume_flag.set()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the capturing thread
        """

        self._stop_flag = True
        self._resume_flag.set()
        if self._thread is not None and self.is_alive():
            self._thread.join(1)
        self._thread = None

    def resume(self) -> None:
        """
        Resume the capturing thread
        """

        self._resume_flag.set()

    def suspend(self) -> None:
        """
        Suspend the capturing thread
        """
        self._resume_flag.clear()
        self._prev_capture_time = 0
        self._actual_fps = 0

    def is_capturing(self) -> bool:
        """
        Check the status of capturing

        Returns:
            bool: "True" means the thread is capturing frames.
        """

        return self.is_alive() and not self._stop_flag and self._resume_flag.is_set()

    def is_alive(self) -> bool:
        """
        Check the alive status of thread

        Returns:
            bool: "True" means the thread is alive.
        """
        return self._thread is not None and self._thread.is_alive()

    def _worker(self) -> None:
        while not self._stop_flag:
            ret, frame = self._capture.read()
            current_time = int(time.time() * 1000)/1000
            tmp_current_time = self._capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if tmp_current_time != 0:
                current_time = tmp_current_time

            if not ret:
                raise IOError

            if self._prev_capture_time != 0:
                self._actual_fps = 1.0 / \
                    (current_time - self._prev_capture_time)
            else:
                self._actual_fps = 0
            self._prev_capture_time = current_time

            if self._capture_callback is not None:
                # Callback
                self._capture_callback(current_time, frame, self._actual_fps)

            self._resume_flag.wait()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {0} <device setting yaml file>')
        exit()

    frame_queue = collections.deque(maxlen=3)

    def _test_callback(timestamp: float, image, actual_fps: float) -> None:
        print(f'Time: {timestamp} @ {actual_fps} fps')
        frame_queue.append((timestamp, image, actual_fps))

    with open(sys.argv[1], 'r') as file:
        device_settings = yaml.safe_load(file)

    bus_index = device_settings['camera']['bus_index']

    resolution = device_settings['camera']['resolution'].split('x')
    resolution[0] = int(resolution[0])
    resolution[1] = int(resolution[1])

    camera = UsbCamera(
        bus_index, resolution[0], resolution[1], 30)
    camera.start(_test_callback)
    print(f'Camera (bus_index: {bus_index}, resolution: {resolution}) start.')

    count = 0
    while camera.is_capturing() and count < 3:
        print('Capturing...')
        if len(frame_queue) != 0:
            cv2.imshow('image', frame_queue.popleft()[1])
            cv2.waitKey(10)
        time.sleep(1)
        count += 1

    camera.suspend()
    print('Camera suspended.')

    count = 0
    while not camera.is_capturing() and count < 3:
        print('Suspended...')
        time.sleep(1)
        count += 1

    camera.resume()
    print('Camera resumed.')

    count = 0
    while camera.is_capturing() and count < 3:
        print('Capturing...')
        if len(frame_queue) != 0:
            cv2.imshow('image', frame_queue.popleft()[1])
            cv2.waitKey(10)
        time.sleep(1)
        count += 1

    camera.stop()
    print('Camera stopping.')

    while camera.is_alive() and count < 3:
        print('Wait stopping...')
        time.sleep(1)
        count += 1

    print('Finish.')

# Copyright (c) 2024- Octa Robotics, Inc. All Rights Reserved.

import cv2
import numpy as np
import sys
import subprocess
import os

aruco = cv2.aruco


def generate(top_id: int, bottom_id: int) -> None:
    """
    Generating function of Shared Marker.
    Shared Marker comprises of two vertically stacked ArUco markers with the codebook of DICT_4X4_1000.

    This function outputs the tiff images with 8mm cell with 350dpi

    In 350 dpi,
    - 120mm x 70mm --> 1654px x 965px
    - 48mm x 48mm  --> 661px x 661px
    - 8mm x 8mm    --> 110px x 110px
    - 148mm x 100mm --> 2039px x 1378px

    Args:
        top_id (int): The ID of the top ArUco marker
        bottom_id (int): The ID of the bottom ArUco marker
    """

    height = 1654
    width = 965

    marker_size = 661
    cell_size = 110

    papar_height = 2039
    paper_width = 1378

    img = np.zeros((papar_height, paper_width), dtype=np.uint8)
    img += 255

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    top_marker = aruco.generateImageMarker(dictionary, top_id, marker_size, 0)
    bottom_marker = aruco.generateImageMarker(
        dictionary, bottom_id, marker_size, 0)

    #    top_marker = aruco.drawMarker(dictionary, top_id, marker_size, 0)
    #    bottom_marker = aruco.drawMarker(dictionary, bottom_id, marker_size, 0)

    x_offset = int((width - marker_size)/2) + 200
    y_offset = int((height - marker_size * 2 - cell_size)/2) + 200

    img[y_offset: (y_offset + marker_size),
        x_offset: (x_offset + marker_size)] = top_marker

    img[(y_offset + marker_size + cell_size): (y_offset + marker_size * 2 + cell_size),
        x_offset: (x_offset + marker_size)] = bottom_marker

    id_text = '{0:0>3}-{1:0>3}'.format(top_id, bottom_id)

    cv2.putText(img, id_text, (width + 200 - 255, height + 200 - 70),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_8)

    # draw cutting lines
    cv2.line(img, pt1=(width + 200, 0), pt2=(width + 200, 400),
             color=(0, 0, 0), thickness=2)
    cv2.line(img, pt1=(width, 200), pt2=(width + 400, 200),
             color=(0, 0, 0), thickness=2)

    cv2.line(img, pt1=(width + 200, height), pt2=(width + 200,
             height + 400), color=(0, 0, 0), thickness=2)
    cv2.line(img, pt1=(width, height + 200), pt2=(width+400,
             height + 200), color=(0, 0, 0), thickness=2)

    cv2.line(img, pt1=(200, 0), pt2=(200, 400), color=(0, 0, 0), thickness=2)
    cv2.line(img, pt1=(0, 200), pt2=(400, 200), color=(0, 0, 0), thickness=2)

    cv2.line(img, pt1=(200, height), pt2=(
        200, height + 400), color=(0, 0, 0), thickness=2)
    cv2.line(img, pt1=(0, height + 200), pt2=(400, height + 200),
             color=(0, 0, 0), thickness=2)

    filename = 'shared_marker_{0}.tiff'.format(id_text)

    cv2.imwrite('/tmp/' + filename, img)

    output_dir = './image'

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        f'magick /tmp/{filename}  -density 350 -depth 1 {output_dir}/{filename}', shell=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f'Usage: python {sys.argv[0]} <ID of top marker> <ID of bottom marker>')
        sys.exit()

    generate(int(sys.argv[1]), int(sys.argv[2]))

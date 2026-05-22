import cv2
import io
import socket
import struct
import time
import pickle
import numpy as np
from prepare_final_img import plot_bboxes_with_labels_and_scores


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('Public IP of your EC2 instance', 8485))
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)

cam.set(3, 360)
cam.set(4, 240)

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    ret, frame = cam.read()

    result, frame_encoded = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame_encoded, 0)
    size = len(data)


    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)

    while True:
        bbox_data = client_socket.recv(4096)
        if not bbox_data:
            print('no data received --> breaking')
            break
        bboxes_data = pickle.loads(bbox_data)
        break


    frame_with_bboxes = plot_bboxes_with_labels_and_scores(frame.copy(), bboxes_data[0], bboxes_data[1], bboxes_data[2])

    cv2.imshow('ImageWindow', frame_with_bboxes)
    cv2.waitKey(1)

    img_counter += 1

cam.release()
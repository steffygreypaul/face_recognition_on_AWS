import socket
import cv2
import pickle
import numpy as np
import struct
from face_recognition import faceRecognizer, load_models

detector, embedder_model, recognition_model = load_models()

HOST=''
PORT=8485

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

s.bind((HOST,PORT))
s.listen(10)
print('Now listening!')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

while True:
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(4096)

    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    face_recognizer = faceRecognizer(detector, embedder_model, recognition_model)
    boxes_properties, labels, scores = face_recognizer.recognize_faces(frame)

    data_to_be_sent_back = [np.array(boxes_properties), np.array(labels), np.array(scores)]

    data_string = pickle.dumps(data_to_be_sent_back)

    conn.send(data_string)
    
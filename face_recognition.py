import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib.patches import Rectangle
from mtcnn import MTCNN
from keras.models import load_model
import pathlib
import os
import pickle

BASE_PATH = pathlib.Path(__file__).parent
MODELS_PATH = os.path.join(BASE_PATH, 'models')


class faceRecognizer:

    def __init__(self, detector, embedder_model, recognition_model):
        self.detector = detector
        self.embedder_model = embedder_model
        self.recognition_model = recognition_model

    def recognize_faces(self, img):
        boxes_properties = []
        labels = []
        scores = []
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        results = self.detector.detect_faces(img)  # "results" is a list of all returned bounding boxes,
        # each element of the list is a dictionary indicating some information about the bounding box
        # i.e. coordinates of the bottom-left corner of the bounding box,width,heigh

        for result in results:
            x1, y1, width, height = result['box']
            x1, y1 = abs(x1), abs(y1)  # fix the bug when x1 or y1 are negative!
            x2, y2 = x1 + width, y1 + height

            boxes_properties.append([x1, y1, width, height])

            face = img[y1:y2, x1:x2]
            # Resize the face to feed it to the FaceNet model
            face = cv2.resize(face, (160, 160))
            face = face.astype('float32')
            mean, std = np.mean(face), np.std(face)
            face = (face - mean) / std
            # add to the dimension of the image
            face = np.expand_dims(face, axis=0)  # the input to the FaceNet should be of shape (None,160,160,3)
            # get the embedding of the face
            embedding = self.embedder_model.predict(face)
            label = self.recognition_model.predict(np.expand_dims(embedding[0], axis=0))
            score = self.recognition_model.predict_proba(np.expand_dims(embedding[0], axis=0))
            if np.max(score) >= 0.5:
                labels.append(label)
                scores.append(str(np.max(score)))
            else:
                labels.append('Unknown')
                scores.append('')

        return boxes_properties, labels, scores


def load_models():
    detector = MTCNN()
    embedder_model = load_model(os.path.join(MODELS_PATH, 'facenet_keras.h5'))
    recognition_model = pickle.load(open(os.path.join(MODELS_PATH, 'recognition_model.sav', 'rb')))

    return detector, embedder_model, recognition_model

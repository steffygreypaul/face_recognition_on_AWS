import cv2

def plot_bboxes_with_labels_and_scores(frame, bboxes, labels, scores):
    '''
    :param frame: frame received from webcam using opencv
    :param bboxes: list of detected bounding boxes in a frame. format: [x1,y1, width, height]
    :return: frame with bounding boxes plotted on it.
    '''

    for i in len(bboxes):
        x1, y1, width, height = bboxes[i]
        frame = cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (0, 0, 255), 2)
        frame = cv2.putText(frame, labels[i] + scores[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

    return frame
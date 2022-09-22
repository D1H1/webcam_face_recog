import os

import cv2 as cv
import numpy as np
import time
import shutil

capture = cv.VideoCapture(0)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

DIR = 'face_images/my'

labels = list()
features = list()


def get_face_images(time_for_capt=20):
    time_start = time.time()
    cnt = 0
    shutil.rmtree(DIR)
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    while True:
        isTrue, frame = capture.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        face_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10)

        for (x, y, w, h) in face_rect:
            face_roi = frame[y:y + h, x:x + w]
            cv.imwrite(f'face_images/my/image_{cnt}.jpg', face_roi)

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.imshow('detecting', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break
        elif time.time() - time_start > time_for_capt:
            break
        time.sleep(0.025)
        cnt += 1


def create_train():
    dirs = os.listdir('face_images')

    for person in dirs:
        label = dirs.index(person)

        path = os.path.join('face_images', person)

        if not os.path.exists(DIR):
            return 'No directory'
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_read = cv.imread(img_path)
            gray = cv.cvtColor(img_read, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)


def train():
    global features
    global labels

    create_train()

    features = np.array(features, dtype='object')
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)

    face_recognizer.save('face_trained.yml')


def detect_face():
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_trained.yml')

    while True:
        isTrue, frame = capture.read()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        face_rect = haar_cascade.detectMultiScale(frame, 1.1, 4)

        for (x, y, w, h) in face_rect:

            faces_roi = gray_frame[y:y + w, x:x + h]

            label, confidence = face_recognizer.predict(faces_roi)
            print(label)
            if str(label) == '0':
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                cv.putText(frame, str(confidence), (frame.shape[0] // 2, frame.shape[1] - 20), cv.FONT_HERSHEY_COMPLEX, 1.0,
                           (0, 255, 0), thickness=2)
                cv.imshow('recognize', frame)

            else:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
                cv.imshow('recognize', frame)
            print(f'Confidence level: {confidence}')

        if cv.waitKey(20) & 0xFF == ord('d'):
            break


if __name__ == '__main__':
    get_face_images()

    print('images captured')

    create_train()

    print('Model created')

    train()

    print('model trained')

    detect_face()

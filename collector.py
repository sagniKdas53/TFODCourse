# Import opencv
import cv2
# Import uuid
import uuid
# Import Operating System
import os
# Import time
import time

labels = ['Ami','Ma','Dida']
number_imgs = 5

IMAGES_PATH = os.path.join('Tensorflow', 'workspace',
                           'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    os.mkdir(IMAGES_PATH)

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir(path)

cap = cv2.VideoCapture(2)
for label in labels:
    print('Collecting images for {}'.format(label))
    imgnum = 0
    while imgnum < number_imgs:
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label, label +
                               '.'+'{}.jpg'.format(str(uuid.uuid1())))
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(imgname, frame)
            imgnum += 1
            print('Collected image {}'.format(imgnum))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

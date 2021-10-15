import os
labels = ['thumbsup', 'thumbsdown', 'fuckyou', 'okey']
number_imgs = 5

IMAGES_PATH = os.path.join('Tensorflow', 'workspace',
                           'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    os.mkdir(IMAGES_PATH)

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir(path)
import cv2
from align import AlignDlib
import numpy as np
import os.path
from model import create_model
import csv
import pickle

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

def align_image(img):
    alignment = AlignDlib('models/landmarks.dat')
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

#Save embedded vector to pkl function
def save_embedded_vectors_to_pkl(filename, embedded_vectors):
    output = open(filename, 'wb')
    pickle.dump(embedded, output)
    output.close()
    print("\n Embedded Vectors successfully saved to " + filename)

#Save embedded vector to csv function
def save_embedded_vectors_to_csv(filename, embedded_vectors):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img_vector in embedded_vectors:
            spamwriter.writerow(img_vector)
        print("\n Embedded Vectors successfully saved to " + filename)

## Intialize
csvfile = 'labeled_ev.csv'
pklfile = 'labeled_ev.pkl'
metadata = load_metadata('images')
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

## Get Embedded Vectors
embedded = np.zeros((metadata.shape[0], 128))
for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    img = (img / 255.).astype(np.float32)
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
    print(embedded[i])
#save_embedded_vectors_to_csv(csvfile)
#save_embedded_vectors_to_pkl(pklfile)
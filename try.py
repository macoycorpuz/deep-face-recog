import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os.path
from sklearn.metrics import f1_score, accuracy_score

from lib.align import AlignDlib
from lib.model import create_model
from lib.evhelper import save_embedded_vectors_to_csv, save_embedded_vectors_to_pkl

import csv
import pickle
import pandas as pd
from sklearn.decomposition import PCA

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
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

########### Mat plotting functions ###########

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))

def show_aligned_images():
    # Show original image
    plt.subplot(131)
    plt.imshow(jc_orig)

    # Show original image with bounding box
    plt.subplot(132)
    plt.imshow(jc_orig)
    plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

    # Show aligned image
    plt.subplot(133)
    plt.imshow(jc_aligned)

def show_embedded_vectors():
    show_pair(2, 3)
    show_pair(2, 12)

def show_embedded_vectors_from_csv(idx1, idx2):
    ev1 = []
    ev2 = []
    with open(evfile, 'r') as csvfile:
        embedded_vectors = csv.reader(csvfile, delimiter=' ', quotechar='|')
        ctr = 0
        for row in embedded_vectors:
            if ctr == idx1:
                for vector in row:
                    ev1.append(float(vector))
            if ctr == idx2:
                for vector in row:
                    ev2.append(float(vector))
            ctr += 1
        ev1 = np.array(ev1)
        ev2 = np.array(ev2)
        print(ev1)
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(ev1, ev2):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))

def show_embedded_vectors_from_pkl(filename):
    pkl_file = open(filename, 'rb')
    output = pickle.load(pkl_file)
    pkl_file.close()
    print(output)


################ Main() ################
evfile = 'embedded.csv'
evfilepkl = 'embedded.pkl'
metadata = load_metadata('images')
# nn4_small2_pretrained = create_model()
# nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

# # Initialize the OpenFace face alignment utility
# alignment = AlignDlib('models/landmarks.dat')

# # Load an image of Jacques Chirac
# jc_orig = load_image(metadata[2].image_path())

# # Detect face and return bounding box
# bb = alignment.getLargestFaceBoundingBox(jc_orig)

# # Transform image using specified face landmark indices and crop image to 96x96
# jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# embedded = np.zeros((metadata.shape[0], 128))
# for i, m in enumerate(metadata):
#     img = load_image(m.image_path())
#     img = align_image(img)
#     # scale RGB values to interval [0,1]
#     img = (img / 255.).astype(np.float32)
#     # obtain embedding vector for image
#     embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


######### Save Embedded Vectors #########
#save_embedded_vectors_to_csv(evfile, embedded)
#save_embedded_vectors_to_pkl(evfilepkl, embedded)

############ OUTPUT #################
#show_aligned_images()
#show_embedded_vectors()
#show_distance_threshold()
show_embedded_vectors_from_csv(2, 3)
#show_embedded_vectors_from_pkl(evfilepkl)
#plt.show()

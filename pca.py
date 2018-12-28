import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os.path
from model import create_model
import cv2

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

def get_distance_from_csv(idx1, idx2):
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
    print(distance(ev1,ev2))

evfile = 'labeled_ev.csv'
metadata = load_metadata('images')
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
df = pd.read_csv(evfile, delimiter=' ', quotechar='|', header=None)
df.dropna(how="all", inplace=True)
data = df.ix[:,0:8].values

z_scaler = StandardScaler()
z_data = z_scaler.fit_transform(data)
pca_trafo = PCA().fit(z_data);

pca_trafo = PCA(n_components=2)
pca_data = pca_trafo.fit_transform(z_data)
pca_inv_data = pca_trafo.inverse_transform(np.eye(2))

# get_distance(0,1)
# print(distance(pca_data[0], pca_data[1]))

#########################################
example_idx = 29

# img = load_image('putin.jpeg')
# img = align_image(img)
# img = (img / 255.).astype(np.float32)
# embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

# plt.figure(figsize=(8,3))
# plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
# plt.subplot(121)
# plt.imshow(load_image('putin.jpeg'))
# plt.subplot(122)
# plt.imshow(load_image(metadata[idx2].image_path()))
# plt.show()
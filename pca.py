import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os.path
from model import create_model
import cv2
from align import AlignDlib
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

#Save embedded vector to pkl function
def save_embedded_vectors_to_pkl(filename, embedded_vectors):
    output = open(filename, 'wb')
    pickle.dump(embedded_vectors, output)
    output.close()
    print("Embedded Vectors successfully saved to " + filename)

#Save embedded vector to csv function
def save_embedded_vectors_to_csv(filename, embedded_vectors):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img_vector in embedded_vectors:
            spamwriter.writerow(img_vector)
        print("Embedded Vectors successfully saved to " + filename)

## Load Files
evfile = 'embedded-vectors/labeled_ev.csv'
metadata = load_metadata('images')

## Read CSV file
df = pd.read_csv(evfile, delimiter=' ', quotechar='|', header=None)
df.dropna(how="all", inplace=True)
data = df.values

## Transform to PCA data
z_scaler = StandardScaler()
z_data = z_scaler.fit_transform(data)
pca_trafo = PCA(n_components=2)
pca_data = pca_trafo.fit_transform(z_data)

## Save pca filew
csvfile = 'pca_labeled_ev.csv'
pklfile = 'pca_labeled_ev.pkl'
save_embedded_vectors_to_csv(csvfile, pca_data)
save_embedded_vectors_to_pkl(pklfile, pca_data)

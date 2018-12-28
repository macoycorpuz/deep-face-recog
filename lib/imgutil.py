import cv2
import numpy as np
import os.path
import csv
import pickle
import numpy as np
from  lib.align import AlignDlib

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

def align_image(alignment, img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

#Get distance of to embedded files
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

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

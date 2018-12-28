import bz2
import os
import numpy as np
from urllib.request import urlopen
from lib.imgutil import load_metadata, load_image, align_image, save_embedded_vectors_to_csv, save_embedded_vectors_to_pkl
from lib.model import create_model
from lib.align import AlignDlib

def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()
    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)
    print("Landmarks Downloaded. *note: if not working, download manually (shape_predictor_68_face_landmarks.dat)")

def download_embedded_vectors(csvf, pklf):
    metadata = load_metadata('images')
    alignment = AlignDlib('models/landmarks.dat')
    nn4 = create_model()
    nn4.load_weights('weights/nn4.small2.v1.h5')
    embedded = np.zeros((metadata.shape[0], 128))
    for i, m in enumerate(metadata):
        img = load_image(m.image_path())
        img = align_image(alignment, img)
        img = (img / 255.).astype(np.float32)
        embedded[i] = nn4.predict(np.expand_dims(img, axis=0))[0]

    save_embedded_vectors_to_csv(csvf, embedded)
    save_embedded_vectors_to_pkl(pklf, embedded)

dir_models = 'models'
dir_embedded_vectors = 'embedded-vectors'
landmark_file = 'landmarks.dat'
csv_file = 'labeled_ev.csv'
pkl_file = 'labeled_ev.pkl'

landmark_file = os.path.join(dir_models, landmark_file)
csv_file = os.path.join(dir_embedded_vectors, csv_file)
pkl_file = os.path.join(dir_embedded_vectors, pkl_file)

if not os.path.exists(dir_models): 
    os.makedirs(dir_models)
if not os.path.exists(dir_embedded_vectors): 
    os.makedirs(dir_embedded_vectors)
if not os.path.exists(landmark_file):
    download_landmarks(landmarkfile)

download_embedded_vectors(csv_file, pkl_file)

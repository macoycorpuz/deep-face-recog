import bz2
import os
from urllib.request import urlopen
from lib.evhelper import save_embedded_vectors_to_csv, save_embedded_vectors_to_pkl
from lib.imglib import load_metadata, load_image, align_image
from lib.model import create_model

def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()
    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)
    print("Landmarks Downloaded. *note: if not working, download manually (shape_predictor_68_face_landmarks.dat)")

def download_embedded_vectors(**file):
    metadata = load_metadata('images')
    nn4_small2_pretrained = create_model()
    nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
    embedded = np.zeros((metadata.shape[0], 128))
    for i, m in enumerate(metadata):
        img = load_image(m.image_path())
        img = align_image(img)
        img = (img / 255.).astype(np.float32)
        embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
    
    save_embedded_vectors_to_csv(file["csvfile"], embedded)
    save_embedded_vectors_to_pkl(file["pklfile"], embedded)
    print("Embedded Vectors downloaded")



dst_dir = 'models'
landmarkfile = os.path.join(dst_dir, 'landmarks.dat')
dst_file = { 
    "pklfile":os.path.join(dst_dir, 'embedded.pkl'), 
    "csvfile":os.path.join(dst_dir, 'embedded.csv')
    }

if not os.path.exists(dst_dir): 
    os.makedirs(dst_dir)
if not os.path.exists(landmarkfile):
    #download_landmarks(landmarkfile)
    download_embedded_vectors(**dst_file)
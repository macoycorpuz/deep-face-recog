import lib.imgutil as iu
from lib.model import create_model
from plot_img import show_pair

metadata = iu.load_metadata('images')
nn4 = create_model()

img1 = iu.load_image(metadata[0].image_path())
img2 = iu.load_image(metadata[1].image_path())
distance = 1
identity = 'Arnold'
show_pair(img1, img2, distance, identity)
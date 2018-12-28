import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_pair(img1, img2, distance, identity):
    plt.figure(figsize=(8,3))
    plt.title(f'Recognized as {identity}')
    plt.suptitle(f'Distance = {distance}')
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()
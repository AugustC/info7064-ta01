import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def get_images(directory = 'images/'):
    images_dir = Path(directory)
    colored_imgs = []
    images = []
    for file_path in images_dir.glob('*.jpg'):
        img = cv2.imread(str(file_path))
        resized = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        colored_imgs.append(resized)
        gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        images.append(gray_img)
    return colored_imgs, images

def show_images(images):
    for img in images:
        plt.imshow(img, cmap='gray')
        plt.show()
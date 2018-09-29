import numpy as np
from cv2 import cv2
import cv_utils as u
from matplotlib import pyplot as plt


def main():

    image_paths = [
        'images/prac03ex04img01.png',
        'images/prac03ex04img02.png',
        'images/prac03ex04img03.png',
        'images/prac03ex02img01.jpg'
    ]

    for path in image_paths:
        image = u.read_color(path)

        u.show_cv_image(image)

        blobs = u.apply_blob_detector(image)

        blob_image = u.draw_blob_circles(image, blobs)

        u.show_cv_image(blob_image)


if __name__ == '__main__':
    main()

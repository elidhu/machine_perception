import os

from cv2 import cv2
import numpy

import cv_utils as utils

INPUT_IMAGE = 'images/prac02ex03img01.jpg'

def main(img_path):
    """performs median filtering as per prac2

    :param img: the path to an image
    :type img: str
    """

    # read in the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    utils.show('before filter', image)

    # apply a median filter to the image of size k = 5
    filtered = cv2.medianBlur(image, 5)
    utils.show('after filter', filtered)


if __name__ == '__main__':
    main(INPUT_IMAGE)
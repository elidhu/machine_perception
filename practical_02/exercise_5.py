import os

from cv2 import cv2
import numpy

import cv_utils as utils

INPUT_IMAGE = 'images/prac02ex05img01.png'

def main(img_path):
    """Using OpenCV’s histogram equalization method equalizeHist, write a
    program that improves the contrast of an input image. Visually inspect the
    output image and comment how histogram equalization changes its contrast.
    Compute the histograms of the input and equalized images and comment how the
    histogram has been stretched out more widely and uniformly.

    :param img: the path to an image
    :type img: str
    """

    # load in the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # show the image
    utils.show('original', image)

    # look at the histogram of the original image
    utils.show_hist_gray(image)

    # equalise the histogram
    equalised = cv2.equalizeHist(image)

    # show the new image
    utils.show('equalised', equalised)

    # look at the histogram now
    utils.show_hist_gray(equalised)

if __name__ == '__main__':
    main(INPUT_IMAGE)
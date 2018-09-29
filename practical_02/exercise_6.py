import os

from cv2 import cv2
import numpy as np

import cv_utils as utils

INPUT_IMAGE_1 = 'images/prac02ex06img01.png'
INPUT_IMAGE_2 = 'images/prac02ex06img01.png'

def erode(img_path):
    """
    :param img: the path to an image
    :type img: str
    """

    # load in the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    erosion = cv2.erode(image, kernel, iterations=1) 

    output = np.hstack((image, erosion))

    cv2.imshow('eroded', output)
    cv2.waitKey(0)

    return erosion


def dilate(img_path):
    """
    :param img: the path to an image
    :type img: str
    """

    # load in the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    dilation = cv2.dilate(image, kernel, iterations=1) 

    output = np.hstack((image, dilation))

    cv2.imshow('dilated', output)
    cv2.waitKey(0)

    return dilation


def opening(img_path):
    """
    :param img: the path to an image
    :type img: str
    """

    # load in the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) 

    output = np.hstack((image, opening))

    cv2.imshow('opening', output)
    cv2.waitKey(0)


def closing(img_path):
    """
    :param img: the path to an image
    :type img: str
    """

    # load in the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) 

    output = np.hstack((image, closing))

    cv2.imshow('closing', output)
    cv2.waitKey(0)

def morphological_gradient(img_path):
    """
    :param img: the path to an image
    :type img: str
    """

    # load in the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    output = np.hstack((image, gradient))

    cv2.imshow('morphological gradient', output)
    cv2.waitKey(0)

def blackhat(img_path):
    """
    :param img: the path to an image
    :type img: str
    """

    # load in the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))

    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    output = np.hstack((image, blackhat))

    cv2.imshow('morphological gradient', output)
    cv2.waitKey(0)

if __name__ == '__main__':
    # erode(INPUT_IMAGE_1)
    # dilate(INPUT_IMAGE_2)
    # opening(INPUT_IMAGE_1)
    # closing(INPUT_IMAGE_2)
    # morphological_gradient(INPUT_IMAGE_1)
    # test = dilate(INPUT_IMAGE_1) - erode(INPUT_IMAGE_1)
    # cv2.imshow('manual', test)
    # cv2.waitKey(0)
    blackhat('images/j_rev.png')
    pass
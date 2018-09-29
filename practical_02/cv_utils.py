import matplotlib
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


def show(title, img):
    """show a opencv2 window that can be destroyed with any key

    :param title: title of the window to create
    :type title: str
    :param img: opencv2 image to show
    :type img: the loaded image object
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)


def show_hist_gray(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

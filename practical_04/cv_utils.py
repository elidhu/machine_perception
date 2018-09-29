import matplotlib
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import os
import math


def gray_to_rgb(image):
    """convert cv2 image from gray to rgb for sane plotting in mpl.

    :param image: the image to be converted
    :type image: cv2 image
    :return: converted image
    :rtype: cv2 image
    """
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def bgr_to_rgb(image):
    """convert form cv2s silly BGR order for sane plotting in mpl.

    :param img: the list of images to be converted
    :type img: cv2 image
    :return: list of the converted images
    :rtype: list[cv2 image]
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr_to_gray(image):
    """convert form cv2s silly BGR order for sane plotting in mpl.

    :param img: the list of images to be converted
    :type img: cv2 image
    :return: list of the converted images
    :rtype: list[cv2 image]
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def read_gray32(img_path):
    """read in the image from the path as a grayscale and convert to float32.

    :param img_path: path to the image
    :type img_path: string
    :return: the loaded image
    :rtype: cv2 image
    """
    # load the image and convert to float32
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.float32(img)

    return img


def read_gray(img_path):
    """read in the image from the path as a grayscale image.

    :param img_path: path to the image
    :type img_path: string
    :return: the loaded image
    :rtype: cv2 image
    """
    # load the image and convert to float32
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return img


def read_color(img_path):
    """read in the image from the path as BGR color.

    :param img_path: path to the image
    :type img_path: string
    :return: the loaded image
    :rtype: cv2 image
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    return img


def save_mpl_subplot(plot, filename):
    """save the mpl subplot with dpi high enough that lines are not lost!

    :param plot: the mpl plot
    :type plot: mpl plot
    :param filename: the name to give the file including dir
    :type filename: str
    """

    plot.savefig(filename, dpi=300)


def create_mpl_subplot(images, color=True):
    """create mpl subplot with all images in list.

    even when the color is set to false it still seems to

    :param images: the list of images to plot
    :type images: cv2 image
    :param color: whether to plot in color or grayscale, defaults to True
    :type color: boolean
    :return: the complete plot
    :rtype: mpl plot
    """
    if not color:
        plt.set_cmap('gray')

    n = math.ceil(math.sqrt(len(images)))
    i = 1

    for img in images:
        plt.subplot(n, n, i)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        i += 1

    return plt


def create_mpl_histogram_gray(img):
    """create mpl histogram from the supplied grayscale image.

    :param img: the loaded image
    :type img: cv2 image
    :return: the computed histogram
    :rtype: mpl plot
    """

    plt.hist(img.ravel(), 256, [0, 256])
    return plt


def create_mpl_histogram_color(img, order=('r', 'g', 'b')):
    """create mpl histogram from the supplied image.

    this can deal with bgr, or rgb images.

    :param img: the loaded image
    :type img: cv2 image
    :param order: the channel ordering, defaults to ('r', 'g', 'b')
    :type order: tuple, optional
    :return: the computed histogram
    :rtype: mpl plot
    """
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    chans = cv2.split(img)

    # loop over the image channels
    for (chan, color) in zip(chans, order):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    return plt

# when convolving with this kernel the sign of the output is irrelevant!
# it merely tells us that one side is much more than the other side indicating
# an edge. note that the 'x' kernel has zeroes down the middle so it will only
# detect vertical edges and not horizontal and the 'y' does the opposite


def get_sobel_kernel(axis):
    """return the sobel kernal for the requested axis.

    :param axis: either 'x' or 'y' depending on which sobel kernel is wanted
    :type axis: string
    """
    sobel = {
        'x': np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3, 3),
        'y': np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape(3, 3)
    }

    return(sobel[axis])


def flip_kernel(kernel):
    """flip a kernel (probably to prep for convolution).

    :param img: the kernel
    :type img: matrix
    """
    k = np.copy(kernel)

    return(cv2.flip(k, -1))


def convolve(image, kernel):
    """convolve the image with the kernel.

    :param img: the image to convolve
    :type img: cv2 image
    :param kernel: the kernel to convolve with
    :type kernel: matrix
    """
    img = np.copy(image)

    return(cv2.filter2D(img, -1, flip_kernel(kernel)))


def normalise(image):
    """normalise an images values.

    :param img: the loaded image in grayscale
    :type img: cv2 image
    :return: copy of the normalised image
    :rtype: cv2 image
    """
    img = np.copy(image)

    maximum = np.max(img)

    img = np.absolute(img)

    img = img * (255.0 / maximum)

    return img


def apply_harris_corners(image):
    """run the harris corner detection function and dilate to increase point
       size.

    :param img: the loaded image
    :type img: cv2 image
    :return: copy of the image with the corner detections
    :rtype: cv2 image
    """

    # copy the image
    img = np.copy(image)

    # convert to gray before corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get the corners
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    corners = cv2.dilate(corners, None)

    # put the corners on the image
    img[corners > 0.01 * corners.max()] = [0, 0, 255]

    return img


def apply_shi_tomasi_corners(image):
    """apply the shi_tomasi corner detection algorithm.

    :param img: the loaded image
    :type img: cv2 image
    :return: copy of the image with detected corners marked
    :rtype: cv2 image
    """

    # copy the image
    img = np.copy(image)

    # convert to gray before corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # employ the shi-tomasi corner algorithm to detect corners
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.05, 20)

    # for each corner draw it on the original image!
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 5, [0, 0, 255], -1)

    return img


def apply_gaussian_blur(image, n, border=0):
    """apply a gaussian blur to the image.

    :param img: the loaded image
    :type img: cv2 image
    :param n: the size of the kernel n x n
    :type n: int
    :param border: what to do when the kernel overlaps the border, defaults to 0
    :param border: int, optional
    :return: copy of the blurred image
    :rtype: cv2 image
    """
    img = np.copy(image)

    return cv2.GaussianBlur(img, (n, n), border)


def apply_sobel_filter(image, axis):
    """apply the sobel filter along 1 axis.

    :param image: the loaded image
    :type image: cv2 image
    :param axis: the axis of the sobel filter, 'x' or 'y'
    :type axis: string
    :return: copy of the filtered image
    :rtype: cv2 image
    """
    img = np.copy(image)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert to float32 so we don't lose the negative values
    img = np.float32(img)

    # get the kernel
    kernel = get_sobel_kernel(axis)

    # convolve the kernel and the image
    filtered = convolve(img, kernel)

    return filtered


def apply_auto_canny(image, epsilon=0.33):
    """apply the Canny edge detector with automatic thresholding.

    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

    :param image: the input grayscale input image
    :type image: cv2 image
    :param epsilon: threshold modifier, defaults to 0.33
    :param epsilon: float, optional
    """
    img = np.copy(image)
    # compute the median of the single channel pixel intensities
    median = np.median(img)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - epsilon) * median))
    upper = int(min(255, (1.0 + epsilon) * median))
    edged = cv2.Canny(img, lower, upper)

    # return the edged image
    return edged


def draw_hough_lines(image, lines):
    """draw the hough lines onto the image for visualisation

    :param lines: array of points that describe lines x1, y1, x2, y2
    :type lines: numpy array
    :param image: the image that the lines were calculated from
    :type image: cv2 image
    :return: the image with the hough lines drawn
    :rtype: cv2 image
    """
    img = np.copy(image)

    # draw the lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return img


def apply_blob_detector(image):
    """apply the simple blob detection algorithm

    :param image: the image to do do blob detection on
    :type image: cv2 image
    :return: the blobs from the image
    :rtype: cv2 blobs
    """

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 100
    params.thresholdStep = 5

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 300

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.75

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    blobs = detector.detect(image)

    return blobs


def draw_blob_circles(image, blobs):
    """draws the blobs from the detection on to the image for visualisation

    https://www.learnopencv.com/blob-detection-using-opencv-python-c/

    :param blobs: the blobs from cv2s blob detector
    :type blobs: cv2 blobs
    :param image: the image to draw the blobs onto
    :type image: cv2 image
    :return: the image with the blobs as circles
    :rtype: cv2 image
    """
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img = cv2.drawKeypoints(image, blobs, np.array(
        []), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img


# image file stuff
EXTENSIONS = ['jpg', 'jpeg', 'png']


def get_images_in_dir(d):
    """get the relative path to all of the images in the given directory

    :param d: the directory to look in relative to the running process
    :type d: str
    :return: all of the image paths
    :rtype: list[str]
    """
    # list to store the paths to the images
    paths = []

    # get all the files in the directory
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        if os.path.isfile(full_path):
            paths.append(full_path)

    # filter out the files that don't have typeical image extensions
    image_paths = [f for f in paths if f.split('.')[-1] in EXTENSIONS]

    return image_paths

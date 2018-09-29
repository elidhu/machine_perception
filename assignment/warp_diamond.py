from collections import OrderedDict
from math import hypot

import numpy as np
from cv2 import cv2
from scipy.spatial import distance as dist

from mp_utils import convert, features, filters, io, utils, vis

# these are in RGB order
COLORS = OrderedDict({
    "Orange": (185, 89, 89),
    "Red": (138, 36, 47),
    "Green": (51, 98, 70),
    "White": (194, 195, 205),
    "Blue": (47, 70, 133),
    "Yellow": (191, 139, 52),
    "Black": (48, 58, 60)
})


@utils.timing
def get_contours(image_paths):

    # not important, for grouping images and debugging...
    i = 0

    # process each image
    for path in image_paths:
        image = io.read_color(path)
        gray = convert.bgr_to_gray(image)

        # threshold the image
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 47, 1)
        # blur the image
        blurred = filters.apply_gaussian_blur(thresh, 9)
        bil = cv2.bilateralFilter(blurred, 11, 60, 60)

        # detect the edges
        canny = features.apply_canny(bil, 175, 190)

        # find the contours
        cnts = features.apply_find_contours(canny, 4, 80000, 5)

        # NOW EXTRACT THE DIAMOND
        # this is what I want the diamond to be
        # dst = np.array([[250, 0], [0, 250],  [250, 500],
        #                 [500, 250]], dtype=np.float32)

        dst = np.array([[0, 0], [0, 500],  [500, 500],
                        [500, 0]], dtype=np.float32)

        # TODO: make it so that you mask the contour and only so it removes
        # the background

        j = 0
        for c in cnts:
            # draw the contours
            r = c.reshape(4, 2)
            r = np.float32(r)

            # USING THE CONTOURS CREATE A MASK AND REMOVE THE BACKGROUND
            mask = np.zeros(image.shape, np.uint8)
            mask = features.draw_contours(mask, [c], (255, 255, 255))
            masked = cv2.bitwise_and(image, mask)

            # get the transform and warp the image
            M = cv2.getPerspectiveTransform(r, dst)
            max_x = np.max(dst[:, 0])
            max_y = np.max(dst[:, 1])
            warp = cv2.warpPerspective(masked, M, (max_x, max_y))

            # removing intesity form the LAB image
            l, a, b = cv2.split(cv2.cvtColor(warp, cv2.COLOR_BGR2LAB))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            lab_img = cv2.merge([l, a, b])
            rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
            ### colors and stuff
            # create LAB colors
            lab = np.zeros((len(COLORS), 1, 3), dtype="uint8")

            thresh = cv2.inRange(convert.bgr_to_lab(
                rgb_img), (55, 0, 0), (217, 255, 255))
            # thresh = cv2.inRange(images.bgr_to_lab(
            #     rgb_img), (10, 10, 10), (245, 245, 245))
            # cv2.imshow('lel', thresh)

            color_names = []
            for i, (name, rgb) in enumerate(COLORS.items()):
                lab[i] = rgb
                color_names.append(name)

            print(lab.shape)

            lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)

            mask_conts = []

            # create masks so that we are looking at the colors in specific areas
            # mask_conts.append(np.array(
            #     [[106, 196], [106, 206], [116, 206], [116, 196]], dtype=np.int32))
            # mask_conts.append(np.array(
            #     [[394, 196], [394, 206], [384, 206], [384, 196]], dtype=np.int32))

            mask_conts.append(np.array(
                [[45, 50], [45, 430], [50, 430], [50, 55], [430, 55], [430, 50]], dtype=np.int32))

            top_half_mask = np.zeros((500, 500), np.uint8)
            top_half_mask = features.draw_contours(
                top_half_mask, mask_conts, (255, 255, 255))
            top_half_mask = cv2.bitwise_and(top_half_mask, thresh)
            mean = cv2.mean(lab_img, top_half_mask)[:3]
            vis.show(lab_img)
            print(mean)

            # go through our colors and check distance, shortest distance is the color
            min_dist = (np.inf, None)
            # loop over the known L*a*b* color values
            for (i, row) in enumerate(lab):
                # compute the distance between the current L*a*b*
                # color value and the mean of the image
                d = dist.euclidean(row[0], mean)

                # if the distance is smaller than the current distance,
                # then update the bookkeeping variable
                if d < min_dist[0]:
                    min_dist = (d, i)

            color = color_names[min_dist[1]]
            print(color)

            vis.show(top_half_mask)
            vis.show(rgb_img)

            j += 1


def distance(p, q):
    """calculate the distance between two points
    
    :param p: point 1
    :type p: tuple
    :param q: point 2
    :type q: tuple
    :return: the distance
    :rtype: float
    """

    return hypot(p[0] - q[0], p[1] - q[1])


def fetch_rgb(img):

    rgb_list = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            red = img[y, x, 2]
            blue = img[y, x, 0]
            green = img[y, x, 1]
            print(red, green, blue)  # prints to command line
            strRGB = str(red) + "," + str(green) + "," + str(blue)
            rgb_list.append([red, green, blue])
            cv2.imshow('original', img)

    cv2.imshow('original', img)
    cv2.setMouseCallback("original", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    print(rgb_list)

    return rgb_list


if __name__ == '__main__':
    image_paths = []
    image_paths = utils.get_images_in_dir('./sample_data/SetD')

    get_contours(image_paths)

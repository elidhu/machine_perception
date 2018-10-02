import concurrent.futures
import json
import os
from collections import OrderedDict
from math import hypot

import numpy as np
import pytesseract
from cv2 import cv2
from PIL import Image

from conf import (BOT_COLOR_CNTS, CLASS_CNTS, CLASS_DICTIONARY, COLOR_NAMES,
                  COLOR_VALUES, LABEL_CNTS, LABEL_DICTIONARY, TOP_COLOR_CNTS,
                  XFORM_DESTINATION, SYMBOL_LOOKUP)
from mp_utils import (convert, features, filters, io, ocr, transforms, utils,
                      vis, orb)


def process(path):
    descriptions = []

    fname, ext = utils.get_fname_and_ext(path)
    descriptions.append(fname + ext)

    # read in the image as bgr
    color_img = io.read_color(path)

    # make a copy of the image in grayscale
    gray_img = convert.bgr_to_gray(color_img)

    #* FIND DIAMONDS
    # get the locations of the diamonds
    diamonds = features.get_diamonds(gray_img)

    for diamond in diamonds:
        # to hold the description for this diamond
        description = {}

        # isolate the current diamond from in the original image
        isolated = features.isolate_diamond(color_img, diamond)

        # transform the diamond to a given position
        extracted = transforms.xform(
            isolated, convert.contour_to_ptarray(diamond), XFORM_DESTINATION)
        
        #* EXTRACT COLORS
        #TODO: employ some sort of white balancing and maybe kNN to improve color detection
        normalised = filters.normlise_intensity(extracted)
        # normalised = filters.white_balance(extracted)

        # remove white and black by creating a threshold mask
        thresh = cv2.inRange(convert.bgr_to_lab(
            normalised), (70, 0, 0), (200, 255, 255))

        # create the masks for the areas where the color will be extracted
        top_color_mask = features.mask_from_contours(
            TOP_COLOR_CNTS, thresh.shape)
        bot_color_mask = features.mask_from_contours(
            BOT_COLOR_CNTS, thresh.shape)
        # and the mask with the thresh output, hopefully stop the blacks and
        # whites messing with the average color
        top_color_mask = utils.combine_masks(top_color_mask, thresh)
        bot_color_mask = utils.combine_masks(bot_color_mask, thresh)

        # get the top color
        color = features.get_color(
            normalised, top_color_mask, COLOR_NAMES, COLOR_VALUES)
        description['top'] = color

        # get the bottom color
        color = features.get_color(
            normalised, bot_color_mask, COLOR_NAMES, COLOR_VALUES)
        description['bottom'] = color

        balanced = filters.white_balance(extracted)

        #* EXTRACT CLASS

        class_loc = utils.crop_to_contour(balanced, CLASS_CNTS)

        clas = ocr.extract_all_words(class_loc, filter='NUMS')

        label = ocr.determine_label(clas, CLASS_DICTIONARY)

        description['class'] = label

        #* EXTRACT LABEL
        # crop to the general area of the text
        label_loc = utils.crop_to_contour(balanced, LABEL_CNTS)

        # find contours to isolate the areas text could be
        cnts, highlighted, found = ocr.find_text(label_loc)

        # if there were areas of text to investigate
        if found:
            # optimise the contours found to only include the important ones
            bbox = ocr.find_optimal_components_subset(cnts, highlighted)

            # crop to the optimal height, keep the entire width
            cropped = utils.crop_to_bbox(label_loc, bbox, ignore='x')

            # extract the words from the image
            words = ocr.extract_all_words(cropped, filter='CAPS')

            # determine the correct label from the words
            label = ocr.determine_label(words, LABEL_DICTIONARY)

            # try again with a larger crop
            if label is '(none)':
                cropped = utils.crop_to_bbox(label_loc, bbox, padding=(13, 13), ignore='x')
                words = ocr.extract_all_words(cropped)
                label = ocr.determine_label(words, LABEL_DICTIONARY)

            # add the label to the description
            description['text'] = label
        # expand the crop and try again
        else:
            # add the label to the description
            description['text'] = '(none)'

        #* EXTRACT SYMBOL
        symbols = utils.get_images_in_dir('./symbols')
        grab = utils.crop_to_bbox(extracted, (0, 0, 500, 250))

        # init to hold the best prediciton
        prediction = ('(none)', np.inf)

        for s in symbols:
            s_img = io.read_color(s)
            matches = orb.quick_orb(grab, s_img)
            distances = [m.distance for m in matches]
            score = sum(distances[:4])
            if score == 0:
                score = np.inf
            if score < prediction[1]:
                prediction = (s, score)

        # get the filename of the symbol
        fname = utils.get_fname(prediction[0])
        # lookup the symbol name from the filename and add it to the description
        symbol = SYMBOL_LOOKUP[fname]
        description['symbol'] = symbol

        #* APPEND DESCRIPTION
        descriptions.append(description)
        # print(json.dumps(description, indent=4))
        # vis.show(balanced)

    return(descriptions)

@utils.timing
def main(image_paths, threaded=True):
    if threaded:
        desc = []
        # Create a pool of processes. By default, one is created for each CPU in your machine.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for d in executor.map(process, image_paths):
                desc += d
    else:
        desc = []

        for image in image_paths:
            desc += process(image)

    for i in desc:
        if isinstance(i, dict):
            for k, v in i.items():
                print('{}: {}'.format(k, v))
            print()
        else:
            print(i)

    test = []
    for i in desc:
        if isinstance(i, dict):
            test.append(i)

    # dirty testing
    from test import compare, actual

    compare(test, actual)

if __name__ == '__main__':

    # get the image paths
    image_paths = utils.get_images_in_dir('./sample_data/SetD')

    # run the main method
    main(image_paths, threaded=False)

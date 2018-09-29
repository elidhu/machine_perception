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
                  XFORM_DESTINATION)
from mp_utils import (convert, features, filters, io, ocr, transforms, utils,
                      vis)


@utils.timing
def main(image_paths):

    descriptions = []
    
    # loop through each image
    for path in image_paths:
        # read in the image as bgr
        color_img = io.read_color(path)

        # make a copy of the image in grayscale
        gray_img = convert.bgr_to_gray(color_img)

        #* FIND DIAMONDS
        # get the locations of the diamonds
        diamonds = features.get_diamonds(gray_img)

        for diamond in diamonds:
            # to hold the description for this diamond
            description = OrderedDict()

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
            top_color_mask = utils.mask_from_contours(
                TOP_COLOR_CNTS, thresh.shape)
            bot_color_mask = utils.mask_from_contours(
                BOT_COLOR_CNTS, thresh.shape)
            # and the mask with the thresh output, hopefully stop the blacks and
            # whites messing with the average color
            top_color_mask = utils.combine_masks(top_color_mask, thresh)
            bot_color_mask = utils.combine_masks(bot_color_mask, thresh)

            # get the top color
            color = features.get_color(
                normalised, top_color_mask, COLOR_NAMES, COLOR_VALUES)
            description['TOP COLOR'] = color

            # get the bottom color
            color = features.get_color(
                normalised, bot_color_mask, COLOR_NAMES, COLOR_VALUES)
            description['BOTTOM COLOR'] = color

            #* EXTRACT LABEL
            balanced = filters.white_balance(extracted)
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
                if label is 'NONE':
                    cropped = utils.crop_to_bbox(label_loc, bbox, padding=(13, 13), ignore='x')
                    words = ocr.extract_all_words(cropped)
                    label = ocr.determine_label(words, LABEL_DICTIONARY)
                    # print('Changed label from NONE to {}'.format(label))

                # add the label to the description
                description['LABEL'] = label
            # expand the crop and try again
            else:
                # add the label to the description
                description['LABEL'] = 'NONE'

            #* EXTRACT CLASS
            # vis.show(balanced)

            class_loc = utils.crop_to_contour(balanced, CLASS_CNTS)

            clas = ocr.extract_all_words(class_loc, filter='NUMS')

            label = ocr.determine_label(clas, CLASS_DICTIONARY)
            # print(clas)
            # vis.show(class_loc)

            description['CLASS'] = label

            # vis.show(class_loc)
            # cnts, highlighted, found = ocr.find_text(class_loc)
            # if found:
                # optimise the contours found to only include the important ones
                # bbox = ocr.find_optimal_components_subset(cnts, highlighted)

                # crop to the optimal height, keep the entire width
                # cropped = utils.crop_to_bbox(class_loc, bbox, padding=(30, 30))

                # classes = ocr.extract_all_words(cropped, filter='NUMS')

                # print(classes)

                # vis.show(cropped)



            #* EXTRACT SYMBOL

            # # Initiate ORB detector
            # orb = cv2.ORB_create()
            # # find the keypoints with ORB
            # kp = orb.detect(g_label,None)
            # # compute the descriptors with ORB
            # kp, des = orb.compute(g_label, kp)
            # # draw only keypoints location,not size and orientation
            # img2 = cv2.drawKeypoints(label, kp, None, color=(0,255,0), flags=0)
            # vis.show(img2)

            # min_points = 4
            # min_area = 20
            # max_area = 3000
            # candidates = []

            # _, cnts, _ = cv2.findContours(
            #     thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # for c in cnts:
            #     (x, y, w, h) = cv2.boundingRect(c)
            #     ar = w / float(h)

            #     # check the number of points and the area
            #     if ar > 0.5 and ar < 2.5:
            #         if (w > 5 and w < 100) and (h > 10 and h < 200):
            #             candidates.append(c)

            # img = features.draw_contours(label, candidates)
            # vis.show(img)

            # vis.show(thresh)

            # config = ('-l eng --oem 1 --psm 6')
            # text = pytesseract.image_to_string(label, config=config)
            # print(text)

            # vis.show(out)

            # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 9))

            # tophat = cv2.morphologyEx(convert.bgr_to_gray(out), cv2.MORPH_TOPHAT, rectKernel)

            # vis.show(tophat)

            # append the image description
            # print(json.dumps(description, indent=4))

            descriptions.append(description)
    
    print(json.dumps(descriptions, indent=4))
    print(len(descriptions))


if __name__ == '__main__':
    image_paths = utils.get_images_in_dir('./sample_data/SetD')
    main(image_paths)

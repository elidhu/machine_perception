from collections import OrderedDict
import numpy as np
from cv2 import cv2

COLOR_NAMES = ['orange', 'red', 'green', 'white', 'blue', 'yellow', 'black']
COLOR_VALUES = np.array([[[89,  89, 185]],
                         [[47,  36, 138]],
                         [[70,  98,  51]],
                         [[205, 195, 194]],
                         [[133,  70,  47]],
                         [[52, 139, 191]],
                         [[60,  58,  48]]], dtype=np.uint8)

XFORM_DESTINATION = np.array([[250, 0], [0, 250],  [250, 500],
                              [500, 250]], dtype=np.float32)

TOP_COLOR_CNTS = [np.array(
    [[250, 45], [55, 240], [60, 235], [250, 55], [440, 235], [445, 240]], dtype=np.int32)]

BOT_COLOR_CNTS = [np.array(
    [[250, 455], [55, 260], [60, 265], [250, 445], [440, 265], [445, 260]], dtype=np.int32)]

LABEL_CNTS = [np.array(
    [[50, 170], [50, 335], [450, 335], [450, 170]], dtype=np.int32)]

CLASS_CNTS = [np.array(
    [[190, 347], [190, 457], [310, 457], [310, 347]], dtype=np.int32)]

LABEL_DICTIONARY = {
    'FLAMMABLE GAS': [
        'FLAMMABLE',
        'AMMABLE',
        'GAS'
    ],
    'NON-FLAMMABLE GAS': [
        'NONFLAMMABLE',
        'CNONFLAMMABLE',
        'GAS'
    ],
    'FLAMMABLE LIQUID': [
        'FLAMMABLE',
        'LIQUID'
    ],
    'POISON': [
        'POISON'
    ],
    'OXIDIZER': [
        'OXIDIZER',
        'IDIZER'
    ],
    'DANGEROUS WHEN WET': [
        'DANGEROUS WHEN WET',
        'DANGEROUS',
        'WHEN',
        'WET',
        'EROUS'
    ],
    'FUEL OIL': [
        'FUEL',
        'OIL'
    ],
    'TOXIC': [
        'TOXIC'
    ],
    'INHALATION HAZARD': [
        'INHALATION',
        'HAZARD'
    ],
    'CORROSIVE': [
        'CORROSIVE',
        'ORROSIVE'
    ],
    'GASOLINE': [
        'GASOLINE'
    ],
    'COMBUSTIBLE': [
        'COMBUSTIBLE'
    ],
    'RADIOACTIVE III': [
        'RADIOACTIVE III',
        'RADIOACTIVE',
        'III',
    ],
    'RADIOACTIVE II': [
        'RADIOACTIVE II',
        'RADIOACTIVE',
        'II'
    ],
    'SPONTANEOUSLY COMBUSTIBLE': [
        'SPONTANEOUSLY COMBUSTIBLE',
        'SPONTANEOUSLY'
    ],
    'ORGANIC PEROXIDE': [
        'ORGANIC PEROXIDE',
        'ORGANIC',
        'PEROXIDE'
    ],
    'OXYGEN': [
        'OXYGEN'
    ],
    'BLASTING AGENTS': [
        'BLASTING',
        'AGENTS'
    ],
    'EXPLOSIVES': [
        'EXPLOSIVES'
    ]
}

CLASS_DICTIONARY = {
    '2': [
        '2',
        '27',
        '2.'
    ],
    '6': [
        '6'
    ],
    '8': [
        '8',
    ],
    '4': [
        '4'
    ],
    '5.2': [
        '5.2',
        '5',
        '32'
    ],
    '5.1': [
        '5.1',
        '1'
    ],
    '3': [
        '3',
        '37'
    ],
    '7': [
        '7'
    ]
}

SYMBOL_LOOKUP = {
    '1_4': '1.4',
    '1_5': '1.5',
    '1_6': '1.6',
    'explosion': 'EXPLOSION',
    'corrosive': 'CORROSIVE',
    'flame_black': 'FLAME',
    'flame_white': 'FLAME',
    'gas_cylinder': 'GAS CYLINDER',
    'oxidizer': 'OXIDIZER',
    'radioactive': 'RADIOACTIVE',
    'skull_crossbones_black': 'SKULL AND CROSSBONES ON BLACK DIAMOND',
    'skull_crossbones': 'SKULL AND CROSSBONES'
}

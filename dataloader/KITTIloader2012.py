import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, log, split_file):
    left_fold = 'colored_0/'
    right_fold = 'colored_1/'
    disp_noc = 'disp_occ/'

    image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]
    if split_file is None:
        random.shuffle(image)
        train = image[:]
        val = image[160:]
    else:
        with open(split_file) as f:
            vallist = sorted([int(x.strip()) for x in f.readlines() if len(x) > 0])
        val = ['{:06d}_10.png'.format(x) for x in vallist]
        train = [x for x in image if x not in val]
    log.info(val)


    left_train = [os.path.join(filepath, left_fold, img) for img in train]
    right_train = [os.path.join(filepath, right_fold, img) for img in train]
    disp_train = [os.path.join(filepath, disp_noc, img) for img in train]

    left_val = [os.path.join(filepath, left_fold, img) for img in val]
    right_val = [os.path.join(filepath, right_fold, img) for img in val]
    disp_val = [os.path.join(filepath, disp_noc, img) for img in val]

    return left_train, right_train, disp_train, left_val, right_val, disp_val

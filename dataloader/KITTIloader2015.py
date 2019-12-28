import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, log, split_file):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    disp_R = 'disp_occ_1/'

    image = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

    all_index = np.arange(200)
    if split_file is None:
        np.random.shuffle(all_index)
        vallist = all_index[:40]
    else:
        with open(split_file) as f:
            vallist = sorted([int(x.strip()) for x in f.readlines() if len(x) > 0])
    log.info(vallist)
    val = ['{:06d}_10.png'.format(x) for x in vallist]
    train = [x for x in image if x not in val]

    left_train = [os.path.join(filepath, left_fold, img) for img in train]
    right_train = [os.path.join(filepath, right_fold, img) for img in train]
    disp_train_L = [os.path.join(filepath, disp_L, img) for img in train]
    # disp_train_R = [filepath+disp_R+img for img in train]

    left_val = [os.path.join(filepath, left_fold, img) for img in val]
    right_val = [os.path.join(filepath, right_fold, img) for img in val]
    disp_val_L = [os.path.join(filepath, disp_L, img) for img in val]
    # disp_val_R = [filepath+disp_R+img for img in val]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L

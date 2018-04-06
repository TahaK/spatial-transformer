from scipy.misc import imresize
from scipy.ndimage import rotate
import random
import pickle
from skimage import exposure
import os
import numpy as np


def square_image(image, rectroi, mask):
    axis = np.argmax(mask.shape)
    diff = mask.shape[axis] - mask.shape[abs(1 - axis)]
    crop_mask = np.array([])
    crop_im = np.array([])
    crop_rect = np.array([])
    if ((rectroi[axis] - diff) > 0): 
        if (axis == 1): 
            crop_im = image[:][diff:]
            crop_mask = mask[:][diff:]
            crop_rect = rectroi - [0, diff, 0, diff]
        else:
            crop_im = image[diff:][:]
            crop_mask = mask[diff:][:]
            crop_rect = rectroi - [diff, 0, diff, 0]
    else:
        extra = diff - rectroi[axis]
        if (axis == 1):
            crop_im = image[:][rectroi[axis]:]
            crop_mask = mask[:][rectroi[axis]:]
            crop_rect = rectroi - [0, rectroi[axis], 0, rectroi[axis]]

            crop_im = crop_im[:][:-extra]
            crop_mask = crop_mask[:][:-extra]
            crop_rect = crop_rect - [0, 0, 0, extra]
        else:
            crop_im = image[rectroi[axis]:][:]
            crop_mask = mask[rectroi[axis]:][:]
            crop_rect = rectroi - [rectroi[axis], 0, rectroi[axis], 0]

            crop_im = crop_im[:-extra][:]
            crop_mask = crop_mask[:-extra][:]
            crop_rect = crop_rect - [0, 0, extra, 0]
    return crop_im, crop_rect, crop_mask


# TODO: resize rectroi as well
def resize(image, rectroi, mask, size=299):
    return imresize(image, (size, size), interp='bicubic'), rectroi, imresize(mask, (size, size), interp='nearest')


def rotate_image(image, mask=[5,5]):
    angle = random.randrange(-5, 5)
    return rotate(image, angle, reshape=False), rotate(mask, angle, reshape=False), angle


def translate_image(image, mask):
    x_trans = random.randrange(-10, 10)
    y_trans = random.randrange(-10, 10)
    mask_t = mask 
    image_t = image
    if x_trans >= 0:
        image_t[:x_trans][:] = 0
        mask_t[:x_trans][:] = 0
    else:
        image_t[x_trans:][:] = 0
        mask_t[:x_trans][:] = 0
    if y_trans >= 0:
        image_t[:][:y_trans] = 0
        mask_t[:][:y_trans] = 0
    else:
        image_t[:][y_trans:] = 0
        mask_t[y_trans:][:] = 0
    return image_t,  mask_t


def histogram_normalization(image):
    return exposure.equalize_hist(image)


def dynamic_normalization(image):
    lmin = float(image.min())
    lmax = float(image.max())
    return (image - lmin) / (lmax - lmin)


def choose_random_data_from_folder(folder):
    pass


# Parse L1 L2 and L3 labels from the generic label
def parse_label(label):
    l1 = ''
    l2 = ''
    l3 = ''
    if label == 'normal':
        return 'normal', 'normal', 'normal'
    else:
        l1 = 'abnormal'
        if 'B' in label:
            l2 = 'B'
            if 'B1' in label:
                l3 = 'B_1'
            elif 'B2' in label:
                l3 = 'B_2'
            else:
                l3 = 'B_3'
        else:
            l2 = 'A'
            if 'A1' in label:
                l3 = 'A_1'
            elif 'A2' in label:
                l3 = 'A_2'
            else:
                l3 = 'A_3'
    return l1, l2, l3


# Writes output image to the appropriate folder
def write_prep_data(img, output_path):

    # TODO: test non if-else
    file_pi = open(os.path.join(output_path, img.L3, img.name + '.pickle'),'wb')
    pickle.dump(img,file_pi)
    file_pi.close()
    #if img.label1 == 'normal':
       # file_pi = open(os.path.join(output_path, 'normal', img.name), 'wb')
       # pickle.dump(img, file_pi)
        #file_pi.close()
    #elif img.label3 == 'A_1':
     #   file_pi = open(os.path.join(output_path, 'A_1', img.name), 'wb')
     #   pickle.dump(img, file_pi)
     #   file_pi.close()
    #elif img.label3 == 'A_2':
    #    file_pi = open(os.path.join(output_path, 'A_2', img.name), 'wb')
    #    pickle.dump(img, file_pi)
    #    file_pi.close()
    #elif img.label3 == 'A_3':
    #    file_pi = open(os.path.join(output_path, 'A_3', img.name), 'wb')
    #    pickle.dump(img, file_pi)
    #    file_pi.close()
    #elif img.label3 == 'B_1':
    #    file_pi = open(os.path.join(output_path, 'B_1', img.name), 'wb')
    #    pickle.dump(img, file_pi)
    #    file_pi.close()
    #elif img.label3 == 'B_2':
    #    file_pi = open(os.path.join(output_path, 'B_2', img.name), 'wb')
    #    pickle.dump(img, file_pi)
    #    file_pi.close()
    #elif img.label3 == 'B_3':
    #    file_pi = open(os.path.join(output_path, 'B_3', img.name), 'wb')
    #    pickle.dump(img, file_pi)
    #    file_pi.close()

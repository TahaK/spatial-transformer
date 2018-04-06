import numpy as np
import random
import math
from create_imbalance import create_imbalance
from scipy.ndimage import rotate, zoom

def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    top = (zh - h) // 2
    left = (zw - w) // 2
    out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
    trim_top = ((out.shape[0] - h) // 2)
    trim_left = ((out.shape[1] - w) // 2)
    out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    return out

def apply_augment(roll,data_point):
    if roll == 0:
        angle = random.randrange(-5, 5)
        return rotate(data_point, angle=angle, reshape=False)
    elif roll == 1:
        x_trans = random.randrange(-5, 5)
        y_trans = random.randrange(-5, 5)
        image_t = data_point
        if x_trans >= 0:
            image_t[:x_trans][:] = 0
        else:
            image_t[x_trans:][:] = 0
        if y_trans >= 0:
            image_t[:][:y_trans] = 0
        else:
            image_t[:][y_trans:] = 0
        return image_t
    else:
        order = random.randint(100,160)/100
        return np.abs(clipped_zoom(data_point,order))

def augment_data(data,percentage):
    original_length = data.shape[0]
    augm = []
    while (original_length*(100-percentage)/percentage) > len(augm):
        random.seed()
        i = random.randint(0,original_length-1)
        roll = random.sample(range(0,3),2)
        augd = apply_augment(roll[1],apply_augment(roll[0],data[i]))
        augm.append(augd)
    return augm

def create_augmented(X_imbalance,y_imbalance,imbalance_list):
    #takes the imbalanced list and balance it.
    imbalance_length = [1,5,10,20,50]

    y_augmented = y_imbalance
    x_shaped = np.reshape(X_imbalance,(-1,60,60))

    counter = 0

    for i in imbalance_list:
        x_imb = x_shaped[np.where(y_imbalance==i)[0].tolist()]
        x_aug = augment_data(x_imb,imbalance_length[counter])
        x_shaped = np.append(x_shaped,np.array(x_aug),axis=0)
        y_augmented = np.append(y_augmented,np.zeros(len(x_aug))+i)
        counter += 1
                       
    X_augmented = np.reshape(x_shaped,(-1,3600))
    y_augmented = np.reshape(y_augmented,(y_augmented.shape[0],1))
    y_augmented = y_augmented.astype(int)
    return X_augmented,y_augmented
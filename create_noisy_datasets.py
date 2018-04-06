import numpy as np
import skimage
from scipy.io import loadmat

def read_data():
    data_set = loadmat('./data/cluttered-mnist.mat')

    training_set = np.swapaxes(np.swapaxes(data_set['x_tr'],1,2),0,3)
    training_set = np.reshape(training_set,(-1,60,60))
    y_training = (data_set['y_tr'])
    y_training = np.reshape(y_training,(-1))

    test_set = np.swapaxes(np.swapaxes(data_set['x_ts'],1,2),0,3)
    test_set = np.reshape(test_set,(-1,60,60))
    y_test = data_set['y_ts']
    y_test = np.reshape(y_test,(-1))

    validation_set = np.swapaxes(np.swapaxes(data_set['x_vl'],1,2),0,3)
    validation_set = np.reshape(validation_set,(-1,60,60))
    y_validation = data_set['y_vl']
    y_validation = np.reshape(y_validation,(-1))

    return training_set,y_training,test_set,y_test,validation_set,y_validation



def sp_noise(image,prob):
    output = np.zeros(image.shape,np.float32)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output


def create_noisy_background(data_set):
    # adds 3 different noises, and returns 3 different data_sets.
    digit_locations = np.where(data_set != 0)
    digit_values = data_set[digit_locations]

    pois_noise = np.empty((data_set.shape[0],data_set.shape[1]))#,data_set.shape[2]]
    gaus_noise = np.empty((data_set.shape[0],data_set.shape[1]))#,data_set.shape[2]]
    gaus_noise_2 = np.empty((data_set.shape[0],data_set.shape[1]))#,data_set.shape[2]]
    papp_noise = np.empty((data_set.shape[0],data_set.shape[1]))#,data_set.shape[2]]

    for i in range(data_set.shape[0]):
        #pois_noise = np.random.poisson(lam = data_set,size=None)
        pois_noise[i] = skimage.util.random_noise(data_set[i],mode = "poisson") #TODO: Figure out how to increase noise

        #gaus_noise = np.clip( np.random.normal(0.0, 0.01, data_set.shape) + data_set,0, 1) 
        gaus_noise[i] = skimage.util.random_noise(data_set[i],mode = "gaussian", mean=0.7, var = 0.1)

        gaus_noise_2[i] = skimage.util.random_noise(data_set[i],mode = "gaussian", mean=0.3, var = 0.1)

        #papp_noise =  sp_noise(data_set,0.1)
        papp_noise[i] = skimage.util.random_noise(data_set[i],mode = "s&p",salt_vs_pepper = 0.95)

    pois_noise[digit_locations] = digit_values #Remove the noise from image pixels.
    gaus_noise[digit_locations] = digit_values #Remove the noise from image pixels.        
    gaus_noise_2[digit_locations] = digit_values
    papp_noise[digit_locations] = digit_values #Remove the noise from image pixels.


    return pois_noise,gaus_noise,gaus_noise_2, papp_noise
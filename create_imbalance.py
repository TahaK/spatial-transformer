def create_imbalance(X_train,y_train,X_valid,y_valid):

	#takes the MNIST list, imbalance random class, returns the list of class sorted to most to less, imbalanced set and labels.

	import numpy as np
	import random
	import math

	X_imbalance = X_train
	y_imbalance = y_train

	X_valid_imbalance = X_valid
	y_valid_imbalance = y_valid

	imbalance_list = random.sample(range(10),k = 5) 
	imbalance_length = [1,5,10,20,50]

	counter = 0
	for i in imbalance_list:
		x_pop = random.sample(np.where(y_imbalance==i)[0].tolist(), k=math.floor(len(np.where(y_imbalance==i)[0])*(100-imbalance_length[counter])/100))
		X_imbalance = np.delete(X_imbalance,x_pop,axis = 0)
		y_imbalance = np.delete(y_imbalance,x_pop,axis = 0)
		counter +=1
		
	return X_imbalance,y_imbalance,imbalance_list

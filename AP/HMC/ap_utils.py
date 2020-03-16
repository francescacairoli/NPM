import numpy as np
from numpy.random import rand
from math import pi

def set_params():

	params = {'dim': 6, 'time_bound': 240,'ranges': np.array([[13.5*4,13.5*8],[13.5*4,13.5*8],[0,2000],[0,2000],[0,2000],[0,2000]])}

	return params


def rand_state():

	params = set_params()
	ranges = params["ranges"]
	
	rand_state = ranges[:,0]+(ranges[:,1]-ranges[:,0])*rand(ranges.shape[0])
	return rand_state



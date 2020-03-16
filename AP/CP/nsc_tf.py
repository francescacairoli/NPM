import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Input

def create_model(input_dim):
	n_input = input_dim
	n_hidden = 10
	n_output = 1
	model = Sequential()
	model.add(Dense(n_hidden, input_dim=n_input, kernel_initializer='normal', activation='tanh'))
	model.add(Dense(n_hidden, activation='tanh'))
	model.add(Dense(n_hidden, activation='tanh'))
	model.add(Dense(n_output, activation='sigmoid'))
	return model

  
#Build your network
def train(x_train, y_train, n_epochs = 5000, batch_size = None, weight_init = None):
	
	model = create_model(x_train.shape[1])
	model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])
	model.fit(x_train, y_train, epochs = n_epochs, batch_size = x_train.shape[0], verbose = 0)

	return model


def evaluate(model, x_test, y_test):
	# Evaluate performance on test data
	res = model.evaluate(x_test, y_test)
	print("Model results: ", res )

	return res

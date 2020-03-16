import scipy.io
import numpy as np


def load_data(train_file_path, test_file_path):
	n_classes = 2

	training_set = scipy.io.loadmat(train_file_path) 
	test_set = scipy.io.loadmat(test_file_path)

	x_train = (training_set["X"].T).astype(np.float32)
	n_training_points = x_train.shape[0]

	y_train = training_set["T"].reshape((n_training_points,)).astype(np.int32)

	x_test = (test_set["X"].T).astype(np.float32)
	n_test_points = x_test.shape[0]
	y_test = test_set["T"].reshape((n_test_points,)).astype(np.int32)

	return x_train, y_train, x_test, y_test


def split_train_calibration(full_x_train, full_y_train, split_rate = 0.7):
	n_full_training_points = full_x_train.shape[0]
	perm = np.random.permutation(n_full_training_points)
	split_index = int(n_full_training_points*split_rate)

	x_train = full_x_train[perm[:split_index]]
	y_train = full_y_train[perm[:split_index]]

	x_cal = full_x_train[perm[split_index:]]
	y_cal = full_y_train[perm[split_index:]]

	return x_train, y_train, x_cal, y_cal

def split_train_calibration_w_indexes(full_x_train, full_y_train, split_rate = 0.7):
	n_full_training_points = full_x_train.shape[0]
	perm = np.random.permutation(n_full_training_points)
	split_index = int(n_full_training_points*split_rate)

	x_train = full_x_train[perm[:split_index]]
	y_train = full_y_train[perm[:split_index]]

	x_cal = full_x_train[perm[split_index:]]
	y_cal = full_y_train[perm[split_index:]]

	return x_train, y_train, x_cal, y_cal, perm

def double_split_train_calibration(full_x_train, full_y1_train, full_y2_train, split_rate = 0.7):
	n_full_training_points = full_x_train.shape[0]
	perm = np.random.permutation(n_full_training_points)
	split_index = int(n_full_training_points*0.7)

	x_train = full_x_train[perm[:split_index]]
	y1_train = full_y1_train[perm[:split_index]]
	y2_train = full_y2_train[perm[:split_index]]

	x_cal = full_x_train[perm[split_index:]]
	y1_cal = full_y1_train[perm[split_index:]]
	y2_cal = full_y2_train[perm[split_index:]]

	return x_train, y1_train, y2_train, x_cal, y1_cal, y2_cal



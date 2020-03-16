from conformal_predictions import *
import time
from sklearn import svm
import ap_utils as ap
import matlab.engine
import matlab

def compute_calibration_uncertainties(nsc, x_cal, y_cal):

	cal_pred_probs = nsc.predict(x_cal)
	alphas, _ = compute_NN_nonconformity_scores(y_cal, cal_pred_probs)
	cal_conf_cred = compute_calibration_p_values(alphas, cal_pred_probs)

	return cal_conf_cred, cal_pred_probs, alphas


def compute_test_uncertainties(nsc, alphas, x):
	pred_probs = nsc.predict(x)
	conf_cred = compute_p_values(alphas, pred_probs)

	return conf_cred, pred_probs


def label_correct_incorrect_pred(predicted_class, real_class):
	n_points = len(predicted_class)
	error_indexes = np.ones(n_points)
	for j in range(n_points):
		if real_class[j] != predicted_class[j]:
			error_indexes[j] = -1
	return error_indexes 


def train_svc_rejection_rule(kernel_type, freq_unc, error_indexes):

	start = time.time()
	wclf = svm.SVC(kernel=kernel_type, gamma='scale', class_weight='balanced', verbose=False, tol = 1e-10, decision_function_shape='ovo')#max_iter=100000,
	wclf.fit(freq_unc, error_indexes) 
	print("Time required to train the SVC rejetion rule: ", time.time()-start)

	return wclf


def apply_svc_rejection_rule(trained_svc, freq_unc):
	return trained_svc.predict(freq_unc)


def active_label_query(number_points, scaler, alphas, nsc, rej_rule, phase):
	
	params = ap.set_params()
	dim = params["dim"]
	time_bound = params["time_bound"]
	# Pool of number_points input points, randomly selected
	x_samples = np.zeros((number_points, dim))
	for i in range(number_points):
		x_samples[i, :] = ap.rand_state()

	x_samples_scaled = scaler.transform(x_samples)

	# NSC prob prediction on these samples
	samples_unc, samples_pred_probs = compute_test_uncertainties(nsc, alphas, x_samples_scaled)

	acc_rej_samples = apply_svc_rejection_rule(rej_rule, samples_unc)
	x_rej = x_samples[(1-acc_rej_samples).astype(bool)]#.reshape((number_points,))
	x_rej_scaled = x_samples_scaled[(1-acc_rej_samples).astype(bool)]
	print('Number of points to add: {}\n'.format(x_rej.shape[0]))

	start = time.time()

	# CALLING MATLAB FUNCTION
	eng = matlab.engine.start_matlab()
	x_add, y_add = eng.gen_labels(matlab.double((x_rej.T).tolist()),matlab.double([time_bound]), nargout = 2)
	eng.quit()
	#x_add = np.array(x_add).T
	y_add = np.array(y_add[0])


	print("STAGING: uncertain points labeled in {} seconds".format(time.time()-start))

	return x_rej_scaled, y_add #scaler.transform(x_add)

def passive_label_query(number_points, scaler):
	
	params = ap.set_params()
	dim = params["dim"]
	time_bound = params["time_bound"]
	# Pool of number_points input points, randomly selected
	x_samples = np.zeros((number_points, dim))
	for i in range(number_points):
		x_samples[i, :] = ap.rand_state()

	x_samples_scaled = scaler.transform(x_samples)
	start = time.time()
	# CALLING MATLAB FUNCTION
	eng = matlab.engine.start_matlab()
	x_add, y_add = eng.gen_labels(matlab.double((x_samples.T).tolist()),matlab.double([time_bound]), nargout = 2)
	eng.quit()
	#x_add = np.array(x_add).T
	y_add = np.array(y_add[0])


	print("PASSIVE STAGING: uncertain points labeled in {} seconds".format(time.time()-start))

	return x_samples_scaled, y_add #scaler.transform(x_add)

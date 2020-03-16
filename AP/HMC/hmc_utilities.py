import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from edward.models import Empirical, Categorical, Normal, Bernoulli, BernoulliWithSigmoidProbs
import edward as ed
from tqdm import tqdm
from tqdm import trange
import os
import scipy
import scipy.io
from sklearn import preprocessing
import h5py
from datasets_utilities import *
import gc
import pickle
from sklearn import svm
import ap_utils as ap
import time
import matlab.engine
import matlab
#from conformal_predictions import *
from sklearn.base import BaseEstimator, clone
from sklearn.gaussian_process import GaussianProcessClassifier 
from sklearn.gaussian_process.kernels import RBF, CompoundKernel,ConstantKernel, DotProduct
from scipy.linalg import cholesky, cho_solve, solve
import scipy.optimize
from scipy.special import erf, expit
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import check_random_state
from sklearn.utils.optimize import _check_optimize_result
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from operator import itemgetter


def compute_avg_std_pred_probs(x_input, weights_lists):
    wfc1_lst, bfc1_lst, wfc2_lst, bfc2_lst, wfc3_lst, bfc3_lst, wfc4_lst, bfc4_lst = weights_lists

    n_points = x_input.shape[0]
    n_samples = len(wfc1_lst)
    pred_probs_lst = np.zeros((n_samples,n_points))
        
    for i in range(n_samples):
        h_fc1 = tf.tanh(tf.matmul(np.float64(x_input), wfc1_lst[i]) + bfc1_lst[i])
        h_fc2 = tf.tanh(tf.matmul(h_fc1, wfc2_lst[i]) + bfc2_lst[i])
        h_fc3 = tf.tanh(tf.matmul(h_fc2, wfc3_lst[i]) + bfc3_lst[i])
        pred_probs = tf.sigmoid(tf.matmul(h_fc3, wfc4_lst[i]) + bfc4_lst[i])
        pred_classes = tf.round(pred_probs)
        pred_probs_lst[i] = (pred_probs.eval()).reshape((n_points,))
        
    avg_pred_probs = np.mean(pred_probs_lst, axis=0)
    std_pred_probs = np.std(pred_probs_lst, axis=0)
    avg_pred_classes = np.round(avg_pred_probs)

    return avg_pred_probs, std_pred_probs, avg_pred_classes

def label_correct_incorrect_pred(avg_pred_class, real_class):
    n_points = len(avg_pred_class)
    error_indexes = np.ones(n_points)
    for j in range(n_points):
        if real_class[j] != avg_pred_class[j]:
            error_indexes[j] = -1
    return error_indexes

BALANCED = False

LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712,128.12323805, -2010.49422654])[:, np.newaxis]

class _myBinaryGaussianProcessClassifierLaplace(BaseEstimator):
	def __init__(self, kernel=None, optimizer="fmin_l_bfgs_b",n_restarts_optimizer=0, max_iter_predict=100,warm_start=False, copy_X_train=True, random_state=None):
		self.kernel = kernel
		self.optimizer = optimizer
		self.n_restarts_optimizer = n_restarts_optimizer
		self.max_iter_predict = max_iter_predict
		self.warm_start = warm_start
		self.copy_X_train = copy_X_train
		self.random_state = random_state

	def fit(self, X, y):
		if self.kernel is None:  # Use an RBF kernel as default
			self.kernel_ = ConstantKernel(1.0, constant_value_bounds="fixed")* RBF(1.0, length_scale_bounds="fixed")
		else:
			self.kernel_ = clone(self.kernel)

		self.rng = check_random_state(self.random_state)
		self.X_train_ = np.copy(X) if self.copy_X_train else X
		# Encode class labels and check that it is a binary classification problem
		label_encoder = LabelEncoder()
		self.y_train_ = label_encoder.fit_transform(y)
		self.classes_ = label_encoder.classes_
		if self.optimizer is not None and self.kernel_.n_dims > 0:
			# Choose hyperparameters based on maximizing the log-marginal likelihood (potentially starting from several initial values)
			if BALANCED:
				def obj_func(theta, eval_gradient=True):
					if eval_gradient:
						lml, grad = self.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=False)
						return -lml, -grad
					else:
						return -self.log_marginal_likelihood(theta, clone_kernel=False)
			else:
				def obj_func(theta, eval_gradient=False): # if unbalanced negative class
					#print("SON ENTRATO QUI!!---------------------")
					K = self.kernel_(self.X_train_)
					_, (self.pi_, self.W_sr_, self.L_, _, _) = self._posterior_mode(K, return_temporaries=True)
					f = K.T.dot(self.y_train_ - self.pi_)
					pred_class = np.where(f > 0, self.classes_[1], self.classes_[0])
					
					n_points = (self.X_train_).shape[0]
					TN = 0 #Number of times an observation is correctly predicted as negative class
					TN_FP = 0 # Total number of actual negative class observations
					for i in range(n_points):
						if self.y_train_[i]==0: # neg class
							TN_FP += 1
							if pred_class[i]<0:
								TN += 1
					return -(TN/TN_FP)
				
			# First optimize starting from theta specified in kernel
			optima = [self._constrained_optimization(obj_func, self.kernel_.theta, self.kernel_.bounds)]
			# Additional runs are performed from log-uniform chosen initial theta
			if self.n_restarts_optimizer > 0:
				if not np.isfinite(self.kernel_.bounds).all():
					raise ValueError("Multiple optimizer restarts (n_restarts_optimizer>0) requires that all bounds are finite.")
				bounds = self.kernel_.bounds
				for iteration in range(self.n_restarts_optimizer):
					theta_initial = np.exp(self.rng.uniform(bounds[:, 0], bounds[:, 1]))
					optima.append(self._constrained_optimization(obj_func, theta_initial, bounds))
			# Select result from run with minimal (negative) log-marginal likelihood
			lml_values = list(map(itemgetter(1), optima))
			self.kernel_.theta = optima[np.argmin(lml_values)][0]
			self.log_marginal_likelihood_value_ = -np.min(lml_values)
		else:
			self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(self.kernel_.theta)

		K = self.kernel_(self.X_train_)
		_, (self.pi_, self.W_sr_, self.L_, _, _) = self._posterior_mode(K, return_temporaries=True)

		return self
	
	def predict(self, X):
		check_is_fitted(self)
		K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
		f_star = K_star.T.dot(self.y_train_ - self.pi_)  # Algorithm 3.2,Line 4
		return np.where(f_star > 0, self.classes_[1], self.classes_[0])

	def predict_proba(self, X):
		check_is_fitted(self)
		K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
		f_star = K_star.T.dot(self.y_train_ - self.pi_)  # Line 4
		v = solve(self.L_, self.W_sr_[:, np.newaxis] * K_star)  # Line 5
		var_f_star = self.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
		alpha = 1 / (2 * var_f_star)
		gamma = LAMBDAS * f_star
		integrals = np.sqrt(np.pi / alpha)* erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS**2)))/ (2 * np.sqrt(var_f_star * 2 * np.pi))
		pi_star = (COEFS * integrals).sum(axis=0) + .5 * COEFS.sum()
		return np.vstack((1 - pi_star, pi_star)).T

	def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
		if theta is None:
			if eval_gradient:
				raise ValueError("Gradient can only be evaluated for theta!=None")
			return self.log_marginal_likelihood_value_

		if clone_kernel:
			kernel = self.kernel_.clone_with_theta(theta)
		else:
			kernel = self.kernel_
			kernel.theta = theta
		if eval_gradient:
			K, K_gradient = kernel(self.X_train_, eval_gradient=True)
		else:
			K = kernel(self.X_train_)

		Z, (pi, W_sr, L, b, a) = self._posterior_mode(K, return_temporaries=True)
		if not eval_gradient:
			return Z
		d_Z = np.empty(theta.shape[0])
		R = W_sr[:, np.newaxis] * cho_solve((L, True), np.diag(W_sr))
		C = solve(L, W_sr[:, np.newaxis] * K)  # Line 8
		s_2 = -0.5 * (np.diag(K) - np.einsum('ij, ij -> j', C, C))* (pi * (1 - pi) * (1 - 2 * pi))  # third derivative
		
		for j in range(d_Z.shape[0]):
			C = K_gradient[:, :, j]
			s_1 = .5 * a.T.dot(C).dot(a) - .5 * R.T.ravel().dot(C.ravel())
			b = C.dot(self.y_train_ - pi)
			s_3 = b - K.dot(R.dot(b))
			d_Z[j] = s_1 + s_2.T.dot(s_3)
			return Z, d_Z

	def _posterior_mode(self, K, return_temporaries=False):
		if self.warm_start and hasattr(self, "f_cached") and self.f_cached.shape == self.y_train_.shape:
			f = self.f_cached
		else:
			f = np.zeros_like(self.y_train_, dtype=np.float64)
	
		log_marginal_likelihood = -np.inf
		for _ in range(self.max_iter_predict):
			pi = expit(f)
			W = pi * (1 - pi)
			W_sr = np.sqrt(W)
			W_sr_K = W_sr[:, np.newaxis] * K
			B = np.eye(W.shape[0]) + W_sr_K * W_sr
			L = cholesky(B, lower=True)
			b = W * f + (self.y_train_ - pi)
			a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
			f = K.dot(a)
			lml = -0.5 * a.T.dot(f)-np.log1p(np.exp(-(self.y_train_ * 2 - 1) * f)).sum()- np.log(np.diag(L)).sum()
			if lml - log_marginal_likelihood < 1e-10:
				break
			log_marginal_likelihood = lml
		self.f_cached = f
		if return_temporaries:
			return log_marginal_likelihood, (pi, W_sr, L, b, a)
		else:
			return log_marginal_likelihood
	
	def _constrained_optimization(self, obj_func, initial_theta, bounds):
		if self.optimizer == "fmin_l_bfgs_b":
			if BALANCED:
				opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
			else:
				opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=False, bounds=bounds)
			
			_check_optimize_result("lbfgs", opt_res)
			theta_opt, func_min = opt_res.x, opt_res.fun
		elif callable(self.optimizer):
			theta_opt, func_min = aelf.optimizer(obj_func, initial_theta, bounds=bounds)
		else:
			raise ValueError("Unknown optimizer %s." % self.optimizer)
		return theta_opt, func_min

class myGaussianProcessClassifier(GaussianProcessClassifier):
	
	def fit(self, X, y):
		if self.kernel is None or self.kernel.requires_vector_input:
			X, y = check_X_y(X, y, multi_output=False, ensure_2d=True, dtype="numeric")
		else:
			X, y = check_X_y(X, y, multi_output=False, ensure_2d=False, dtype=None)
		self.base_estimator_ = _myBinaryGaussianProcessClassifierLaplace(self.kernel, self.optimizer, self.n_restarts_optimizer,self.max_iter_predict, self.warm_start, self.copy_X_train,self.random_state)
		self.classes_ = np.unique(y)
		self.n_classes_ = self.classes_.size

		self.base_estimator_.fit(X, y)
		self.log_marginal_likelihood_value_ = self.base_estimator_.log_marginal_likelihood()

		return self



def train_gpc_rejection_rule(kernel_type, unc_measures, error_indexes):
	# unc_measures shape = (n_samples, n_features)
	start = time.time()
	n_points = len(error_indexes)
	print("Positive class ratio: ", np.sum((error_indexes==1))/n_points)
	print("Negative class ratio: ", np.sum((error_indexes==-1))/n_points)
	kernel = kernel_type #default
	# Laplace approx and kernel hyperparam tuning
	gpc = myGaussianProcessClassifier(kernel = kernel).fit(unc_measures.T, error_indexes)
	print("Time required to train the GPC: ", time.time()-start)

	#print("TRAINED GPC PARAMETERS: ", gpc.get_params())
	predict_probs = gpc.predict_proba(unc_measures.T)
	
	#print("NEGATIVE CLASS ", predict_probs[(error_indexes==-1)])
	print("NEGATIVE PROB AVG ", np.mean(predict_probs[(error_indexes==-1)], axis=0))
	
	#print("POSITIVE CLASS ", predict_probs[(error_indexes==1)])
	print("POSITIVE PROB AVG ", np.mean(predict_probs[(error_indexes==1)], axis=0))

	return  gpc


def apply_gpc_rejection_rule(trained_gpc, new_unc_measures, decision_threshold = 0.5):
	predict_probs_pos_class = trained_gpc.predict_proba(new_unc_measures.T)[:,1]
	n_points = new_unc_measures.shape[1]
	pred_labels = np.zeros(n_points)
	for i in range(n_points):
		if predict_probs_pos_class[i]>decision_threshold:
			pred_labels[i] = 1
		else:
			pred_labels[i] = -1

	return pred_labels


def active_label_query(type_rej_rule, number_points, scaler, weights_lists, rej_rule, n_emp_samples, decision_threshold=None):
	
	params = ap.set_params()
	dim = params["dim"]
	time_bound = params["time_bound"]
	# Pool of number_points input points, randomly selected
	x_samples = np.zeros((number_points, dim))
	for i in range(number_points):
		x_samples[i, :] = ap.rand_state()
	x_samples_scaled = scaler.transform(x_samples)

	samples_avg_probs, samples_std_probs, samples_avg_pred_class = compute_avg_std_pred_probs(x_samples_scaled, weights_lists)
	samples_unc = np.vstack((samples_avg_probs, samples_std_probs))
	if type_rej_rule == "SVC":
		acc_rej_samples = apply_svc_rejection_rule(rej_rule, samples_unc)
	elif type_rej_rule == "GPC":
		acc_rej_samples = apply_gpc_rejection_rule(rej_rule, samples_unc, decision_threshold)
	else:
		print("Unknown type of rejection rule!")
	#x_rej = x_samples[(1-acc_rej_samples).astype(bool)]
	#x_rej_scaled = x_samples_scaled[(1-acc_rej_samples).astype(bool)]
	x_rej = x_samples[(acc_rej_samples<0)]
	x_rej_scaled = x_samples_scaled[(acc_rej_samples<0)]
	print('Number of points to add: {}\n'.format(x_rej_scaled.shape[0]))
	start = time.time()
	#rej_samples_avg_pred_class = samples_avg_pred_class[(1-acc_rej_samples).astype(bool)]
	#rej_samples_avg_probs = samples_avg_probs[(1-acc_rej_samples).astype(bool)]
	#rej_samples_std_probs = samples_std_probs[(1-acc_rej_samples).astype(bool)]
	rej_samples_avg_pred_class = samples_avg_pred_class[(acc_rej_samples<0)]
	rej_samples_avg_probs = samples_avg_probs[(acc_rej_samples<0)]
	rej_samples_std_probs = samples_std_probs[(acc_rej_samples<0)]
	
	rej_samples_unc = np.vstack((rej_samples_avg_probs, rej_samples_std_probs))
	# CALLING MATLAB FUNCTION
	eng = matlab.engine.start_matlab()
	x_add, y_add = eng.gen_labels(matlab.double((x_rej.T).tolist()),matlab.double([time_bound]), nargout = 2)
	eng.quit()
	#x_add = np.array(x_add).T
	y_add = np.array(y_add[0])

	print("STAGING: uncertain points labeled in {} seconds".format(time.time()-start))

	return x_rej_scaled, y_add, rej_samples_unc, rej_samples_avg_pred_class


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

	print("STAGING: uncertain points labeled in {} seconds".format(time.time()-start))

	return x_samples_scaled, y_add

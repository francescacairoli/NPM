import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import pickle

import tensorflow as tf
from keras.models import load_model

from dataset_utilities import *
import nsc_tf

from CP_utilities import *

SPLIT_RATE = 0.7

def run_original():
    

    # Load datasets
    full_x_train, full_y_train, x_test, y_test = load_data('../AP_20K_unif_train.mat', '../AP_10K_unif_test.mat')

    # shuffling data before splitting the dataset in train and calibration set
    x_train, y_train, x_cal, y_cal = split_train_calibration(full_x_train, full_y_train, split_rate = SPLIT_RATE)

    # Data standardization
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_cal_scaled = scaler.transform(x_cal)
    x_test_scaled = scaler.transform(x_test)


    dim = x_train.shape[1]
    n_test_points = x_test.shape[0]

    N_EPOCHS = 5000
    print("Number of epochs: ", N_EPOCHS)

    # Train the NN with the original training set
    nsc_orig = nsc_tf.train(x_train_scaled, y_train, n_epochs = N_EPOCHS) 
    nsc_tf.evaluate(nsc_orig, x_test_scaled, y_test)
    nsc_orig.save('Files/nsc_orig.h5')

    kernel_type = 'linear'
    print("KERNEL_TYPE: ", kernel_type)

    # preparing dataset to train the rejection rule
    cal_unc_orig, cal_pred_probs_orig, alphas_orig = compute_calibration_uncertainties(nsc_orig, x_cal_scaled, y_cal) # conf and cred
    cal_error_indexes_orig = label_correct_incorrect_pred(np.round(cal_pred_probs_orig), y_cal)

    rej_rule_orig = train_svc_rejection_rule(kernel_type, cal_unc_orig, cal_error_indexes_orig)

    test_unc_orig, test_pred_probs_orig = compute_test_uncertainties(nsc_orig, alphas_orig, x_test_scaled) # conf and cred
    test_error_indexes_orig = label_correct_incorrect_pred(np.round(test_pred_probs_orig), y_test)
    
    svc_test_pred_orig = apply_svc_rejection_rule(rej_rule_orig, test_unc_orig)
    
    svc_orig_accuracy = np.sum((svc_test_pred_orig==test_error_indexes_orig))/n_test_points
    n_test_errors_orig = np.sum((test_error_indexes_orig==-1))
    n_rej_points_orig = np.sum((svc_test_pred_orig==-1))
    print("SVC rejection rate ORIGINAL: ", 100*n_rej_points_orig/n_test_points, "%")
    print("SVC accuracy ORIGINAL: ", svc_orig_accuracy)

    recognized_orig = 0
    for i in range(n_test_points):
        if test_error_indexes_orig[i] == -1 and svc_test_pred_orig[i] == -1:
            recognized_orig += 1

    print("SVC recognized errors ORIGINAL: {}/{}".format(recognized_orig, n_test_errors_orig))

    return 0


number_trials = 5
for i in range(number_trials):
    print("DNN+CP+SVC original: Results of trial number ", i)
    run_original()

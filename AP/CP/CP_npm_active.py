import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import pickle

import tensorflow as tf
from keras.models import load_model

from dataset_utilities import *
import nsc_tf

from CP_utilities import *


def run_active(split_rate):
    
    # Load datasets
    full_x_train, full_y_train, x_test, y_test = load_data('../AP_20K_unif_train.mat', '../AP_10K_unif_test.mat')

    # shuffling data before splitting the dataset in train and calibration set
    x_train, y_train, x_cal, y_cal = split_train_calibration(full_x_train, full_y_train, split_rate = split_rate)

    # Data standardization
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_cal_scaled = scaler.transform(x_cal)
    x_test_scaled = scaler.transform(x_test)


    dim = x_train.shape[1]
    n_test_points = x_test.shape[0]

    N_EPOCHS = 5000
    print("Number of epochs: ", N_EPOCHS)
    start_orig = time.time()
    # Train the NN with the original training set
    nsc_orig = nsc_tf.train(x_train_scaled, y_train, n_epochs = N_EPOCHS) 
    nsc_tf.evaluate(nsc_orig, x_test_scaled, y_test)
    nsc_orig.save('Files/nsc_orig.h5')

    kernel_type = 'rbf'
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
    print("Time original phase: ", time.time()-start_orig)

    pool_size = 50000
    number_points_to_add = np.array([pool_size, 2*pool_size])

    print("::: Active learning starts ::::")
    start_active = time.time()
    # active learning function
    x_add_cal, y_add_cal = active_label_query(number_points_to_add[0], scaler, alphas_orig, nsc_orig, rej_rule_orig, 'refinement')
    print("STAGING: labels query for refinement completed.")
    
    # Refined calibration set
    x_cal_refined = np.vstack((x_cal_scaled, x_add_cal)) # scaled
    y_cal_refined = np.hstack((y_cal, y_add_cal))

    cal_unc_ref, cal_pred_probs_ref, alphas_ref = compute_calibration_uncertainties(nsc_orig, x_cal_refined, y_cal_refined) # conf and cred
    cal_error_indexes_ref = label_correct_incorrect_pred(np.round(cal_pred_probs_ref), y_cal_refined)

    rej_rule_ref = train_svc_rejection_rule(kernel_type, cal_unc_ref, cal_error_indexes_ref)

    x_add_act, y_add_act = active_label_query(number_points_to_add[1], scaler, alphas_ref, nsc_orig, rej_rule_ref, 'retrain')
    print("STAGING: labels query for retrain completed")
    
    n_active_points = len(y_add_act)
    n_retrain = int(np.round(n_active_points*split_rate))

    x_retrain = np.vstack((x_train_scaled, x_add_act[:n_retrain, :]))
    y_retrain = np.hstack((y_train, y_add_act[:n_retrain]))
    x_recal = np.vstack((x_cal_scaled, x_add_act[n_retrain:, :]))
    y_recal = np.hstack((y_cal, y_add_act[n_retrain:]))

    nsc_active = nsc_tf.train(x_retrain, y_retrain, n_epochs = 5000) 
    nsc_tf.evaluate(nsc_active, x_test_scaled, y_test)
    nsc_orig.save('Files/nsc_active.h5')

    cal_unc_active, cal_pred_probs_active, alphas_active = compute_calibration_uncertainties(nsc_active, x_recal, y_recal) # conf and cred
    cal_error_indexes_active = label_correct_incorrect_pred(np.round(cal_pred_probs_active), y_recal)

    rej_rule_active = train_svc_rejection_rule(kernel_type, cal_unc_active, cal_error_indexes_active)

    test_unc_active, test_pred_probs_active = compute_test_uncertainties(nsc_active, alphas_active, x_test_scaled) # conf and cred
    test_error_indexes_active = label_correct_incorrect_pred(np.round(test_pred_probs_active), y_test)
    
    svc_test_pred_active = apply_svc_rejection_rule(rej_rule_active, test_unc_active)
    
    svc_active_accuracy = np.sum((svc_test_pred_active==test_error_indexes_active))/n_test_points
    n_test_errors_active = np.sum((test_error_indexes_active==-1))
    n_rej_points_active = np.sum((svc_test_pred_active==-1))
    
    print("SVC rejection rate ACTIVE: ", 100*n_rej_points_active/n_test_points, "%")
    print("SVC accuracy ACTIVE: ", svc_active_accuracy)

    recognized_active = 0
    for i in range(n_test_points):
        if test_error_indexes_active[i] == -1 and svc_test_pred_active[i] == -1:
            recognized_active += 1

    print("SVC recognized errors ACTIVE: {}/{}".format(recognized_active, n_test_errors_active))
    print("Time active phase: ", time.time()-start_active)
    #n_active_points = len(y_add_act)
    #n_retrain = int(np.round(n_active_points*split_rate))
    passive_comparison = True
    if passive_comparison:
        start_passive = time.time()
        x_add_pass, y_add_pass = passive_label_query(n_active_points, scaler)
    
        x_retrain_pass = np.vstack((x_train_scaled, x_add_pass[:n_retrain, :]))
        y_retrain_pass = np.hstack((y_train, y_add_pass[:n_retrain]))
        x_recal_pass = np.vstack((x_cal_scaled, x_add_pass[n_retrain:, :]))
        y_recal_pass = np.hstack((y_cal, y_add_pass[n_retrain:]))

        nsc_passive = nsc_tf.train(x_retrain_pass, y_retrain_pass, n_epochs = 5000) 
        nsc_tf.evaluate(nsc_passive, x_test_scaled, y_test)
        nsc_orig.save('Files/nsc_passive.h5')

        cal_unc_passive, cal_pred_probs_passive, alphas_passive = compute_calibration_uncertainties(nsc_passive, x_recal_pass, y_recal_pass) # conf and cred
        cal_error_indexes_passive = label_correct_incorrect_pred(np.round(cal_pred_probs_passive), y_recal_pass)

        rej_rule_passive = train_svc_rejection_rule(kernel_type, cal_unc_passive, cal_error_indexes_passive)

        test_unc_passive, test_pred_probs_passive = compute_test_uncertainties(nsc_passive, alphas_passive, x_test_scaled) # conf and cred
        test_error_indexes_passive = label_correct_incorrect_pred(np.round(test_pred_probs_passive), y_test)
        
        svc_test_pred_passive = apply_svc_rejection_rule(rej_rule_passive, test_unc_passive)
        
        svc_passive_accuracy = np.sum((svc_test_pred_passive==test_error_indexes_passive))/n_test_points
        n_test_errors_passive = np.sum((test_error_indexes_passive==-1))
        n_rej_points_passive = np.sum((svc_test_pred_passive==-1))
        
        print("SVC rejection rate PASSIVE: ", 100*n_rej_points_passive/n_test_points, "%")
        print("SVC accuracy PASSIVE: ", svc_passive_accuracy)

        recognized_passive = 0
        for i in range(n_test_points):
            if test_error_indexes_passive[i] == -1 and svc_test_pred_passive[i] == -1:
                recognized_passive += 1

        print("SVC recognized errors PASSIVE: {}/{}".format(recognized_passive, n_test_errors_passive))
        print("Time passive phase: ", time.time()-start_passive)
    
    return 0


SPLIT_RATE = 0.7
number_trials = 1
for i in range(number_trials):
    print("DNN+CP+SVC ACTIVE: Results of trial number ", i)
    run_active(SPLIT_RATE)

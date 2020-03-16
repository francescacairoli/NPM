import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import pickle

import tensorflow as tf
from keras.models import load_model

from dataset_utilities import *
import nsc_tf

from CP_utilities import *

print("-------Running in Tensorflow-------")

#np.random.seed(seed=777)#260791
SPLIT_RATE = 0.7

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

train_net = True
N_EPOCHS = 10000
if train_net:
    # Train the NN with the original training set
    nsc_orig = nsc_tf.train(x_train_scaled, y_train, n_epochs = N_EPOCHS) 
    nsc_tf.evaluate(nsc_orig, x_test_scaled, y_test)
    nsc_orig.save('Files/nsc_orig.h5')

else:
    #Load:
    nsc_orig = load_model('Files/nsc_orig.h5')
    
pool_size = 100000
number_points_to_add = np.array([pool_size, 2*pool_size])

kernel_type = 'rbf'

orig_decision = True
train_active = True

if orig_decision:
    start = time.time()
    
    # preparing dataset to train the rejection rule
    cal_unc_orig, cal_pred_probs_orig, alphas_orig = compute_calibration_uncertainties(nsc_orig, x_cal_scaled, y_cal) # conf and cred
    cal_error_indexes_orig = label_correct_incorrect_pred(np.round(cal_pred_probs_orig), y_cal)

    rej_rule_orig = train_svc_rejection_rule(kernel_type, cal_unc_orig, cal_error_indexes_orig)

    eval_time = time.time()
    test_unc_orig, test_pred_probs_orig = compute_test_uncertainties(nsc_orig, alphas_orig, x_test_scaled) # conf and cred
    svc_test_pred_orig = apply_svc_rejection_rule(rej_rule_orig, test_unc_orig)
    final_eval_time = time.time()-eval_time
    print("Time to evaluate CP-NPM on the test set: ", final_eval_time, ". Per point time: ", final_eval_time/n_test_points)


    test_error_indexes_orig = label_correct_incorrect_pred(np.round(test_pred_probs_orig), y_test)

    svc_orig_accuracy = np.sum((svc_test_pred_orig==test_error_indexes_orig))/n_test_points
    n_test_errors_orig = np.sum((test_error_indexes_orig==-1))
    n_rej_points_orig = np.sum((svc_test_pred_orig==-1))
    print("SVC rejection rate ORIGINAL: ", 100*n_rej_points_orig/n_test_points, "%")
    print("SVC accuracy ORIGINAL: ", svc_orig_accuracy)

    orig_rejection_rate = 100*n_rej_points_orig/n_test_points

    recognized_orig = 0
    for i in range(n_test_points):
        if test_error_indexes_orig[i] == -1 and svc_test_pred_orig[i] == -1:
            recognized_orig += 1

    print("SVC recognized errors ORIGINAL: {}/{}".format(recognized_orig, n_test_errors_orig))
    orig_recogn_rate = 100*recognized_orig/n_test_errors_orig
'''
fig, ax = plt.subplots(1, 3, figsize=(12,8))
ax[0].scatter(cal_unc_orig[:,0],cal_unc_orig[:,1], c=cal_error_indexes_orig, s=1)
ax[0].set_xlabel('confidence')
ax[0].set_ylabel('credibility')
ax[0].set_title('Calidation errors')
ax[1].scatter(test_unc_orig[:,0],test_unc_orig[:,1], c=svc_test_pred_orig, s=1)
ax[1].set_xlabel('confidence')
ax[1].set_ylabel('credibility')
ax[1].set_title('Rejected test points')
ax[2].scatter(test_unc_orig[:,0],test_unc_orig[:,1], c=test_error_indexes_orig, s=1)
ax[2].set_xlabel('confidence')
ax[2].set_ylabel('credibility')
ax[2].set_title('Test errors')
plt.tight_layout()
string_name = 'Plots/CP_SVC_original_rejection_conf_cred.png'
fig.savefig(string_name)
'''

if train_active:
    print("::: Active learning starts ::::")
    start = time.time()
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
    n_retrain = int(np.round(n_active_points*SPLIT_RATE))

    x_retrain = np.vstack((x_train_scaled, x_add_act[:n_retrain, :]))
    y_retrain = np.hstack((y_train, y_add_act[:n_retrain]))
    x_recal = np.vstack((x_cal_scaled, x_add_act[n_retrain:, :]))
    y_recal = np.hstack((y_cal, y_add_act[n_retrain:]))

    nsc_active = nsc_tf.train(x_retrain, y_retrain, n_epochs = N_EPOCHS) 
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

    active_rejection_rate = 100*n_rej_points_active/n_test_points

    recognized_active = 0
    for i in range(n_test_points):
        if test_error_indexes_active[i] == -1 and svc_test_pred_active[i] == -1:
            recognized_active += 1

    print("SVC recognized errors ACTIVE: {}/{}".format(recognized_active, n_test_errors_active))
    active_recogn_rate = 100*recognized_active/n_test_errors_active
    '''
    fig, ax = plt.subplots(1, 3, figsize=(12,8))
    ax[0].scatter(cal_unc_active[:,0],cal_unc_active[:,1], c=cal_error_indexes_active, s=1)
    ax[0].set_xlabel('confidence')
    ax[0].set_ylabel('credibility')
    ax[0].set_title('Validation errors')
    ax[1].scatter(test_unc_active[:,0],test_unc_active[:,1], c=svc_test_pred_active, s=1)
    ax[1].set_xlabel('confidence')
    ax[1].set_ylabel('credibility')
    ax[1].set_title('Rejected test points')
    ax[2].scatter(test_unc_active[:,0],test_unc_active[:,1], c=test_error_indexes_active, s=1)
    ax[2].set_xlabel('confidence')
    ax[2].set_ylabel('credibility')
    ax[2].set_title('Test errors')
    plt.tight_layout()
    string_name = 'Plots/CP_active_SVC_rejection_mean_std_pool_size{}.png'.format(2*pool_size)
    fig.savefig(string_name)
    '''

#---- Last phase--------------------------
# Reval could also be splitted in two sets and we could use a subset to compute the conformal set (prediction region) 
# related to the svc trained on a subset of reval
last_phase = True
if last_phase:
    final_split_rate = 0.7
    x_svc_train, y_svc_train, x_svc_cal, y_svc_cal, perm = split_train_calibration_w_indexes(x_recal, y_recal, split_rate = final_split_rate)
    split_index = int(x_recal.shape[0]*final_split_rate)

    final_val_unc_active = cal_unc_active[perm[:split_index], :]
    final_val_error_indexes_active = cal_error_indexes_active[perm[:split_index]]

    # rejection rule trained on a smaller validation set
    svc_final = train_svc_rejection_rule(kernel_type, final_val_unc_active, final_val_error_indexes_active)

    # punti tenuti da parte per calcolare i calibration scores della rejection rule
    final_cal_unc = cal_unc_active[perm[split_index:], :]
    final_cal_error_indexes = cal_error_indexes_active[perm[split_index:]]

    final_pred_errors = apply_svc_rejection_rule(svc_final, test_unc_active)
    active_rej_indexes_final = (final_pred_errors < 0).astype(bool)
    print("FINAL nb rej points: ", np.sum(active_rej_indexes_final))

    betas,_ = compute_SVC_nonconformity_scores(final_cal_unc, final_cal_error_indexes, svc_final)
    q_svc = betas.shape[0]
    cal_error_count = np.sum((final_cal_error_indexes==-1))

    print("Errors present in the svc calibration set: {}/{}".format(cal_error_count, q_svc))

    test_svc_pvalues = compute_p_values(betas, x= test_unc_active, svc = svc_final, cal_labels = final_cal_error_indexes, mondrian = True, classifier = "SVC", class_dict = {"class_pos": 1, "class_neg": -1})

    print("PVALUES OVERVIEW    ----\n", test_svc_pvalues)


    xxx = 51
    rej_rate_risky = np.zeros(xxx)
    rej_rate_cons = np.zeros(xxx)
    recogn_rate_risky = np.zeros(xxx)
    recogn_rate_cons = np.zeros(xxx)
    #epsilon_vector = np.array([0.005, 0.01, 0.15, 0.02, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25])
    epsilon_vector = np.linspace(0.005,0.15,xxx)
    for j, epsilon in enumerate(epsilon_vector):
        print("epsilon = ", epsilon)
        # conformal region: first row label -1, second row label +1
        # if index = 1 that class belong to the conformal region
        conformal_region = (test_svc_pvalues > epsilon)
        n_labels_in_pred_region = np.sum(conformal_region, axis=0)

        #LABELS 0: rejected AND error, 1 FP: rejected BUT correct
        #       2 FN: accepted BUT error, 3: accepted AND correct


        conservative_pred = -1*np.ones(n_test_points)
        risky_pred = np.ones(n_test_points)
        final_combined_preds = np.zeros(n_test_points)
        for i in range(n_test_points):
            if n_labels_in_pred_region[i] == 0: # empty set
                final_combined_preds[i] = 0
            elif n_labels_in_pred_region[i] == 2: # entire set
                final_combined_preds[i] = 2      
            else:
                if conformal_region[0,i] == 1:
                    final_combined_preds[i] = -1
                    risky_pred[i] = -1
                else:
                    conservative_pred[i] = 1
                    final_combined_preds[i] = 1 

        # conservative and risky performances
        #svc_accuracy_conservative = np.sum((conservative_pred==test_error_indexes_active))/n_test_points
        #svc_accuracy_risky = np.sum((risky_pred==test_error_indexes_active))/n_test_points

        n_rej_points_conservative = np.sum((conservative_pred==-1))
        rej_rate_cons[j] = 100*n_rej_points_conservative/n_test_points

        n_rej_points_risky = np.sum((risky_pred==-1))
        rej_rate_risky[j] = 100*n_rej_points_risky/n_test_points

        recognized_conservative = 0
        recognized_risky = 0
        for i in range(n_test_points):
            if test_error_indexes_active[i] == -1 and conservative_pred[i] == -1:
                recognized_conservative += 1
            if test_error_indexes_active[i] == -1 and risky_pred[i] == -1:
                recognized_risky += 1
        recogn_rate_cons[j] = 100*recognized_conservative/n_test_errors_active
        recogn_rate_risky[j] = 100*recognized_risky/n_test_errors_active
        

print("RISKY rejection rates: ", rej_rate_risky)
print("RISKY recognition rates: ", recogn_rate_risky)
print("CONS rejection rates: ", rej_rate_cons)
print("CONS recognition rates: ", recogn_rate_cons)
#------------------------------------------------------

'''
fig1 = plt.figure()
plt.plot(epsilon_vector, rej_rate_risky,  'ro--', label='risky')
plt.plot(epsilon_vector, active_rejection_rate*np.ones(xxx), 'go--', label='active')
plt.plot(epsilon_vector, orig_rejection_rate*np.ones(xxx), 'bo--', label='original')
plt.legend()
plt.title('Rejection Rate')
plt.xlabel("epsilon")
plt.ylabel("rej. rate")
string_name = 'Plots/AP_REJ_RATE.png'
plt.rcParams.update({'font.size': 22})
plt.tight_layout()
fig1.savefig(string_name)

fig2 = plt.figure()
plt.plot(epsilon_vector, recogn_rate_risky,  'ro--', label='risky')
plt.plot(epsilon_vector, active_recogn_rate*np.ones(xxx), 'go--', label='active')
plt.plot(epsilon_vector, orig_recogn_rate*np.ones(xxx), 'bo--', label='original')
plt.legend()
plt.xlabel("epsilon")
plt.ylabel("recogn. rate")
plt.title('Recognition Rate')
string_name = 'Plots/AP_RECOGN_RATE.png'
plt.rcParams.update({'font.size': 22})
plt.tight_layout()
fig2.savefig(string_name)
'''
plt.rcParams.update({'font.size': 22})
fig3, ax = plt.subplots(2, 1, figsize=(12,8))
ax[0].plot(epsilon_vector, rej_rate_risky,  'ro--', label='risk')
ax[0].plot(epsilon_vector, active_rejection_rate*np.ones(xxx), 'go--', label='act')
ax[0].plot(epsilon_vector, orig_rejection_rate*np.ones(xxx), 'bo--', label='orig')
ax[0].legend()
ax[0].set_ylabel("rej.rate (%)")
ax[0].set_title('Artificial Pancreas')
ax[1].plot(epsilon_vector, recogn_rate_risky,  'ro--',label='risk')
ax[1].plot(epsilon_vector, active_recogn_rate*np.ones(xxx), 'go--',label='act')
ax[1].plot(epsilon_vector, orig_recogn_rate*np.ones(xxx), 'bo--',label='orig')
ax[1].legend()
ax[1].set_ylabel("recogn.rate (%)")
ax[1].set_xlabel("epsilon")
plt.tight_layout()
string_name = 'Plots/AP_REJ_AND_RECOGN_RATES_rnd_trial5.png'
fig3.savefig(string_name)


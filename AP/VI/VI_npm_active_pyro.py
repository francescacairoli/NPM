import numpy as np
from sklearn.datasets import make_moons
from sklearn import preprocessing

from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, SGD
import torch.nn.functional as F
from pyro.distributions import Normal, Categorical, Bernoulli


from vi_utilities_pt import *

bern = False

class NN(nn.Module):
	
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.fc1(x)
        output = torch.tanh(output)
        output = self.fc2(output)
        output = torch.tanh(output)
        output = self.fc3(output)
        output = torch.tanh(output)
        output = self.out(output)
        #output = torch.sigmoid(output)
        output = F.softmax(output,dim=0)
        return output

n_epochs = 20001 # number of epochs
n_pred_samples = 100

split_rate = 0.7

print("n epochs = ", n_epochs)
print("n pred samples = ", n_pred_samples)
print("split_rate = ", split_rate)

x_train_full, y_train_full, x_test, y_test = load_data('../AP_20K_unif_train.mat', '../AP_10K_unif_test.mat')
# Data standardization
scaler = preprocessing.StandardScaler().fit(x_train_full)
x_train_full = scaler.transform(x_train_full)
x_test = scaler.transform(x_test)
x_train, y_train, x_val, y_val = split_train_calibration(x_train_full, y_train_full, split_rate = split_rate)

n_training_points = x_train.shape[0]
n_val_points = x_val.shape[0]
n_test_points = x_test.shape[0]

if bern:
    net = NN(x_train.shape[1],10,1)
else:
    net = NN(x_train.shape[1],10,2)
BATCH_SIZE = n_training_points

#log_softmax = nn.Softmax(dim=0)

def model(x_data, y_data):
    
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

    fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weight), scale=torch.ones_like(net.fc2.weight))
    fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias), scale=torch.ones_like(net.fc2.bias))

    fc3w_prior = Normal(loc=torch.zeros_like(net.fc3.weight), scale=torch.ones_like(net.fc3.weight))
    fc3b_prior = Normal(loc=torch.zeros_like(net.fc3.bias), scale=torch.ones_like(net.fc3.bias))
    
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))
    
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior, 'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    
    #lhat = log_softmax(lifted_reg_model(x_data))
    lhat = lifted_reg_model(x_data)
        
    if bern:
        pyro.sample("obs", Bernoulli(probs=lhat), obs=y_data)
    else:
        pyro.sample("obs", Categorical(lhat), obs=y_data)

softplus = torch.nn.Softplus()

def guide(x_data, y_data):
    
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    # Second layer weight distribution priors
    fc2w_mu = torch.randn_like(net.fc2.weight)
    fc2w_sigma = torch.randn_like(net.fc2.weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
    # Second layer bias distribution priors
    fc2b_mu = torch.randn_like(net.fc2.bias)
    fc2b_sigma = torch.randn_like(net.fc2.bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)

        # Third layer weight distribution priors
    fc3w_mu = torch.randn_like(net.fc3.weight)
    fc3w_sigma = torch.randn_like(net.fc3.weight)
    fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
    fc3w_sigma_param = softplus(pyro.param("fc3w_sigma", fc3w_sigma))
    fc3w_prior = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param)
    # Third layer bias distribution priors
    fc3b_mu = torch.randn_like(net.fc3.bias)
    fc3b_sigma = torch.randn_like(net.fc3.bias)
    fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
    fc3b_sigma_param = softplus(pyro.param("fc3b_sigma", fc3b_sigma))
    fc3b_prior = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)

    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()

#adam_params = {"lr": 0.075, "betas": (0.95, 0.999)}
adam_params = {}
optim = Adam(adam_params)
elbo = Trace_ELBO()
svi = SVI(model, guide, optim, loss=elbo)

if bern:
    batch_y_t = torch.FloatTensor(y_train)
else:
    batch_y_t = torch.LongTensor(y_train)

batch_x_t = torch.FloatTensor(x_train)

start_orig = time.time()

for j in range(n_epochs):
    loss = svi.step(batch_x_t, batch_y_t)/ n_training_points
    if j%50==0:
        print("Epoch ", j, " Loss ", loss)
if bern:
    x_test_t = torch.FloatTensor(x_test)
    y_test_t = torch.LongTensor(y_test)
    x_val_t = torch.FloatTensor(x_val)
    y_val_t = torch.LongTensor(y_val)
else:
    x_test_t = torch.FloatTensor(x_test)
    y_test_t = torch.FloatTensor(y_test)
    x_val_t = torch.FloatTensor(x_val)
    y_val_t = torch.FloatTensor(y_val)

val_avg_pred_classes, val_avg_pred_probs, val_std_pred_probs = predictions_on_set(x_val_t, n_val_points, n_pred_samples, guide)
start_eval = time.time()
test_avg_pred_classes, test_avg_pred_probs, test_std_pred_probs = predictions_on_set(x_test_t, n_test_points, n_pred_samples, guide)
eval_time = time.time()-start_eval

val_unc = np.vstack((val_avg_pred_probs, val_std_pred_probs))
test_unc = np.vstack((test_avg_pred_probs, test_std_pred_probs))

val_error_indexes = label_correct_incorrect_pred(val_avg_pred_classes, y_val)
test_error_indexes = label_correct_incorrect_pred(test_avg_pred_classes, y_test)
print("Val avg pred accuracy: ", np.sum((val_error_indexes==1))/n_val_points)
print("Test avg pred accuracy ORIGINAL: ", np.sum((test_error_indexes==1))/n_test_points)

val_n_errors = np.sum(val_error_indexes==-1)
val_n_corrects = np.sum(val_error_indexes==1)

# train original rejection rule using GPC
kernelstring = "RBF"
print("KERNEL_TYPE = "+kernelstring)
if kernelstring=="RBF":
    kernel_gp = 1.0*RBF(1.0)
if kernelstring=="DP":
    kernel_gp = DotProduct()

gpc_orig = train_gpc_rejection_rule(kernel_gp, val_unc, val_error_indexes)

# Find the optimal threshold value ORIGINAL
grid_size = 51
DT_vector = np.linspace(0.5,0.999,grid_size)
TPR = np.zeros(grid_size)
FPR = np.zeros(grid_size)
for j, dt_j in enumerate(DT_vector):
    gpc_val_pred = apply_gpc_rejection_rule(gpc_orig, val_unc, dt_j)
    n_recogn_errors = 0
    n_wrongly_rej = 0
    for jj in range(len(gpc_val_pred)):
        if gpc_val_pred[jj] == -1:
            if val_error_indexes[jj] == 1:
                n_wrongly_rej += 1
            else:
                n_recogn_errors += 1
    TPR[j] = n_recogn_errors/val_n_errors
    FPR[j] = n_wrongly_rej/val_n_corrects

roc_diff = TPR-FPR
opt_index = np.argmax(roc_diff)
orig_optimal_threshold = DT_vector[opt_index]
print("decision thresholds: ", DT_vector)
print("TPR-FPR = ", roc_diff)
print("optimal threshold original= ", orig_optimal_threshold)


fig = plt.figure()
plt.plot(FPR, TPR, 'bo-')
plt.plot(FPR[opt_index], TPR[opt_index], 'ro')
plt.title("GPC ROC curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
fig.savefig("Plots/"+kernelstring+"_orig_ROC_curve.png")
#-----

start_eval = time.time()
gpc_test_pred_opt = apply_gpc_rejection_rule(gpc_orig, test_unc, orig_optimal_threshold)
eval_time += time.time()-start_eval
print("AP EVAL TIME: ", eval_time)

# original performances
gpc_accuracy=np.sum((gpc_test_pred_opt==test_error_indexes))/n_test_points

n_test_errors = np.sum((test_error_indexes==-1))
n_rej_points = np.sum((gpc_test_pred_opt==-1))

print("Rejection rate ORIGINAL: ", 100*n_rej_points/n_test_points, "%")
print("GPC test accuracy ORIGINAL: ", gpc_accuracy)

recognized = 0
for i in range(n_test_points):
    if test_error_indexes[i] == -1 and gpc_test_pred_opt[i] == -1:
        recognized += 1

print("Time ORIG phase: ", time.time()-start_orig)


print("Recognized errors ORIGINAL: {}/{}".format(recognized, n_test_errors))
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print("\n")

fig, ax = plt.subplots(1, 3, figsize=(12,8))
ax[0].scatter(val_unc[0],val_unc[1], c=val_error_indexes, s=1)
ax[0].set_ylabel('std')
ax[0].set_xlabel('mean')
ax[0].set_title('Validation errors')
ax[1].scatter(test_avg_pred_probs,test_std_pred_probs, c=gpc_test_pred_opt, s=1)
ax[1].set_ylabel('std')
ax[1].set_xlabel('mean')
ax[1].set_title('Rejected test points')
ax[2].scatter(test_avg_pred_probs,test_std_pred_probs, c=test_error_indexes, s=1)
ax[2].set_ylabel('std')
ax[2].set_xlabel('mean')
ax[2].set_title('Test errors')
plt.tight_layout()
string_name = 'Plots/PYRO_SN_VI_ORIGINAL_rejection_mean_std_CAT_fixed_SOFTMAX.png'
fig.savefig(string_name)


DO_ACTIVE = False
n_active_points = 5295
if DO_ACTIVE:
	print("::: Active learning starts ::::")
	# retrain
	start_active = time.time()

	n_points_retrain = 100000

	x_samp_retrain, samp_retrain_real_class = active_label_query(n_points_retrain, scaler, guide, gpc_orig, n_pred_samples, orig_optimal_threshold)
	n_active_points = len(samp_retrain_real_class)
	print("N ACTIVE POINTS: ", n_active_points)
	x_add_train, y_add_train, x_add_val, y_add_val = split_train_calibration(x_samp_retrain,samp_retrain_real_class, split_rate = split_rate)

	x_retrain = np.vstack((x_train, x_add_train))
	#y_retrain = np.vstack((y_train, y_add_train.reshape((len(y_add_train),n_output))))
	y_retrain = np.concatenate((y_train, y_add_train), axis=0)

	x_reval = np.vstack((x_val, x_add_val))
	#y_reval = np.vstack((y_val, y_add_val.reshape((len(y_add_val),n_output))))
	y_reval =  np.concatenate((y_val, y_add_val), axis=0)
	# RETRAIN BNN

	svi_act = SVI(model, guide, optim, loss=elbo)

	if bern:
	    act_batch_y_t = torch.FloatTensor(y_retrain)
	    act_batch_x_t = torch.FloatTensor(x_retrain)
	    x_reval_t = torch.FloatTensor(x_reval)
	    y_reval_t = torch.FloatTensor(y_reval)
	else:
	    act_batch_y_t = torch.LongTensor(y_retrain)
	    act_batch_x_t = torch.FloatTensor(x_retrain)
	    x_reval_t = torch.FloatTensor(x_reval)
	    y_reval_t = torch.LongTensor(y_reval)

	for j in range(n_epochs):
	    loss_act = svi_act.step(act_batch_x_t, act_batch_y_t)/ (x_retrain.shape[0])
	    if j%50==0:
	        print("Epoch ", j, " Loss ", loss_act)

	reval_avg_pred_class_active, reval_avg_probs_active, reval_std_probs_active = predictions_on_set(x_reval_t, x_reval.shape[0], n_pred_samples, guide)
	test_avg_pred_class_active, test_avg_probs_active, test_std_probs_active = predictions_on_set(x_test_t, n_test_points, n_pred_samples, guide)

	reval_unc_active = np.vstack((reval_avg_probs_active, reval_std_probs_active))
	reval_error_indexes_active = label_correct_incorrect_pred(reval_avg_pred_class_active, y_reval)

	test_unc_active = np.vstack((test_avg_probs_active, test_std_probs_active))
	test_error_indexes_active = label_correct_incorrect_pred(test_avg_pred_class_active, y_test)

	print("Test avg pred accuracy RETRAINED: ", np.sum((test_error_indexes_active==1))/n_test_points)

	reval_n_errors_act = np.sum(reval_error_indexes_active==-1)
	reval_n_corrects_act = np.sum(reval_error_indexes_active==1)
	    
	gpc_active = train_gpc_rejection_rule(kernel_gp, reval_unc_active, reval_error_indexes_active)

	# Find the optimal threshold value ORIGINAL
	TPR_act = np.zeros(grid_size)
	FPR_act = np.zeros(grid_size)
	for j, dt_j in enumerate(DT_vector):
	    gpc_val_pred = apply_gpc_rejection_rule(gpc_active, reval_unc_active, dt_j)
	    n_recogn_errors = 0
	    n_wrongly_rej = 0
	    for jj in range(len(gpc_val_pred)):
	        if gpc_val_pred[jj] == -1:
	            if reval_error_indexes_active[jj] == 1:
	                n_wrongly_rej += 1
	            else:
	                n_recogn_errors += 1
	    TPR_act[j] = n_recogn_errors/reval_n_errors_act
	    FPR_act[j] = n_wrongly_rej/reval_n_corrects_act

	roc_diff_act = TPR_act-FPR_act
	opt_index_act = np.argmax(roc_diff_act)
	active_optimal_threshold = DT_vector[opt_index_act]
	print("decision thresholds: ", DT_vector)
	print("TPR-FPR = ", roc_diff_act)
	print("optimal threshold active= ", active_optimal_threshold)


	gpc_test_pred_active_opt = apply_gpc_rejection_rule(gpc_active, test_unc_active, active_optimal_threshold)

	# active performances
	gpc_accuracy_active = np.sum((gpc_test_pred_active_opt==test_error_indexes_active))/n_test_points
	n_test_errors_active = np.sum((test_error_indexes_active==-1))

	n_rej_points_active = np.sum((gpc_test_pred_active_opt==-1))
	print("Rejection rate ACTIVE: ", 100*n_rej_points_active/n_test_points, "%")
	print("GPC accuracy ACTIVE: ", gpc_accuracy_active)

	recognized_act = 0
	for i in range(n_test_points):
	    if test_error_indexes_active[i] == -1 and gpc_test_pred_active_opt[i] == -1:
	        recognized_act += 1

	print("Recognized errors ACTIVE: {}/{}".format(recognized_act, n_test_errors_active))
	print("Time active phase: ", time.time()-start_active)

	fig, ax = plt.subplots(1, 3, figsize=(12,8))
	ax[0].scatter(reval_unc_active[0],reval_unc_active[1], c=reval_error_indexes_active, s=1)
	ax[0].set_ylabel('std')
	ax[0].set_xlabel('mean')
	ax[0].set_title('Validation errors')
	ax[1].scatter(test_avg_probs_active,test_std_probs_active, c=gpc_test_pred_active_opt, s=1)
	ax[1].set_ylabel('std')
	ax[1].set_xlabel('mean')
	ax[1].set_title('Rejected test points')
	ax[2].scatter(test_avg_probs_active,test_std_probs_active, c=test_error_indexes_active, s=1)
	ax[2].set_ylabel('std')
	ax[2].set_xlabel('mean')
	ax[2].set_title('Test errors')
	plt.tight_layout()
	string_name = 'Plots/PYRO_SN_VI_ACTIVE_rejection_mean_std_CAT_fixed_SOFTMAX.png'
	fig.savefig(string_name)

else:
	
	start_passive = time.time()
	print("::: Passive learning starts ::::")
	x_samp_retrain_pass, samp_retrain_real_class_pass = passive_label_query(n_active_points, scaler)
	    
	x_add_train_pass, y_add_train_pass, x_add_val_pass, y_add_val_pass = split_train_calibration(x_samp_retrain_pass, samp_retrain_real_class_pass, split_rate = split_rate)

	x_retrain_pass = np.vstack((x_train, x_add_train_pass))
	#y_retrain_pass = np.vstack((y_train, y_add_train_pass.reshape((len(y_add_train),n_output))))
	y_retrain_pass = np.concatenate((y_train, y_add_train_pass), axis=0)

	x_reval_pass = np.vstack((x_val, x_add_val_pass))
	#y_reval_pass = np.vstack((y_val, y_add_val_pass.reshape((len(y_add_val),n_output))))
	y_reval_pass = np.concatenate((y_val, y_add_val_pass), axis=0)

	# retrain bnn
	print("Retraining the BNN PASSIVE...")
	        
	svi_pass = SVI(model, guide, optim, loss=elbo)

	if bern:
	    pass_batch_y_t = torch.FloatTensor(y_retrain_pass)
	    pass_batch_x_t = torch.FloatTensor(x_retrain_pass)

	    x_reval_pass_t = torch.FloatTensor(x_reval_pass)
	    y_reval_pass_t = torch.FloatTensor(y_reval_pass)
	else:
	    pass_batch_y_t = torch.LongTensor(y_retrain_pass)
	    pass_batch_x_t = torch.FloatTensor(x_retrain_pass)

	    x_reval_pass_t = torch.FloatTensor(x_reval_pass)
	    y_reval_pass_t = torch.LongTensor(y_reval_pass)

	for j in range(n_epochs):
	    loss_pass = svi_pass.step(pass_batch_x_t, pass_batch_y_t)/ (x_retrain_pass.shape[0])
	    if j%50==0:
	        print("Epoch ", j, " Loss ", loss_pass)

	reval_avg_pred_class_passive, reval_avg_probs_passive, reval_std_probs_passive = predictions_on_set(x_reval_pass_t, x_reval_pass.shape[0], n_pred_samples, guide)
	test_avg_pred_class_passive, test_avg_probs_passive, test_std_probs_passive = predictions_on_set(x_test_t, n_test_points, n_pred_samples, guide)

	reval_unc_passive = np.vstack((reval_avg_probs_passive, reval_std_probs_passive))
	reval_error_indexes_passive = label_correct_incorrect_pred(reval_avg_pred_class_passive, y_reval_pass)

	test_unc_passive = np.vstack((test_avg_probs_passive, test_std_probs_passive))
	test_error_indexes_passive = label_correct_incorrect_pred(test_avg_pred_class_passive, y_test)

	print("Test avg pred accuracy RETRAINED: ", np.sum((test_error_indexes_passive==1))/n_test_points)

	reval_n_errors_pass = np.sum(reval_error_indexes_passive==-1)
	reval_n_corrects_pass = np.sum(reval_error_indexes_passive==1)
	    
	gpc_passive = train_gpc_rejection_rule(kernel_gp, reval_unc_passive, reval_error_indexes_passive)

	# Find the optimal threshold value ORIGINAL
	TPR_pass = np.zeros(grid_size)
	FPR_pass = np.zeros(grid_size)
	for j, dt_j in enumerate(DT_vector):
	    gpc_val_pred = apply_gpc_rejection_rule(gpc_passive, reval_unc_passive, dt_j)
	    n_recogn_errors = 0
	    n_wrongly_rej = 0
	    for jj in range(len(gpc_val_pred)):
	        if gpc_val_pred[jj] == -1:
	            if reval_error_indexes_passive[jj] == 1:
	                n_wrongly_rej += 1
	            else:
	                n_recogn_errors += 1
	    TPR_pass[j] = n_recogn_errors/reval_n_errors_pass
	    FPR_pass[j] = n_wrongly_rej/reval_n_corrects_pass

	roc_diff_pass = TPR_pass-FPR_pass
	opt_index_pass = np.argmax(roc_diff_pass)
	passive_optimal_threshold = DT_vector[opt_index_pass]
	print("decision thresholds: ", DT_vector)
	print("TPR-FPR = ", roc_diff_pass)
	print("optimal threshold active= ", passive_optimal_threshold)


	gpc_test_pred_passive_opt = apply_gpc_rejection_rule(gpc_passive, test_unc_passive, passive_optimal_threshold)

	# active performances
	gpc_accuracy_passive = np.sum((gpc_test_pred_passive_opt==test_error_indexes_passive))/n_test_points
	n_test_errors_passive = np.sum((test_error_indexes_passive==-1))

	n_rej_points_passive = np.sum((gpc_test_pred_passive_opt==-1))
	print("Rejection rate PASSIVE: ", 100*n_rej_points_passive/n_test_points, "%")
	print("GPC accuracy PASSIVE: ", gpc_accuracy_passive)

	recognized_pass = 0
	for i in range(n_test_points):
	    if test_error_indexes_passive[i] == -1 and gpc_test_pred_passive_opt[i] == -1:
	        recognized_pass += 1

	print("Recognized errors PASSIVE: {}/{}".format(recognized_pass, n_test_errors_passive))
	print("Time passive phase: ", time.time()-start_passive)

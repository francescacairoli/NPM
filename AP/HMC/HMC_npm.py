from hmc_utilities import *

SPLIT_RATE = 0.7
print("----------------------ARTIFICIAL PANCREAS---------------------")

def run_original(split_rate):
    x_train_full, y_train_full, x_test, y_test = load_data('../AP_20K_unif_train.mat', '../AP_10K_unif_test.mat')

    # Data standardization
    scaler = preprocessing.StandardScaler().fit(x_train_full)
    x_train_full = scaler.transform(x_train_full)
    x_test = scaler.transform(x_test)

    x_train, y_train, x_val, y_val = split_train_calibration(x_train_full, y_train_full, split_rate = split_rate)

    step_size = 0.01
    n_steps = 10
    T = 200
    print("Step size: ", step_size)
    print("Number of n steps: ", n_steps)
    print("T = ", T)

    n_samples = 100
    print("Number of n_samples: ", n_samples)

    n_input = x_train.shape[1]
    n_hidden = 10
    n_output = 1

    n_training_points = x_train.shape[0]
    n_test_points = x_test.shape[0]
    n_val_points = x_val.shape[0]

    batch_size = n_training_points

    n_batch = int(n_training_points/batch_size)

    P = np.ones(8)
    PP = np.ones(8)

    X = tf.placeholder(tf.float32, shape = [None, n_input], name = "x_placeholder")
    Y = tf.placeholder(tf.int32, [None, n_output])

    W_fc1 = Normal(loc=tf.zeros([n_input,n_hidden]), scale=P[0]*tf.ones([n_input,n_hidden]), name="W_fc1")
    b_fc1 = Normal(loc=tf.zeros([n_hidden]), scale=P[1]*tf.ones([n_hidden]), name="b_fc1")
    W_fc2 = Normal(loc=tf.zeros([n_hidden, n_hidden]), scale=P[2]*tf.ones([n_hidden, n_hidden]), name="W_fc2")
    b_fc2 = Normal(loc=tf.zeros([n_hidden]), scale=P[3]*tf.ones([n_hidden]), name="b_fc2")
    W_fc3 = Normal(loc=tf.zeros([n_hidden, n_hidden]), scale=P[4]*tf.ones([n_hidden, n_hidden]), name="W_fc3")
    b_fc3 = Normal(loc=tf.zeros([n_hidden]), scale=P[5]*tf.ones([n_hidden]), name="b_fc3")
    W_fc4 = Normal(loc=tf.zeros([n_hidden, n_output]), scale=P[6]*tf.ones([n_hidden, n_output]), name="W_fc4")
    b_fc4 = Normal(loc=tf.zeros([n_output]), scale=P[7]*tf.ones([n_output]), name="b_fc4")
    h_fc1 = tf.tanh(tf.matmul(X,W_fc1)+b_fc1)
    h_fc2 = tf.tanh(tf.matmul(h_fc1,W_fc2)+b_fc2)
    h_fc3 = tf.tanh(tf.matmul(h_fc2,W_fc3)+b_fc3)
    hy = tf.sigmoid(tf.matmul(h_fc3,W_fc4)+b_fc4)

    y = Bernoulli(hy)

    with tf.name_scope("posterior"):
        qW_fc1 = Empirical(params = tf.Variable(PP[0]*tf.random_normal([T,n_input,n_hidden])))
        qb_fc1 = Empirical(params = tf.Variable(PP[1]*tf.random_normal([T,n_hidden])))
        qW_fc2 = Empirical(params = tf.Variable(PP[2]*tf.random_normal([T,n_hidden, n_hidden])))
        qb_fc2 = Empirical(params = tf.Variable(PP[3]*tf.random_normal([T,n_hidden])))
        qW_fc3 = Empirical(params = tf.Variable(PP[4]*tf.random_normal([T,n_hidden, n_hidden])))
        qb_fc3 = Empirical(params = tf.Variable(PP[5]*tf.random_normal([T,n_hidden])))
        qW_fc4 = Empirical(params = tf.Variable(PP[6]*tf.random_normal([T,n_hidden, n_output])))
        qb_fc4 = Empirical(params = tf.Variable(PP[7]*tf.random_normal([T,n_output])))


    inference = ed.HMC({W_fc1: qW_fc1, b_fc1: qb_fc1, W_fc2: qW_fc2, b_fc2: qb_fc2, W_fc3: qW_fc3, b_fc3: qb_fc3, W_fc4: qW_fc4, b_fc4: qb_fc4}, data={y: Y})
    inference.initialize(step_size=step_size, n_steps=n_steps)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    burn_in = inference.n_iter-n_samples

    y_train = y_train.reshape((n_training_points,n_output))
    y_test = y_test.reshape((n_test_points,n_output))
    y_val = y_val.reshape((n_val_points,n_output))

    # train original bnn
    wfc1_lst, wfc2_lst, wfc3_lst, wfc4_lst = np.zeros((n_samples,n_input,n_hidden)), np.zeros((n_samples,n_hidden,n_hidden)), np.zeros((n_samples,n_hidden,n_hidden)), np.zeros((n_samples,n_hidden,n_output))
    bfc1_lst,  bfc2_lst,  bfc3_lst, bfc4_lst = np.zeros((n_samples,n_hidden)), np.zeros((n_samples,n_hidden)), np.zeros((n_samples,n_hidden)), np.zeros((n_samples,n_output))

    start_orig = time.time()

    iii = 0
    for j in range(inference.n_iter):
        info_dict = inference.update(feed_dict={X: x_train, Y: y_train})
        inference.print_progress(info_dict)

        if j>= burn_in:
            W_fc1 = qW_fc1.eval()
            b_fc1 = qb_fc1.eval()
            wfc1_lst[iii] = W_fc1
            bfc1_lst[iii] = b_fc1
            W_fc2 = qW_fc2.eval()
            b_fc2 = qb_fc2.eval() 
            wfc2_lst[iii] = W_fc2
            bfc2_lst[iii] = b_fc2
            W_fc3 = qW_fc3.eval()
            b_fc3 = qb_fc3.eval()
            wfc3_lst[iii] = W_fc3
            bfc3_lst[iii] = b_fc3
            W_fc4 = qW_fc4.eval()
            b_fc4 = qb_fc4.eval()
            wfc4_lst[iii] = W_fc4
            bfc4_lst[iii] = b_fc4
            iii += 1

    orig_weights_lists = (wfc1_lst, bfc1_lst, wfc2_lst, bfc2_lst, wfc3_lst, bfc3_lst, wfc4_lst, bfc4_lst)



    val_avg_probs, val_std_probs, val_avg_pred_class = compute_avg_std_pred_probs(x_val, orig_weights_lists)
    eval_time = time.time()
    test_avg_probs, test_std_probs, test_avg_pred_class = compute_avg_std_pred_probs(x_test, orig_weights_lists)
    eval_time = time.time()-eval_time
    val_unc = np.vstack((val_avg_probs, val_std_probs))
    test_unc = np.vstack((test_avg_probs, test_std_probs))

    val_error_indexes = label_correct_incorrect_pred(val_avg_pred_class, y_val)
    test_error_indexes = label_correct_incorrect_pred(test_avg_pred_class, y_test)
    print("Val avg pred accuracy: ", np.sum((val_error_indexes==1))/n_val_points)
    print("Test avg pred accuracy ORIGINAL: ", np.sum((test_error_indexes==1))/n_test_points)

    val_n_errors = np.sum(val_error_indexes==-1)
    val_n_corrects = np.sum(val_error_indexes==1)

    # train original rejection rule
    #kernelstring = "RBF"
    KV = ["RBF"]
    for kernelstring in KV:
        print("KERNEL TYPE: "+kernelstring)

        if kernelstring=="RBF":
            kernel_type = 'BBF'
            kernel_gp = 1.0*RBF(1.0)
        if kernelstring=="DP":
            kernel_type = 'DP'
            kernel_gp = DotProduct()
        gpc_orig = train_gpc_rejection_rule(kernel_gp, val_unc, val_error_indexes)

        # Find the optimal threshold value ORIGINAL
        grid_size = 11
        DT_vector = np.linspace(0.75,0.999,grid_size)
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
        # original performances
        eval_time_2 = time.time()
        gpc_test_pred_opt = apply_gpc_rejection_rule(gpc_orig, test_unc, orig_optimal_threshold)
        eval_time_2 = time.time()-eval_time_2
        print("EVALUATION TIME: ", eval_time, eval_time_2, (eval_time+eval_time_2)/n_test_points)
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

        print("Recognized errors ORIGINAL: {}/{}".format(recognized, n_test_errors))
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------")
        print("\n")

        print("TIME ORIG PHASE: ", time.time()-start_orig)

        fig, ax = plt.subplots(1, 3, figsize=(12,8))
        ax[0].scatter(val_unc[0],val_unc[1], c=val_error_indexes, s=1)
        ax[0].set_ylabel('std')
        ax[0].set_xlabel('mean')
        ax[0].set_title('Validation errors')
        ax[1].scatter(test_avg_probs,test_std_probs, c=gpc_test_pred_opt, s=1)
        ax[1].set_ylabel('std')
        ax[1].set_xlabel('mean')
        ax[1].set_title('Rejected test points')
        ax[2].scatter(test_avg_probs,test_std_probs, c=test_error_indexes, s=1)
        ax[2].set_ylabel('std')
        ax[2].set_xlabel('mean')
        ax[2].set_title('Test errors')
        plt.tight_layout()
        string_name = 'Plots/HMC_original_rejection_mean_std_'+kernel_type+'.png'
        fig.savefig(string_name)

print("split_rate = ", SPLIT_RATE)

number_trials = 5
for i in range(number_trials):
    
    print("HMC+GP original: Results of trial number ", i)

    run_original(SPLIT_RATE)

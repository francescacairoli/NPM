from vi_utilities import *

SPLIT_RATE = 0.7
print("----------------------ARTIFICIAL PANCREAS---------------------")

def run_active(split_rate):
    x_train_full, y_train_full, x_test, y_test = load_data('../AP_20K_unif_train.mat', '../AP_10K_unif_test.mat')

    # Data standardization
    scaler = preprocessing.StandardScaler().fit(x_train_full)
    x_train_full = scaler.transform(x_train_full)
    x_test = scaler.transform(x_test)
    x_train, y_train, x_val, y_val = split_train_calibration(x_train_full, y_train_full, split_rate = split_rate)


    n_epochs = 20000
    n_samples = 100
    print("Number of epochs: ", n_epochs)
    print("Number of n_samples: ", n_samples)
    
    n_input = x_train.shape[1]
    n_hidden = 10
    n_output = 1

    n_training_points = x_train.shape[0]
    n_test_points = x_test.shape[0]
    n_val_points = x_val.shape[0]

    batch_size = n_training_points

    n_batch = int(n_training_points/batch_size)

    # Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
    X = tf.placeholder(tf.float32, shape = [None, n_input], name = "x_placeholder")
    Y = tf.placeholder(tf.int32, [None,n_output])

    W_fc1 = Normal(loc=tf.zeros([n_input,n_hidden]), scale=tf.ones([n_input,n_hidden]), name="W_fc1")
    b_fc1 = Normal(loc=tf.zeros([n_hidden]), scale=tf.ones([n_hidden]), name="b_fc1")
    W_fc2 = Normal(loc=tf.zeros([n_hidden, n_hidden]), scale=tf.ones([n_hidden, n_hidden]), name="W_fc2")
    b_fc2 = Normal(loc=tf.zeros([n_hidden]), scale=tf.ones([n_hidden]), name="b_fc2")
    W_fc3 = Normal(loc=tf.zeros([n_hidden, n_hidden]), scale=tf.ones([n_hidden, n_hidden]), name="W_fc3")
    b_fc3 = Normal(loc=tf.zeros([n_hidden]), scale=tf.ones([n_hidden]), name="b_fc3")
    W_fc4 = Normal(loc=tf.zeros([n_hidden, n_output]), scale=tf.ones([n_hidden, n_output]), name="W_fc4")
    b_fc4 = Normal(loc=tf.zeros([n_output]), scale=tf.ones([n_output]), name="b_fc4")
    h_fc1 = tf.tanh(tf.matmul(X,W_fc1)+b_fc1)
    h_fc2 = tf.tanh(tf.matmul(h_fc1,W_fc2)+b_fc2)
    h_fc3 = tf.tanh(tf.matmul(h_fc2,W_fc3)+b_fc3)
    hy = tf.sigmoid(tf.matmul(h_fc3,W_fc4)+b_fc4)

    y = Bernoulli(hy)

    qW_fc1 = Normal(loc=tf.Variable(tf.random_normal([n_input, n_hidden])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_input, n_hidden])))) 
    qb_fc1 = Normal(loc=tf.Variable(tf.random_normal([n_hidden])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
    qW_fc2 = Normal(loc = tf.Variable(tf.random_normal([n_hidden, n_hidden])), scale= tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, n_hidden])))) 
    qb_fc2 = Normal(loc=tf.Variable(tf.random_normal([n_hidden])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
    qW_fc3 = Normal(loc = tf.Variable(tf.random_normal([n_hidden, n_hidden])), scale= tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, n_hidden]))))
    qb_fc3 = Normal(loc=tf.Variable(tf.random_normal([n_hidden])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
    qW_fc4 = Normal(loc = tf.Variable(tf.random_normal([n_hidden, n_output])), scale= tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, n_output]))))
    qb_fc4 = Normal(loc=tf.Variable(tf.random_normal([n_output])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_output]))))

    # Define the VI inference technique, ie. minimise the KL divergence between q and p.
    # KLqp - for VI

    start_orig = time.time()

    inference = ed.KLqp({W_fc1: qW_fc1, b_fc1: qb_fc1, W_fc2: qW_fc2, b_fc2: qb_fc2, W_fc3: qW_fc3, b_fc3: qb_fc3, W_fc4: qW_fc4, b_fc4: qb_fc4}, data={y: Y})
    # Initialse the infernce variables
    inference.initialize(n_iter=n_batch * n_epochs, scale={y: n_training_points / batch_size}, n_print=200)
    # We will use an interactive session.
    sess = tf.InteractiveSession()
    # Initialise all the vairables in the session.
    tf.global_variables_initializer().run()

    y_train = y_train.reshape((n_training_points,n_output))
    y_test = y_test.reshape((n_test_points,n_output))
    y_val = y_val.reshape((n_val_points,n_output))

    # train original bnn
    for j in range(inference.n_iter):
    	info_dict = inference.update(feed_dict={X: x_train, Y: y_train})
    	inference.print_progress(info_dict)

    bnn_orig = (qW_fc1, qb_fc1, qW_fc2, qb_fc2, qW_fc3, qb_fc3, qW_fc4, qb_fc4)
    mass = True
    if mass:
        val_avg_probs, val_std_probs, val_avg_pred_class, val_pred_probs = compute_avg_std_pred_probs_with_ECDF(sess, x_val, n_samples, bnn_orig)
        test_avg_probs, test_std_probs, test_avg_pred_class, test_pred_probs = compute_avg_std_pred_probs_with_ECDF(sess, x_test, n_samples, bnn_orig)
    else:
        val_avg_probs, val_std_probs, val_avg_pred_class = compute_avg_std_pred_probs(sess, x_val, n_samples, bnn_orig)
        test_avg_probs, test_std_probs, test_avg_pred_class = compute_avg_std_pred_probs(sess, x_test, n_samples, bnn_orig)
    val_unc = np.vstack((val_avg_probs, val_std_probs))
    test_unc = np.vstack((test_avg_probs, test_std_probs))

    if mass:
        val_error_indexes = label_correct_incorrect_pred(np.round(val_pred_probs), y_val) 
        test_error_indexes = label_correct_incorrect_pred(np.round(test_pred_probs), y_test) 
    else:
        val_error_indexes = label_correct_incorrect_pred(val_avg_pred_class, y_val)
        test_error_indexes = label_correct_incorrect_pred(test_avg_pred_class, y_test)
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
    start_thres = time.time()
    grid_size = 21
    DT_vector = np.linspace(0.95,0.999,grid_size)
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
    print("Time for finding optimal threshold: ", time.time()-start_thres)
    
    fig = plt.figure()
    plt.plot(FPR, TPR, 'bo-')
    plt.plot(FPR[opt_index], TPR[opt_index], 'ro')
    plt.title("GPC ROC curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    fig.savefig("Plots/"+kernelstring+"_orig_ROC_curve.png")
    #-----


    gpc_test_pred_opt = apply_gpc_rejection_rule(gpc_orig, test_unc, orig_optimal_threshold)

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
    
    print("Time original phase: ", time.time()-start_orig)
    
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("\n")

    print("::: Active learning starts ::::")
    # retrain
    start_active = time.time()

    n_points_retrain = 100000
    x_samp_retrain, samp_retrain_real_class, samp_retrain_unc, samp_retrain_avg_pred_class = active_label_query(sess, "GPC", n_points_retrain, scaler, bnn_orig, gpc_orig, n_samples, orig_optimal_threshold)
    n_active_points = len(samp_retrain_real_class)

    x_add_train, y_add_train, x_add_val, y_add_val = split_train_calibration(x_samp_retrain,samp_retrain_real_class, split_rate = split_rate)

    x_retrain = np.vstack((x_train, x_add_train))
    y_retrain = np.vstack((y_train, y_add_train.reshape((len(y_add_train),n_output))))

    x_reval = np.vstack((x_val, x_add_val))
    y_reval = np.vstack((y_val, y_add_val.reshape((len(y_add_val),n_output))))

    # retrain bnn
    print("Retraining the BNN...")
    sess = tf.InteractiveSession()
    # Initialise all the vairables in the session.
    tf.global_variables_initializer().run()
    for j in range(inference.n_iter):
        info_dict = inference.update(feed_dict={X: x_retrain, Y: y_retrain})
        inference.print_progress(info_dict)

    bnn_retrained = (qW_fc1, qb_fc1, qW_fc2, qb_fc2, qW_fc3, qb_fc3, qW_fc4, qb_fc4)

    reval_avg_probs_active, reval_std_probs_active, reval_avg_pred_class_active = compute_avg_std_pred_probs(sess, x_reval, n_samples, bnn_retrained)
    reval_unc_active = np.vstack((reval_avg_probs_active, reval_std_probs_active))
    reval_error_indexes_active = label_correct_incorrect_pred(reval_avg_pred_class_active, y_reval)

    # retrain the rejection rule as well (as the uncertainty values depend on the BNN)
    gpc_active = train_gpc_rejection_rule(kernel_gp, reval_unc_active, reval_error_indexes_active)

    test_avg_probs_retrained, test_std_probs_retrained, test_avg_pred_class_retrained = compute_avg_std_pred_probs(sess, x_test, n_samples, bnn_retrained)
    test_error_indexes_retrained = label_correct_incorrect_pred(test_avg_pred_class_retrained, y_test)
    print("Test avg pred accuracy RETRAINED: ", np.sum((test_error_indexes_retrained==1))/n_test_points)

    reval_n_errors = np.sum(reval_error_indexes_active==-1)
    reval_n_corrects = np.sum(reval_error_indexes_active==1)
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
        TPR_act[j] = n_recogn_errors/reval_n_errors
        FPR_act[j] = n_wrongly_rej/reval_n_corrects

    roc_diff_act = TPR_act-FPR_act
    opt_index_act = np.argmax(roc_diff_act)
    active_optimal_threshold = DT_vector[opt_index_act]

    print("Active TPR-FPR = ", roc_diff_act)
    print("Optimal active threshold = ", active_optimal_threshold)
    fig = plt.figure()
    plt.plot(FPR_act, TPR_act, 'bo-')
    plt.plot(FPR_act[opt_index_act], TPR_act[opt_index_act], 'ro')
    plt.title("GPC ROC curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    fig.savefig("Plots/"+kernelstring+"_active_ROC_curve.png")
    #-----



    test_unc_retrained = np.vstack((test_avg_probs_retrained, test_std_probs_retrained))
    gpc_test_pred_active_opt = apply_gpc_rejection_rule(gpc_active, test_unc_retrained, active_optimal_threshold)

    # active performances
    gpc_accuracy_active = np.sum((gpc_test_pred_active_opt==test_error_indexes_retrained))/n_test_points
    n_test_errors_active = np.sum((test_error_indexes_retrained==-1))

    n_rej_points_active = np.sum((gpc_test_pred_active_opt==-1))
    print("Rejection rate ACTIVE: ", 100*n_rej_points_active/n_test_points, "%")
    print("GPC accuracy ACTIVE: ", gpc_accuracy_active)

    recognized_act = 0
    for i in range(n_test_points):
        if test_error_indexes_retrained[i] == -1 and gpc_test_pred_active_opt[i] == -1:
            recognized_act += 1

    print("Recognized errors ACTIVE: {}/{}".format(recognized_act, n_test_errors_active))
    print("Time active phase: ", time.time()-start_active)
    

    passive_comparison = True
    if passive_comparison:
        start_passive = time.time()
        x_samp_retrain_pass, samp_retrain_real_class_pass = passive_label_query(n_active_points, scaler)
    
        x_add_train_pass, y_add_train_pass, x_add_val_pass, y_add_val_pass = split_train_calibration(x_samp_retrain_pass, samp_retrain_real_class_pass, split_rate = split_rate)

        x_retrain_pass = np.vstack((x_train, x_add_train_pass))
        y_retrain_pass = np.vstack((y_train, y_add_train_pass.reshape((len(y_add_train),n_output))))

        x_reval_pass = np.vstack((x_val, x_add_val_pass))
        y_reval_pass = np.vstack((y_val, y_add_val_pass.reshape((len(y_add_val),n_output))))

        # retrain bnn
        print("Retraining the BNN PASSIVE...")
        sess = tf.InteractiveSession()
        # Initialise all the vairables in the session.
        tf.global_variables_initializer().run()
        for j in range(inference.n_iter):
            info_dict = inference.update(feed_dict={X: x_retrain_pass, Y: y_retrain_pass})
            inference.print_progress(info_dict)

        bnn_retrained_pass = (qW_fc1, qb_fc1, qW_fc2, qb_fc2, qW_fc3, qb_fc3, qW_fc4, qb_fc4)

        reval_avg_probs_passive, reval_std_probs_passive, reval_avg_pred_class_passive = compute_avg_std_pred_probs(sess, x_reval_pass, n_samples, bnn_retrained_pass)
        reval_unc_passive = np.vstack((reval_avg_probs_passive, reval_std_probs_passive))
        reval_error_indexes_passive = label_correct_incorrect_pred(reval_avg_pred_class_passive, y_reval_pass)

        # retrain the rejection rule as well (as the uncertainty values depend on the BNN)
        gpc_passive = train_gpc_rejection_rule(kernel_gp, reval_unc_passive, reval_error_indexes_passive)

        test_avg_probs_retrained_pass, test_std_probs_retrained_pass, test_avg_pred_class_retrained_pass = compute_avg_std_pred_probs(sess, x_test, n_samples, bnn_retrained_pass)
        test_error_indexes_retrained_pass = label_correct_incorrect_pred(test_avg_pred_class_retrained_pass, y_test)
        print("Test avg pred accuracy RETRAINED PASSIVE: ", np.sum((test_error_indexes_retrained_pass==1))/n_test_points)

        reval_n_errors_pass = np.sum(reval_error_indexes_passive==-1)
        reval_n_corrects_pass = np.sum(reval_error_indexes_passive==1)
        # Find the optimal threshold value ORIGINAL

        TPR_pass = np.zeros(grid_size)
        FPR_pass = np.zeros(grid_size)
        for j, dt_j in enumerate(DT_vector):
            gpc_val_pred_pass = apply_gpc_rejection_rule(gpc_passive, reval_unc_passive, dt_j)
            n_recogn_errors = 0
            n_wrongly_rej = 0
            for jj in range(len(gpc_val_pred_pass)):
                if gpc_val_pred_pass[jj] == -1:
                    if reval_error_indexes_passive[jj] == 1:
                        n_wrongly_rej += 1
                    else:
                        n_recogn_errors += 1
            TPR_pass[j] = n_recogn_errors/reval_n_errors_pass
            FPR_pass[j] = n_wrongly_rej/reval_n_corrects_pass

        roc_diff_pass = TPR_pass-FPR_pass
        opt_index_pass = np.argmax(roc_diff_pass)
        passive_optimal_threshold = DT_vector[opt_index_pass]

        print("Passive TPR-FPR = ", roc_diff_pass)
        print("Optimal passive threshold = ", passive_optimal_threshold)
        fig = plt.figure()
        plt.plot(FPR_pass, TPR_pass, 'bo-')
        plt.plot(FPR_pass[opt_index_pass], TPR_pass[opt_index_pass], 'ro')
        plt.title("GPC ROC curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        fig.savefig("Plots/"+kernelstring+"_passive_ROC_curve.png")
        #-----

        test_unc_retrained_pass = np.vstack((test_avg_probs_retrained_pass, test_std_probs_retrained_pass))
        gpc_test_pred_passive_opt = apply_gpc_rejection_rule(gpc_passive, test_unc_retrained_pass, passive_optimal_threshold)

        # passive performances
        gpc_accuracy_passive = np.sum((gpc_test_pred_passive_opt==test_error_indexes_retrained_pass))/n_test_points
        n_test_errors_passive = np.sum((test_error_indexes_retrained_pass==-1))

        n_rej_points_passive = np.sum((gpc_test_pred_passive_opt==-1))
        print("Rejection rate PASSIVE: ", 100*n_rej_points_passive/n_test_points, "%")
        print("GPC accuracy PASSIVE: ", gpc_accuracy_passive)

        recognized_pass = 0
        for i in range(n_test_points):
            if test_error_indexes_retrained_pass[i] == -1 and gpc_test_pred_passive_opt[i] == -1:
                recognized_pass += 1

        print("Recognized errors PASSIVE: {}/{}".format(recognized_pass, n_test_errors_passive))
        print("Time passive phase: ", time.time()-start_passive)
    
    return 0

print("split_rate = ", SPLIT_RATE)

number_trials = 1
for i in range(number_trials):
    
    print("VI+GP active: Results of trial number ", i)

    run_active(SPLIT_RATE)
